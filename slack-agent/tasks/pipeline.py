import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from tasks.argo import restart_rollout
from tasks.config import TaskSettings
from tasks.github import GitHubClient
from tasks.models import TaskSpec
from tasks.yaml_edit import apply_yaml_edits

PENDING, RUNNING, DONE, FAILED, SKIPPED = (
    "pending",
    "running",
    "done",
    "failed",
    "skipped",
)


@dataclass
class Step:
    name: str
    status: str = PENDING
    detail: str = ""


@dataclass
class Progress:
    """Ordered step list; ``on_change`` is awaited after every update (e.g. to
    refresh the Slack progress message)."""

    steps: list[Step]
    on_change: Callable[["Progress"], Awaitable[None]]
    pr_url: str | None = None

    async def set(self, index: int, status: str, detail: str = "") -> None:
        self.steps[index].status = status
        self.steps[index].detail = detail
        await self.on_change(self)

    async def skip_remaining(self, reason: str) -> None:
        for step in self.steps:
            if step.status == PENDING:
                step.status, step.detail = SKIPPED, reason
        await self.on_change(self)


def pr_steps(spec: TaskSpec) -> list[Step]:
    if not spec.edits:
        return []
    return [
        Step("Verify GitHub identity"),
        Step(f"Create branch in `{spec.repo}`"),
        Step(f"Commit {len(spec.edits)} YAML change(s)"),
        Step("Open pull request"),
    ]


def restart_steps(spec: TaskSpec) -> list[Step]:
    return [Step(f"Restart rollout `{r.label()}`") for r in spec.rollouts]


def branch_name(task_id: str, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40].rstrip("-")
    return f"task/{task_id}-{slug}" if slug else f"task/{task_id}"


async def run_pr_steps(
    progress: Progress,
    *,
    spec: TaskSpec,
    task_id: str,
    github: GitHubClient,
    expected_login: str,
    source_link: str,
) -> str:
    """Create branch, commit edits and open a PR — all as the token's owner."""
    step = 0
    try:
        await progress.set(step, RUNNING)
        login = await github.get_login()
        if login.lower() != expected_login.lower():
            raise RuntimeError(
                f"Token belongs to `{login}`, expected `{expected_login}`"
            )
        await progress.set(step, DONE, f"acting as `{login}`")

        step = 1
        await progress.set(step, RUNNING)
        base = spec.base_branch or await github.get_default_branch(spec.repo)
        branch = branch_name(task_id, spec.title)
        base_sha = await github.get_branch_sha(spec.repo, base)
        await github.create_branch(spec.repo, branch, base_sha)
        await progress.set(step, DONE, f"`{branch}` from `{base}`")

        step = 2
        await progress.set(step, RUNNING)
        edits_by_path: dict[str, list] = {}
        for edit in spec.edits:
            edits_by_path.setdefault(edit.path, []).append(edit)
        for path, edits in edits_by_path.items():
            text, sha = await github.get_file(spec.repo, path, branch)
            updated = apply_yaml_edits(text, edits)
            if updated == text:
                continue
            keys = ", ".join(f"{e.key}={e.value}" for e in edits)
            await github.update_file(
                spec.repo, path, branch, updated, sha, f"{spec.title}\n\n{path}: {keys}"
            )
        await progress.set(step, DONE, ", ".join(f"`{p}`" for p in edits_by_path))

        step = 3
        await progress.set(step, RUNNING)
        pr_url = await github.create_pull_request(
            spec.repo, branch, base, spec.title, _pr_body(spec, source_link)
        )
        progress.pr_url = pr_url
        await progress.set(step, DONE, pr_url)
        return pr_url
    except Exception as e:
        await progress.set(step, FAILED, str(e))
        await progress.skip_remaining("previous step failed")
        raise


async def run_restart_steps(
    progress: Progress, *, spec: TaskSpec, settings: TaskSettings, offset: int = 0
) -> None:
    """Restart each rollout in order; stop at the first failure."""
    for i, target in enumerate(spec.rollouts):
        step = offset + i
        await progress.set(step, RUNNING)
        try:
            await restart_rollout(target, settings.kubectl_path)
        except Exception as e:
            await progress.set(step, FAILED, str(e))
            await progress.skip_remaining("previous step failed")
            raise
        await progress.set(step, DONE)


def _pr_body(spec: TaskSpec, source_link: str) -> str:
    lines = [spec.summary, "", "### Changes"]
    lines += [f"- `{e.path}`: `{e.key}` → `{e.value}`" for e in spec.edits]
    if spec.rollouts:
        lines += ["", "### Rollouts to restart after merge"]
        lines += [f"- `{r.label()}`" for r in spec.rollouts]
    lines += ["", f"Requested in Slack: {source_link}"]
    return "\n".join(lines)
