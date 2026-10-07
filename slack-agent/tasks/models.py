import re
from dataclasses import asdict, dataclass, field
from typing import Any

from tasks.config import TaskSettings

# Kubernetes DNS-1123 names (namespaces, rollouts)
_K8S_NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REPO_PATH = re.compile(r"^[A-Za-z0-9_./-]+\.ya?ml$")
_YAML_KEY = re.compile(r"^[A-Za-z0-9_\-]+(\.[A-Za-z0-9_\-]+)*$")


@dataclass
class FileEdit:
    """Set a dotted key (e.g. ``image.tag``) in a YAML file of the repo."""

    path: str
    key: str
    value: str | int | float | bool


@dataclass
class RolloutTarget:
    context: str
    namespace: str
    name: str

    def label(self) -> str:
        return f"{self.context}/{self.namespace}/{self.name}"


@dataclass
class TaskSpec:
    title: str
    summary: str
    requester: str = ""
    repo: str = ""
    base_branch: str = ""
    edits: list[FileEdit] = field(default_factory=list)
    rollouts: list[RolloutTarget] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        return cls(
            title=str(data.get("title", "")).strip(),
            summary=str(data.get("summary", "")).strip(),
            requester=str(data.get("requester") or "").strip(),
            repo=str(data.get("repo") or "").strip(),
            base_branch=str(data.get("base_branch") or "").strip(),
            edits=[FileEdit(**e) for e in data.get("edits") or []],
            rollouts=[RolloutTarget(**r) for r in data.get("rollouts") or []],
        )

    def validate(self, settings: TaskSettings) -> list[str]:
        """Return a list of problems that block automation (empty = safe to run)."""
        errors: list[str] = []
        if not self.title:
            errors.append("Missing title")

        if self.edits:
            if not self.repo:
                errors.append("File edits requested but no repo given")
            elif not _REPO.match(self.repo):
                errors.append(f"Invalid repo name `{self.repo}`")
            elif self.repo not in settings.allowed_repos:
                errors.append(f"Repo `{self.repo}` is not in TASK_ALLOWED_REPOS")
        for edit in self.edits:
            if ".." in edit.path or not _REPO_PATH.match(edit.path):
                errors.append(f"Invalid YAML file path `{edit.path}`")
            if not _YAML_KEY.match(edit.key):
                errors.append(f"Invalid YAML key `{edit.key}`")

        for r in self.rollouts:
            if r.context not in settings.allowed_kube_contexts:
                errors.append(
                    f"Kube context `{r.context}` is not in TASK_ALLOWED_KUBE_CONTEXTS"
                )
            if not _K8S_NAME.match(r.namespace) or not _K8S_NAME.match(r.name):
                errors.append(f"Invalid rollout reference `{r.label()}`")

        if not self.edits and not self.rollouts:
            errors.append("Nothing to automate (no file edits and no rollouts)")
        return errors
