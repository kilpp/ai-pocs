"""Block Kit builders for the task hand-off flow.

`build_task_card` is the message posted to the task channel — change it to
reshape how incoming requests are presented to whoever picks them up.
"""

from tasks.models import TaskSpec
from tasks.pipeline import DONE, FAILED, RUNNING, SKIPPED, Progress

STEP_ICONS = {
    "pending": ":white_circle:",
    RUNNING: ":hourglass_flowing_sand:",
    DONE: ":white_check_mark:",
    FAILED: ":x:",
    SKIPPED: ":fast_forward:",
}


def build_task_card(
    task_id: str,
    spec: TaskSpec,
    source_link: str,
    problems: list[str] | None = None,
    missing_info: list[str] | None = None,
    assignee: str | None = None,
) -> list[dict]:
    """The task message posted to the task channel."""
    fields = []
    if spec.requester:
        fields.append({"type": "mrkdwn", "text": f"*Requested by*\n{spec.requester}"})
    if spec.repo:
        fields.append({"type": "mrkdwn", "text": f"*Repo*\n`{spec.repo}`"})

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"📋 {spec.title}"[:150]},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": spec.summary or "_No summary_"},
        },
    ]
    if fields:
        blocks.append({"type": "section", "fields": fields})

    plan: list[str] = []
    if spec.edits:
        plan.append("*1. Open PR* (as the person who picks this up)")
        plan += [f"   • `{e.path}`: `{e.key}` → `{e.value}`" for e in spec.edits]
    if spec.rollouts:
        n = 2 if spec.edits else 1
        plan.append(f"*{n}. Restart Argo Rollouts* (after confirmation)")
        plan += [f"   • `{r.label()}`" for r in spec.rollouts]
    if plan:
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(plan)}}
        )

    if problems:
        text = ":warning: *Automation disabled — handle manually:*\n" + "\n".join(
            f"• {p}" for p in problems
        )
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})
    if missing_info:
        text = ":grey_question: *Missing info:* " + "; ".join(missing_info)
        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}
        )

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"<{source_link}|Original message> · Task `{task_id}`",
                }
            ],
        }
    )

    if assignee:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f":raising_hand: Picked up by <@{assignee}>",
                    }
                ],
            }
        )
    else:
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "action_id": "task_pick_up",
                        "text": {"type": "plain_text", "text": "🙋 Pick up"},
                        "style": "primary",
                        "value": task_id,
                    }
                ],
            }
        )
    return blocks


def build_progress_blocks(title: str, progress: Progress) -> list[dict]:
    lines = []
    for step in progress.steps:
        line = f"{STEP_ICONS.get(step.status, '')} {step.name}"
        if step.detail:
            line += f" — {step.detail}"
        lines.append(line)
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}},
    ]


def build_restart_prompt_blocks(
    task_id: str, spec: TaskSpec, assignee: str
) -> list[dict]:
    targets = "\n".join(f"• `{r.label()}`" for r in spec.rollouts)
    prefix = "Once the PR is merged and Argo has synced, " if spec.edits else ""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"<@{assignee}> {prefix}confirm to restart:\n{targets}",
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "action_id": "task_restart_rollouts",
                    "text": {"type": "plain_text", "text": "🔄 Restart rollouts"},
                    "style": "danger",
                    "value": task_id,
                    "confirm": {
                        "title": {"type": "plain_text", "text": "Restart rollouts?"},
                        "text": {"type": "mrkdwn", "text": targets},
                        "confirm": {"type": "plain_text", "text": "Restart"},
                        "deny": {"type": "plain_text", "text": "Cancel"},
                    },
                }
            ],
        },
    ]
