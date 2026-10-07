import re

from slack_bolt.async_app import AsyncApp

from .feedback_buttons import handle_feedback_button
from .issue_buttons import handle_issue_button
from .task_buttons import handle_task_pick_up, handle_task_restart_rollouts


def register(app: AsyncApp):
    app.action(re.compile(r"^category_"))(handle_issue_button)
    app.action("feedback")(handle_feedback_button)
    app.action("task_pick_up")(handle_task_pick_up)
    app.action("task_restart_rollouts")(handle_task_restart_rollouts)
