from logging import Logger

from slack_bolt import Ack
from slack_sdk.web.async_client import AsyncWebClient

from listeners.views.task_builder import (
    build_progress_blocks,
    build_restart_prompt_blocks,
    build_task_card,
)
from tasks import get_settings, get_task_store
from tasks.github import GitHubClient
from tasks.pipeline import (
    Progress,
    pr_steps,
    restart_steps,
    run_pr_steps,
    run_restart_steps,
)
from tasks.store import (
    STATUS_AWAITING_RESTART,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_MANUAL,
    STATUS_RESTARTING,
)


async def _ephemeral(client: AsyncWebClient, body: dict, text: str) -> None:
    await client.chat_postEphemeral(
        channel=body["channel"]["id"],
        user=body["user"]["id"],
        thread_ts=body["message"].get("thread_ts"),
        text=text,
    )


def _progress_updater(client: AsyncWebClient, channel: str, ts: str, title: str):
    async def on_change(progress: Progress) -> None:
        await client.chat_update(
            channel=channel,
            ts=ts,
            text=title,
            blocks=build_progress_blocks(title, progress),
        )

    return on_change


async def handle_task_pick_up(
    ack: Ack, body: dict, client: AsyncWebClient, logger: Logger
):
    """Assign the task to the clicker and run the PR steps with *their* GitHub token."""
    await ack()

    settings = get_settings()
    store = get_task_store()
    task_id = body["actions"][0]["value"]
    user_id = body["user"]["id"]

    record = store.get(task_id)
    if record is None or record.spec is None:
        await _ephemeral(client, body, ":warning: I can't find this task anymore.")
        return
    spec = record.spec
    problems = spec.validate(settings)

    # PRs must be opened as the assignee, so require a linked GitHub account
    # *before* claiming — otherwise the task would be stuck with them.
    github_identity = store.get_github_token(user_id)
    if spec.edits and not problems and github_identity is None:
        await _ephemeral(
            client,
            body,
            ":link: This task opens a PR in your name, so link your GitHub first: "
            "run `/link-github`, then click *Pick up* again.",
        )
        return

    if not store.claim(task_id, user_id):
        current = store.get(task_id)
        if current and current.assignee == user_id:
            await _ephemeral(client, body, "You already picked this up.")
        else:
            owner = (
                f"<@{current.assignee}>" if current and current.assignee else "someone"
            )
            await _ephemeral(client, body, f"Too slow! {owner} already picked this up.")
        return

    try:
        source_link = (
            await client.chat_getPermalink(
                channel=record.source_channel, message_ts=record.source_ts
            )
        )["permalink"]
        await client.chat_update(
            channel=record.task_channel,
            ts=record.task_ts,
            text=f"Task picked up by <@{user_id}>: {spec.title}",
            blocks=build_task_card(
                task_id, spec, source_link, problems, assignee=user_id
            ),
        )

        if problems:
            store.update(task_id, status=STATUS_MANUAL)
            await client.chat_postMessage(
                channel=record.task_channel,
                thread_ts=record.task_ts,
                text=f"<@{user_id}> it's yours. Automation is disabled for this task "
                "(see the warnings above), so please handle it manually.",
            )
            return

        if spec.edits:
            login, token = github_identity
            title = f"Opening PR as {login}"
            progress_msg = await client.chat_postMessage(
                channel=record.task_channel,
                thread_ts=record.task_ts,
                text=title,
            )
            store.update(task_id, progress_ts=progress_msg["ts"])
            progress = Progress(
                steps=pr_steps(spec),
                on_change=_progress_updater(
                    client, record.task_channel, progress_msg["ts"], title
                ),
            )
            await progress.on_change(progress)
            pr_url = await run_pr_steps(
                progress,
                spec=spec,
                task_id=task_id,
                github=GitHubClient(token, settings.github_api_url),
                expected_login=login,
                source_link=source_link,
            )
            store.update(task_id, pr_url=pr_url)
            await client.chat_postMessage(
                channel=record.source_channel,
                thread_ts=record.source_ts,
                text=f":rocket: <@{user_id}> picked this up and opened {pr_url}",
            )

        if spec.rollouts:
            store.update(task_id, status=STATUS_AWAITING_RESTART)
            await client.chat_postMessage(
                channel=record.task_channel,
                thread_ts=record.task_ts,
                text="Ready to restart rollouts",
                blocks=build_restart_prompt_blocks(task_id, spec, user_id),
            )
        else:
            store.update(task_id, status=STATUS_DONE)

    except Exception as e:
        logger.exception("Task %s failed after pick up", task_id)
        store.update(task_id, status=STATUS_FAILED)
        await client.chat_postMessage(
            channel=record.task_channel,
            thread_ts=record.task_ts,
            text=f":x: <@{user_id}> the automation failed: {e}",
        )


async def handle_task_restart_rollouts(
    ack: Ack, body: dict, client: AsyncWebClient, logger: Logger
):
    """Restart the task's Argo Rollouts after the assignee confirms."""
    await ack()

    settings = get_settings()
    store = get_task_store()
    task_id = body["actions"][0]["value"]
    user_id = body["user"]["id"]

    record = store.get(task_id)
    if record is None or record.spec is None:
        await _ephemeral(client, body, ":warning: I can't find this task anymore.")
        return
    if record.assignee != user_id:
        await _ephemeral(
            client, body, f"Only <@{record.assignee}> (the assignee) can run this step."
        )
        return
    if not store.transition(task_id, STATUS_AWAITING_RESTART, STATUS_RESTARTING):
        await _ephemeral(client, body, "This restart is already running or finished.")
        return

    spec = record.spec
    channel = body["channel"]["id"]
    title = f"Restarting rollouts (triggered by <@{user_id}>)"
    progress = Progress(
        steps=restart_steps(spec),
        # Replace the prompt (and its button) with the live progress
        on_change=_progress_updater(client, channel, body["message"]["ts"], title),
    )
    try:
        await progress.on_change(progress)
        await run_restart_steps(progress, spec=spec, settings=settings)
        store.update(task_id, status=STATUS_DONE)
        await client.chat_postMessage(
            channel=record.source_channel,
            thread_ts=record.source_ts,
            text=f":white_check_mark: Done — rollouts restarted by <@{user_id}>.",
        )
    except Exception:
        logger.exception("Rollout restart failed for task %s", task_id)
        store.update(task_id, status=STATUS_FAILED)
