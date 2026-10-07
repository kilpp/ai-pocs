from logging import Logger

from slack_bolt.context.async_context import AsyncBoltContext
from slack_sdk.web.async_client import AsyncWebClient

from listeners.views.task_builder import build_task_card
from tasks import get_settings, get_task_store
from tasks.classifier import classify_message
from tasks.store import STATUS_FAILED, STATUS_IGNORED, STATUS_OPEN


async def is_source_channel_message(event: dict) -> bool:
    """Matcher: top-level messages in the configured source channel."""
    settings = get_settings()
    return (
        settings.enabled
        and event.get("channel") == settings.source_channel_id
        and event.get("thread_ts") in (None, event.get("ts"))
    )


def _message_text(event: dict) -> str:
    # Workflow / integration posts often carry their content in attachments
    parts = [event.get("text") or ""]
    for attachment in event.get("attachments") or []:
        parts.append(attachment.get("text") or attachment.get("fallback") or "")
    return "\n".join(p for p in parts if p).strip()


async def handle_source_message(
    client: AsyncWebClient,
    context: AsyncBoltContext,
    event: dict,
    logger: Logger,
):
    """Turn task-like messages in the source channel into task cards."""
    # Allow other bots/workflows (subtype bot_message) but skip edits, joins,
    # deletes, etc. and our own posts.
    if event.get("subtype") not in (None, "bot_message"):
        return
    if event.get("bot_id") and event.get("bot_id") == context.bot_id:
        return

    text = _message_text(event)
    if not text:
        return

    settings = get_settings()
    store = get_task_store()
    channel_id = event["channel"]
    message_ts = event["ts"]

    task_id = store.create_pending(channel_id, message_ts)
    if task_id is None:
        return  # already processed (Slack retry)

    try:
        classified = await classify_message(text, settings)
        if classified is None:
            store.update(task_id, status=STATUS_IGNORED)
            return
        spec, missing_info = classified
        problems = spec.validate(settings)

        source_link = (
            await client.chat_getPermalink(channel=channel_id, message_ts=message_ts)
        )["permalink"]

        card = await client.chat_postMessage(
            channel=settings.task_channel_id,
            text=f"New task: {spec.title}",
            blocks=build_task_card(task_id, spec, source_link, problems, missing_info),
            unfurl_links=False,
        )
        store.update(
            task_id,
            spec=spec,
            status=STATUS_OPEN,
            task_channel=card["channel"],
            task_ts=card["ts"],
        )

        card_link = (
            await client.chat_getPermalink(
                channel=card["channel"], message_ts=card["ts"]
            )
        )["permalink"]
        await client.chat_postMessage(
            channel=channel_id,
            thread_ts=message_ts,
            text=f":clipboard: Got it — task `{task_id}` is waiting to be picked up: <{card_link}|view task>",
            unfurl_links=False,
        )
    except Exception:
        logger.exception("Failed to process source channel message")
        store.update(task_id, status=STATUS_FAILED)
        await client.reactions_add(
            channel=channel_id, timestamp=message_ts, name="warning"
        )
