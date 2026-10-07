import dataclasses
import re
from functools import lru_cache

from slack_sdk.web.async_client import AsyncWebClient

from .config import TaskSettings
from .store import TaskStore

_CHANNEL_ID = re.compile(r"^[CG][A-Z0-9]{8,}$")
_settings: TaskSettings | None = None


# Lazy so that .env (loaded in app.py after listeners are imported) is applied.
def get_settings() -> TaskSettings:
    global _settings
    if _settings is None:
        _settings = TaskSettings.from_env()
    return _settings


@lru_cache(maxsize=1)
def get_task_store() -> TaskStore:
    settings = get_settings()
    return TaskStore(settings.db_path, settings.token_encryption_key)


async def resolve_channel_names(client: AsyncWebClient) -> TaskSettings:
    """Allow channels to be configured by name (e.g. ``all-gk``) instead of ID.

    Slack events only carry channel IDs, so names are resolved once at startup.
    """
    global _settings
    settings = get_settings()
    wanted = {
        field: value.lstrip("#")
        for field in ("source_channel_id", "task_channel_id")
        if (value := getattr(settings, field)) and not _CHANNEL_ID.match(value)
    }
    if not wanted:
        return settings

    resolved: dict[str, str] = {}
    cursor = None
    while True:
        resp = await client.conversations_list(
            types="public_channel,private_channel",
            exclude_archived=True,
            limit=1000,
            cursor=cursor,
        )
        for channel in resp["channels"]:
            for field, name in wanted.items():
                if channel["name"] == name:
                    resolved[field] = channel["id"]
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor or len(resolved) == len(wanted):
            break

    missing = [f"#{name}" for field, name in wanted.items() if field not in resolved]
    if missing:
        raise RuntimeError(
            f"Channel(s) not found or not visible to the bot: {', '.join(missing)}"
        )
    _settings = dataclasses.replace(settings, **resolved)
    return _settings


__all__ = [
    "TaskSettings",
    "TaskStore",
    "get_settings",
    "get_task_store",
    "resolve_channel_names",
]
