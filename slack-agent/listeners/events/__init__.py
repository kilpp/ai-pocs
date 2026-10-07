from slack_bolt.async_app import AsyncApp

from .app_home_opened import handle_app_home_opened
from .app_mentioned import handle_app_mentioned
from .message import handle_message
from .source_message import handle_source_message, is_source_channel_message


def register(app: AsyncApp):
    app.event("app_home_opened")(handle_app_home_opened)
    app.event("app_mention")(handle_app_mentioned)
    # Must be registered before the generic message handler: Bolt runs only
    # the first matching listener for an event.
    app.event("message", matchers=[is_source_channel_message])(handle_source_message)
    app.event("message")(handle_message)
