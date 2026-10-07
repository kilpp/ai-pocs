from slack_bolt.async_app import AsyncApp

from .link_github import handle_link_github


def register(app: AsyncApp):
    app.command("/link-github")(handle_link_github)
