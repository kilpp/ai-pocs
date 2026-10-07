from logging import Logger

from slack_bolt import Ack
from slack_bolt.context.respond.async_respond import AsyncRespond

from tasks import get_settings, get_task_store
from tasks.github import GitHubClient, GitHubDeviceFlow


async def handle_link_github(
    ack: Ack, command: dict, respond: AsyncRespond, logger: Logger
):
    """`/link-github` links the caller's GitHub account via the OAuth device flow.

    `/link-github unlink` removes the stored token.
    """
    await ack()

    settings = get_settings()
    store = get_task_store()
    user_id = command["user_id"]

    if command.get("text", "").strip() == "unlink":
        store.delete_github_token(user_id)
        await respond(text=":wave: Your GitHub account has been unlinked.")
        return

    if not settings.github_client_id or not settings.token_encryption_key:
        await respond(
            text=":warning: GitHub linking isn't configured "
            "(GITHUB_OAUTH_CLIENT_ID / TOKEN_ENCRYPTION_KEY)."
        )
        return

    try:
        flow = GitHubDeviceFlow(settings.github_client_id, settings.github_oauth_url)
        code = await flow.start()
        await respond(
            text=(
                f":key: Open {code.verification_uri} and enter the code "
                f"*`{code.user_code}`* (expires in {code.expires_in // 60} min). "
                "I'll confirm here once it's done."
            )
        )
        token = await flow.poll_for_token(code)
        login = await GitHubClient(token, settings.github_api_url).get_login()
        store.save_github_token(user_id, login, token)
        await respond(
            text=f":white_check_mark: Linked to GitHub as *{login}*. "
            "PRs for tasks you pick up will be opened as you."
        )
    except Exception as e:
        logger.exception("GitHub linking failed")
        await respond(text=f":x: GitHub linking failed: {e}")
