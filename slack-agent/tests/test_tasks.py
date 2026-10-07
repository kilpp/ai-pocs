import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from cryptography.fernet import Fernet
from slack_bolt.context.async_context import AsyncBoltContext
from slack_sdk.web.async_client import AsyncWebClient

from listeners.actions.task_buttons import handle_task_pick_up
from listeners.events.source_message import handle_source_message
from tasks.config import TaskSettings
from tasks.models import FileEdit, RolloutTarget, TaskSpec
from tasks.pipeline import DONE, FAILED, SKIPPED, Progress, pr_steps, run_pr_steps
from tasks.store import STATUS_AWAITING_RESTART, TaskStore
from tasks.yaml_edit import apply_yaml_edits

test_logger = logging.getLogger(__name__)

SETTINGS = TaskSettings(
    source_channel_id="CSRC",
    task_channel_id="CTASK",
    allowed_repos=frozenset({"acme/gitops"}),
    allowed_kube_contexts=frozenset({"prod"}),
    token_encryption_key=Fernet.generate_key().decode(),
    db_path=":memory:",
)


def make_spec(**overrides) -> TaskSpec:
    spec = TaskSpec(
        title="Bump api to 1.4.2",
        summary="Deploy api 1.4.2 to prod",
        repo="acme/gitops",
        edits=[FileEdit(path="apps/api/values.yaml", key="image.tag", value="1.4.2")],
        rollouts=[RolloutTarget(context="prod", namespace="api", name="api")],
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def make_store() -> TaskStore:
    return TaskStore(":memory:", SETTINGS.token_encryption_key)


class TestValidation:
    def test_valid_spec(self):
        assert make_spec().validate(SETTINGS) == []

    def test_rejects_repo_and_context_outside_allowlist(self):
        spec = make_spec(
            repo="evil/repo",
            rollouts=[RolloutTarget(context="other", namespace="api", name="api")],
        )
        problems = spec.validate(SETTINGS)
        assert any("evil/repo" in p for p in problems)
        assert any("other" in p for p in problems)

    def test_rejects_path_traversal(self):
        spec = make_spec(edits=[FileEdit(path="../x.yaml", key="a", value="b")])
        assert spec.validate(SETTINGS)


class TestYamlEdit:
    def test_sets_nested_key_and_keeps_comments(self):
        text = "# api values\nimage:\n  repository: api  # repo\n  tag: 1.4.1\n"
        out = apply_yaml_edits(
            text, [FileEdit(path="v.yaml", key="image.tag", value="1.4.2")]
        )
        assert out == "# api values\nimage:\n  repository: api  # repo\n  tag: 1.4.2\n"

    def test_unknown_key_fails(self):
        with pytest.raises(KeyError):
            apply_yaml_edits(
                "image:\n  tag: 1\n", [FileEdit(path="v.yaml", key="image.tg", value=2)]
            )


class TestStore:
    def test_claim_is_first_come_first_served(self):
        store = make_store()
        task_id = store.create_pending("CSRC", "1.0")
        store.update(task_id, spec=make_spec(), status="open")
        assert store.claim(task_id, "U1") is True
        assert store.claim(task_id, "U2") is False
        assert store.get(task_id).assignee == "U1"

    def test_duplicate_source_message_is_ignored(self):
        store = make_store()
        assert store.create_pending("CSRC", "1.0") is not None
        assert store.create_pending("CSRC", "1.0") is None

    def test_github_token_is_encrypted_at_rest(self):
        store = make_store()
        store.save_github_token("U1", "octocat", "gho_secret")
        raw = store._conn.execute("SELECT token FROM github_tokens").fetchone()[0]
        assert b"gho_secret" not in raw
        assert store.get_github_token("U1") == ("octocat", "gho_secret")


def fake_github(login="octocat"):
    gh = Mock()
    gh.get_login = AsyncMock(return_value=login)
    gh.get_default_branch = AsyncMock(return_value="main")
    gh.get_branch_sha = AsyncMock(return_value="abc")
    gh.create_branch = AsyncMock()
    gh.get_file = AsyncMock(return_value=("image:\n  tag: 1.4.1\n", "sha1"))
    gh.update_file = AsyncMock()
    gh.create_pull_request = AsyncMock(
        return_value="https://github.com/acme/gitops/pull/7"
    )
    return gh


class TestPipeline:
    @pytest.mark.asyncio
    async def test_opens_pr_as_user(self):
        spec = make_spec()
        gh = fake_github()
        progress = Progress(steps=pr_steps(spec), on_change=AsyncMock())

        pr_url = await run_pr_steps(
            progress,
            spec=spec,
            task_id="t1",
            github=gh,
            expected_login="octocat",
            source_link="https://slack/x",
        )

        assert pr_url.endswith("/pull/7")
        assert all(s.status == DONE for s in progress.steps)
        _, _, branch, text, _, _ = gh.update_file.call_args.args
        assert branch.startswith("task/t1-")
        assert "tag: 1.4.2" in text

    @pytest.mark.asyncio
    async def test_token_for_wrong_user_stops_pipeline(self):
        spec = make_spec()
        gh = fake_github(login="someone-else")
        progress = Progress(steps=pr_steps(spec), on_change=AsyncMock())

        with pytest.raises(RuntimeError):
            await run_pr_steps(
                progress,
                spec=spec,
                task_id="t1",
                github=gh,
                expected_login="octocat",
                source_link="x",
            )
        assert progress.steps[0].status == FAILED
        assert all(s.status == SKIPPED for s in progress.steps[1:])
        gh.create_branch.assert_not_called()


def fake_client():
    client = Mock(AsyncWebClient)
    client.chat_postMessage = AsyncMock(return_value={"channel": "CTASK", "ts": "2.0"})
    client.chat_update = AsyncMock()
    client.chat_postEphemeral = AsyncMock()
    client.chat_getPermalink = AsyncMock(return_value={"permalink": "https://slack/p"})
    client.reactions_add = AsyncMock()
    return client


class TestSourceMessage:
    @pytest.mark.asyncio
    async def test_task_message_posts_card_to_task_channel(self):
        store, client = make_store(), fake_client()
        context = Mock(AsyncBoltContext)
        context.bot_id = "BME"
        with (
            patch(
                "listeners.events.source_message.get_settings", return_value=SETTINGS
            ),
            patch("listeners.events.source_message.get_task_store", return_value=store),
            patch(
                "listeners.events.source_message.classify_message",
                AsyncMock(return_value=(make_spec(), [])),
            ),
        ):
            await handle_source_message(
                client=client,
                context=context,
                event={"channel": "CSRC", "ts": "1.0", "text": "please bump api"},
                logger=test_logger,
            )

        card_call = client.chat_postMessage.call_args_list[0].kwargs
        assert card_call["channel"] == "CTASK"
        assert any(b.get("type") == "actions" for b in card_call["blocks"])

    @pytest.mark.asyncio
    async def test_non_task_is_ignored(self):
        store, client = make_store(), fake_client()
        context = Mock(AsyncBoltContext)
        context.bot_id = "BME"
        with (
            patch(
                "listeners.events.source_message.get_settings", return_value=SETTINGS
            ),
            patch("listeners.events.source_message.get_task_store", return_value=store),
            patch(
                "listeners.events.source_message.classify_message",
                AsyncMock(return_value=None),
            ),
        ):
            await handle_source_message(
                client=client,
                context=context,
                event={"channel": "CSRC", "ts": "1.0", "text": "thanks all!"},
                logger=test_logger,
            )
        client.chat_postMessage.assert_not_called()


class TestPickUp:
    def _setup(self):
        store, client = make_store(), fake_client()
        task_id = store.create_pending("CSRC", "1.0")
        store.update(
            task_id,
            spec=make_spec(),
            status="open",
            task_channel="CTASK",
            task_ts="2.0",
        )
        body = {
            "actions": [{"value": task_id}],
            "user": {"id": "U1"},
            "channel": {"id": "CTASK"},
            "message": {"ts": "2.0"},
        }
        return store, client, task_id, body

    @pytest.mark.asyncio
    async def test_requires_linked_github_before_claiming(self):
        store, client, task_id, body = self._setup()
        with (
            patch("listeners.actions.task_buttons.get_settings", return_value=SETTINGS),
            patch("listeners.actions.task_buttons.get_task_store", return_value=store),
        ):
            await handle_task_pick_up(
                ack=AsyncMock(), body=body, client=client, logger=test_logger
            )

        assert "/link-github" in client.chat_postEphemeral.call_args.kwargs["text"]
        assert store.get(task_id).assignee is None

    @pytest.mark.asyncio
    async def test_opens_pr_and_waits_for_restart_confirmation(self):
        store, client, task_id, body = self._setup()
        store.save_github_token("U1", "octocat", "gho_token")
        gh = fake_github()
        with (
            patch("listeners.actions.task_buttons.get_settings", return_value=SETTINGS),
            patch("listeners.actions.task_buttons.get_task_store", return_value=store),
            patch("listeners.actions.task_buttons.GitHubClient", return_value=gh),
        ):
            await handle_task_pick_up(
                ack=AsyncMock(), body=body, client=client, logger=test_logger
            )

        record = store.get(task_id)
        assert record.assignee == "U1"
        assert record.status == STATUS_AWAITING_RESTART
        assert record.pr_url.endswith("/pull/7")
        gh.create_pull_request.assert_called_once()
