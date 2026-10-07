import os
from dataclasses import dataclass, field


def _csv(name: str) -> frozenset[str]:
    raw = os.environ.get(name, "")
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class TaskSettings:
    """Configuration for the task hand-off workflow, read from the environment."""

    # Channel watched for incoming requests and channel where tasks are posted
    source_channel_id: str = ""
    task_channel_id: str = ""

    # Safety allowlists: the classifier output is untrusted, so the pipeline
    # only touches repos and kube contexts listed here.
    allowed_repos: frozenset[str] = field(default_factory=frozenset)
    allowed_kube_contexts: frozenset[str] = field(default_factory=frozenset)

    # GitHub OAuth App (or GitHub App) client ID with device flow enabled
    github_client_id: str = ""
    github_api_url: str = "https://api.github.com"
    github_oauth_url: str = "https://github.com"

    # Fernet key used to encrypt stored GitHub tokens
    token_encryption_key: str = ""
    db_path: str = "data/tasks.db"

    classifier_model: str | None = None
    kubectl_path: str = "kubectl"

    @property
    def enabled(self) -> bool:
        return bool(self.source_channel_id and self.task_channel_id)

    @classmethod
    def from_env(cls) -> "TaskSettings":
        return cls(
            source_channel_id=os.environ.get("TASK_SOURCE_CHANNEL_ID", ""),
            task_channel_id=os.environ.get("TASK_TARGET_CHANNEL_ID", ""),
            allowed_repos=_csv("TASK_ALLOWED_REPOS"),
            allowed_kube_contexts=_csv("TASK_ALLOWED_KUBE_CONTEXTS"),
            github_client_id=os.environ.get("GITHUB_OAUTH_CLIENT_ID", ""),
            github_api_url=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
            github_oauth_url=os.environ.get("GITHUB_OAUTH_URL", "https://github.com"),
            token_encryption_key=os.environ.get("TOKEN_ENCRYPTION_KEY", ""),
            db_path=os.environ.get("TASKS_DB_PATH", "data/tasks.db"),
            classifier_model=os.environ.get("TASK_CLASSIFIER_MODEL") or None,
            kubectl_path=os.environ.get("KUBECTL_PATH", "kubectl"),
        )
