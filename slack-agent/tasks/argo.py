import asyncio
import json
from datetime import UTC, datetime

from tasks.models import RolloutTarget


class ArgoError(Exception):
    pass


async def restart_rollout(target: RolloutTarget, kubectl_path: str = "kubectl") -> str:
    """Restart an Argo Rollout.

    Equivalent to ``kubectl argo rollouts restart``: the plugin just sets
    ``spec.restartAt`` to now, so this works with plain kubectl and no plugin.
    Uses the kubeconfig of the process running the bot.
    """
    restart_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    patch = json.dumps({"spec": {"restartAt": restart_at}})
    proc = await asyncio.create_subprocess_exec(
        kubectl_path,
        "--context", target.context,
        "--namespace", target.namespace,
        "patch", "rollouts.argoproj.io", target.name,
        "--type", "merge",
        "--patch", patch,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    except TimeoutError:
        proc.kill()
        raise ArgoError(f"Timed out restarting {target.label()}")
    if proc.returncode != 0:
        raise ArgoError(stderr.decode().strip() or f"kubectl exited {proc.returncode}")
    return stdout.decode().strip()
