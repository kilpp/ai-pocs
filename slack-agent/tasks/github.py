import asyncio
import base64
from dataclasses import dataclass
from urllib.parse import quote

import aiohttp


class GitHubError(Exception):
    pass


@dataclass
class DeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


class GitHubDeviceFlow:
    """OAuth device flow: works with Socket Mode since no redirect URL is needed.

    Requires an OAuth App (or GitHub App) with "Enable Device Flow" checked.
    """

    def __init__(self, client_id: str, oauth_url: str = "https://github.com"):
        self._client_id = client_id
        self._oauth_url = oauth_url.rstrip("/")

    async def start(self) -> DeviceCode:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self._oauth_url}/login/device/code",
                data={"client_id": self._client_id, "scope": "repo"},
                headers={"Accept": "application/json"},
            ) as resp,
        ):
            data = await resp.json()
        if "device_code" not in data:
            raise GitHubError(f"Device flow failed: {data}")
        return DeviceCode(
            device_code=data["device_code"],
            user_code=data["user_code"],
            verification_uri=data["verification_uri"],
            expires_in=int(data["expires_in"]),
            interval=int(data.get("interval", 5)),
        )

    async def poll_for_token(self, code: DeviceCode) -> str:
        """Poll until the user authorizes (or the code expires)."""
        interval = code.interval
        deadline = asyncio.get_running_loop().time() + code.expires_in
        async with aiohttp.ClientSession() as session:
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(interval)
                async with session.post(
                    f"{self._oauth_url}/login/oauth/access_token",
                    data={
                        "client_id": self._client_id,
                        "device_code": code.device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers={"Accept": "application/json"},
                ) as resp:
                    data = await resp.json()
                if "access_token" in data:
                    return data["access_token"]
                error = data.get("error")
                if error == "authorization_pending":
                    continue
                if error == "slow_down":
                    interval = int(data.get("interval", interval + 5))
                    continue
                raise GitHubError(data.get("error_description") or error or str(data))
        raise GitHubError("Device code expired before it was authorized")


class GitHubClient:
    """Minimal async GitHub REST client acting *as the user* who owns the token."""

    def __init__(self, token: str, api_url: str = "https://api.github.com"):
        self._api_url = api_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        async with (
            aiohttp.ClientSession(headers=self._headers) as session,
            session.request(method, f"{self._api_url}{path}", **kwargs) as resp,
        ):
            data = await resp.json(content_type=None)
            if resp.status >= 400:
                message = data.get("message") if isinstance(data, dict) else data
                raise GitHubError(f"{method} {path} -> {resp.status}: {message}")
            return data

    async def get_login(self) -> str:
        return (await self._request("GET", "/user"))["login"]

    async def get_default_branch(self, repo: str) -> str:
        return (await self._request("GET", f"/repos/{repo}"))["default_branch"]

    async def get_branch_sha(self, repo: str, branch: str) -> str:
        data = await self._request(
            "GET", f"/repos/{repo}/git/ref/heads/{quote(branch)}"
        )
        return data["object"]["sha"]

    async def create_branch(self, repo: str, branch: str, sha: str) -> None:
        await self._request(
            "POST",
            f"/repos/{repo}/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": sha},
        )

    async def get_file(self, repo: str, path: str, ref: str) -> tuple[str, str]:
        """Return (text, blob_sha) of a file at ``ref``."""
        data = await self._request(
            "GET", f"/repos/{repo}/contents/{quote(path)}", params={"ref": ref}
        )
        return base64.b64decode(data["content"]).decode(), data["sha"]

    async def update_file(
        self, repo: str, path: str, branch: str, text: str, sha: str, message: str
    ) -> None:
        await self._request(
            "PUT",
            f"/repos/{repo}/contents/{quote(path)}",
            json={
                "message": message,
                "content": base64.b64encode(text.encode()).decode(),
                "sha": sha,
                "branch": branch,
            },
        )

    async def create_pull_request(
        self, repo: str, head: str, base: str, title: str, body: str
    ) -> str:
        data = await self._request(
            "POST",
            f"/repos/{repo}/pulls",
            json={"title": title, "head": head, "base": base, "body": body},
        )
        return data["html_url"]
