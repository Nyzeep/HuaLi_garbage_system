"""Download a Rust acceleration wheel from GitHub Releases.

Usage:
    python download_rust_wheel.py --repo OWNER/REPO --tag TAG --pattern GLOB --wheel-dir DIR

Exit codes:
    0  success
    1  network / API error (retries exhausted)
    2  no matching asset found in the release
    3  invalid arguments
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MAX_RETRIES = 3
RETRY_DELAYS = (2.0, 4.0, 8.0)
API_TIMEOUT = 15
DOWNLOAD_TIMEOUT = 120


def _build_api_url(repo: str, tag: str) -> str:
    if tag == "latest":
        return f"https://api.github.com/repos/{repo}/releases/latest"
    return f"https://api.github.com/repos/{repo}/releases/tags/{tag}"


def _fetch_release_json(api_url: str, attempt: int) -> dict:
    req = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "HuaLi-garbage-system-launcher",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            reset = exc.headers.get("X-RateLimit-Reset", "unknown")
            print(
                f"[ERROR] GitHub API rate limit hit (HTTP 403). "
                f"Rate limit resets at: {reset}. "
                f"Unauthenticated requests are limited to 60/hour.",
                file=sys.stderr,
            )
            raise
        if exc.code == 404:
            print(
                f"[ERROR] GitHub Release not found (HTTP 404): {api_url}",
                file=sys.stderr,
            )
            raise
        if 500 <= exc.code < 600:
            print(
                f"[WARN] GitHub server error (HTTP {exc.code}), attempt {attempt}/{MAX_RETRIES}",
                file=sys.stderr,
            )
            raise
        print(
            f"[ERROR] GitHub API HTTP error {exc.code}: {exc.reason}",
            file=sys.stderr,
        )
        raise
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        print(
            f"[WARN] Network error, attempt {attempt}/{MAX_RETRIES}: {exc}",
            file=sys.stderr,
        )
        raise


def _find_matching_asset(assets: list[dict], pattern: str) -> dict | None:
    return next(
        (a for a in assets if fnmatch.fnmatch(a.get("name", ""), pattern)),
        None,
    )


def _download_wheel(url: str, target: Path, attempt: int) -> Path:
    try:
        urllib.request.urlretrieve(url, target, reporthook=None)
        return target
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        print(
            f"[WARN] Download failed, attempt {attempt}/{MAX_RETRIES}: {exc}",
            file=sys.stderr,
        )
        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a Rust wheel from GitHub Releases",
    )
    parser.add_argument("--repo", required=True, help="GitHub repo (OWNER/REPO)")
    parser.add_argument("--tag", required=True, help='Release tag or "latest"')
    parser.add_argument("--pattern", required=True, help="Asset name glob pattern")
    parser.add_argument(
        "--wheel-dir",
        required=True,
        help="Directory to save the downloaded wheel",
    )
    args = parser.parse_args()

    wheel_dir = Path(args.wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    api_url = _build_api_url(args.repo, args.tag)

    data: dict | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = _fetch_release_json(api_url, attempt)
            break
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout, OSError):
            if attempt >= MAX_RETRIES:
                print(
                    f"[ERROR] Failed to fetch GitHub Release info after {MAX_RETRIES} attempts.",
                    file=sys.stderr,
                )
                return 1
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            print(f"[INFO] Retrying in {delay:.0f}s...", file=sys.stderr)
            time.sleep(delay)

    if data is None:
        print("[ERROR] No release data received.", file=sys.stderr)
        return 1

    assets = data.get("assets", [])
    match = _find_matching_asset(assets, args.pattern)
    if match is None:
        asset_names = [a.get("name", "") for a in assets]
        print(
            f"[ERROR] No asset matching '{args.pattern}' in release. "
            f"Available assets: {asset_names or '(none)'}",
            file=sys.stderr,
        )
        return 2

    target = wheel_dir / match["name"]
    download_url = match["browser_download_url"]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _download_wheel(download_url, target, attempt)
            print(target)
            return 0
        except (urllib.error.URLError, socket.timeout, OSError):
            if attempt >= MAX_RETRIES:
                print(
                    f"[ERROR] Failed to download wheel after {MAX_RETRIES} attempts.",
                    file=sys.stderr,
                )
                return 1
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            print(f"[INFO] Retrying download in {delay:.0f}s...", file=sys.stderr)
            time.sleep(delay)

    return 1


if __name__ == "__main__":
    sys.exit(main())
