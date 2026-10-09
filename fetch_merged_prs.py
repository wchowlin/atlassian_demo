#!/usr/bin/env python3
"""
Fetch merged pull request data from the GitHub API and write it to a CSV file.

For each merged PR the CSV includes:
  * Author details (GitHub user profile)
  * Merger details (GitHub user profile of the user who merged the PR)
  * Additions, deletions
  * Created / merged timestamps
  * Time difference between creation and merge

Usage:
  export GITHUB_TOKEN=ghp_xxx          # optional but strongly recommended
  python fetch_merged_prs.py owner/repo -o merged_prs.csv --max-prs 200

Requires: requests  (pip install requests)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Iterator

import requests

API_ROOT = "https://api.github.com"
USER_FIELDS = ["login", "id", "name", "company", "location", "email",
               "public_repos", "followers", "created_at", "html_url"]

log = logging.getLogger("merged_prs")


class GitHubClient:
    """Thin GitHub REST client with pagination, rate-limit handling and caching."""

    def __init__(self, token: str | None, timeout: int = 30, max_retries: int = 5):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "merged-pr-exporter",
        })
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.timeout = timeout
        self.max_retries = max_retries
        self._user_cache: dict[str, dict[str, Any]] = {}

    def _request(self, url: str, params: dict | None = None) -> requests.Response:
        for attempt in range(1, self.max_retries + 1):
            resp = self.session.get(url, params=params, timeout=self.timeout)

            # Primary or secondary rate limit: wait until reset, then retry.
            if resp.status_code in (403, 429) and (
                resp.headers.get("X-RateLimit-Remaining") == "0"
                or "Retry-After" in resp.headers
            ):
                if "Retry-After" in resp.headers:
                    wait = int(resp.headers["Retry-After"])
                else:
                    reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait = max(reset - int(time.time()), 1)
                log.warning("Rate limited. Sleeping %ss (attempt %s/%s)",
                            wait, attempt, self.max_retries)
                time.sleep(wait + 1)
                continue

            # Transient server errors: exponential backoff.
            if resp.status_code >= 500:
                wait = 2 ** attempt
                log.warning("Server error %s. Retrying in %ss", resp.status_code, wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp

        raise RuntimeError(f"Giving up on {url} after {self.max_retries} attempts")

    def paginate(self, path: str, params: dict | None = None) -> Iterator[dict]:
        url: str | None = f"{API_ROOT}{path}"
        params = {**(params or {}), "per_page": 100}
        while url:
            resp = self._request(url, params=params)
            yield from resp.json()
            url = resp.links.get("next", {}).get("url")
            params = None  # the "next" URL already carries the query string

    def get(self, path: str) -> dict:
        return self._request(f"{API_ROOT}{path}").json()

    def get_user(self, login: str | None) -> dict:
        if not login:
            return {}
        if login not in self._user_cache:
            self._user_cache[login] = self.get(f"/users/{login}")
        return self._user_cache[login]


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}d {hours:02d}h {minutes:02d}m {secs:02d}s"


def user_columns(prefix: str, user: dict) -> dict:
    return {f"{prefix}_{field}": user.get(field) for field in USER_FIELDS}


def iter_merged_prs(client: GitHubClient, repo: str, base: str | None,
                    since: datetime | None, max_prs: int | None) -> Iterator[dict]:
    """Yield full PR objects for merged PRs, newest-updated first."""
    params = {"state": "closed", "sort": "updated", "direction": "desc"}
    if base:
        params["base"] = base

    count = 0
    for pr in client.paginate(f"/repos/{repo}/pulls", params):
        # Sorted by updated desc, so once we pass `since` nothing older can qualify.
        if since and parse_ts(pr["updated_at"]) < since:
            break
        if not pr.get("merged_at"):
            continue  # closed without merging
        if since and parse_ts(pr["merged_at"]) < since:
            continue

        # The list endpoint omits additions/deletions/merged_by; fetch the detail.
        yield client.get(f"/repos/{repo}/pulls/{pr['number']}")
        count += 1
        if max_prs and count >= max_prs:
            break


def build_row(client: GitHubClient, pr: dict) -> dict:
    author_login = (pr.get("user") or {}).get("login")
    # merged_by can be null (e.g. deleted accounts, some automated merges).
    # In this exercise the merger is treated as the author, so fall back to it.
    merger_login = (pr.get("merged_by") or {}).get("login") or author_login

    created = parse_ts(pr["created_at"])
    merged = parse_ts(pr["merged_at"])
    delta = (merged - created).total_seconds()

    row = {
        "pr_number": pr["number"],
        "pr_title": pr["title"],
        "pr_url": pr["html_url"],
        "base_branch": pr["base"]["ref"],
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "changed_files": pr.get("changed_files"),
        "created_at": pr["created_at"],
        "merged_at": pr["merged_at"],
        "time_to_merge_seconds": int(delta),
        "time_to_merge_hours": round(delta / 3600, 2),
        "time_to_merge_readable": format_duration(delta),
    }
    row.update(user_columns("author", client.get_user(author_login)))
    row.update(user_columns("merger", client.get_user(merger_login)))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Export merged GitHub PRs to CSV.")
    parser.add_argument("repo", help="Repository in owner/name form, e.g. psf/requests")
    parser.add_argument("-o", "--output", default="merged_prs.csv", help="Output CSV path")
    parser.add_argument("--base", help="Only PRs merged into this base branch")
    parser.add_argument("--since", help="Only PRs merged on/after this date (YYYY-MM-DD)")
    parser.add_argument("--max-prs", type=int, default=100,
                        help="Max merged PRs to export (0 = no limit)")
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN"),
                        help="GitHub token (defaults to $GITHUB_TOKEN)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if "/" not in args.repo:
        parser.error("repo must be in owner/name form")
    if not args.token:
        log.warning("No token provided: limited to 60 requests/hour.")

    since = (datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
             if args.since else None)

    client = GitHubClient(args.token)
    rows = []
    try:
        for pr in iter_merged_prs(client, args.repo, args.base, since, args.max_prs or None):
            rows.append(build_row(client, pr))
            log.info("Processed PR #%s", pr["number"])
    except requests.HTTPError as exc:
        log.error("GitHub API error: %s", exc)
        if not rows:
            return 1
        log.warning("Writing %d rows collected before the error.", len(rows))

    if not rows:
        log.info("No merged PRs found.")
        return 0

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %d merged PRs to %s", len(rows), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())