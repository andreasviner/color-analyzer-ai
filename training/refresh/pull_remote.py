"""
Pull the live Color Polygraph survey database down for retraining.

Talks to the authenticated `GET /color-polygraph/export` endpoint added to
the Cloudflare worker (cloudflare/worker.py). Walks the table in pages and
writes one raw JSON dump that the retraining orchestrator
(refresh_and_retrain.py) cleans and folds into the training set.

The dump is the *raw* database rows (offered/r1/r2/r3/final/valg/tider JSON
columns + confirmed_gender/age/mood targets). No cleaning happens here; that
is data_cleaning.py's job, so the dump stays an honest snapshot of the DB.

Usage:
    # base URL + token from the environment (recommended)
    export CP_API_BASE="https://api.andreaslindeman.com"
    export CP_EXPORT_TOKEN="<the EXPORT_TOKEN secret you set with wrangler>"
    python pull_remote.py

    # or pass them explicitly
    python pull_remote.py --base https://api.andreaslindeman.com --token XXXX

    # include not-yet-completed rows too (default: completed rows only)
    python pull_remote.py --all

Output:
    training/refresh/remote_dump.json
        {"pulled_at": "...", "completed_only": true, "count": N, "rows": [...]}
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "remote_dump.json")

# color-polygraph root (the folder with cloudflare/) holds the .env.
_CP_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def _load_dotenv(path):
    """Minimal KEY=VALUE loader. Does not override variables already set in the
    real environment, so an explicit `export` still wins over the file."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


_load_dotenv(os.path.join(_CP_ROOT, ".env"))

DEFAULT_BASE = os.environ.get("CP_API_BASE", "https://api.andreaslindeman.com")
PAGE_LIMIT = 1000          # matches the worker default; capped at 5000 there
MIN_PAGE_LIMIT = 50        # smallest page we bother trying before giving up
PAGE_ATTEMPTS = 8          # tries per page before shrinking it (cold-start 1101s)
RETRY_SLEEP = 1.5          # seconds between attempts
REQUEST_TIMEOUT = 60       # seconds per page request


class ExportError(Exception):
    """A page request failed. `code` is the HTTP status (0 if unreachable)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _fetch_page(base, token, limit, offset, completed_only):
    qs = urllib.parse.urlencode({
        "limit": limit,
        "offset": offset,
        "completed_only": "1" if completed_only else "0",
    })
    url = f"{base.rstrip('/')}/color-polygraph/export?{qs}"
    # A browser-like User-Agent: Cloudflare's Bot Fight Mode / Browser Integrity
    # Check rejects the default "Python-urllib/..." UA at the edge (error 1010)
    # before the request reaches the worker.
    req = urllib.request.Request(url, headers={
        "x-export-token": token,
        "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            pass
        raise ExportError(
            exc.code,
            f"export request failed: HTTP {exc.code} {exc.reason}\n  url: {url}\n  body: {body}"
        )
    except urllib.error.URLError as exc:
        raise ExportError(0, f"could not reach {url}: {exc.reason}")


def pull_all(base, token, completed_only=True, out_path=DEFAULT_OUT,
             page_limit=PAGE_LIMIT):
    """Walk the export endpoint in pages.

    The worker's per-request CPU budget scales with the SIZE of the payloads on
    the page, not just the row count: long-survey rows (256 colours) are far
    heavier than short ones, so a page size that works at the start of the table
    can trip error 1102 ("Worker exceeded resource limits") further in. On a 5xx
    we halve the page size and retry the same offset rather than losing the rows
    already pulled.
    """
    rows = []
    offset = 0
    total = None
    limit = page_limit
    t0 = time.time()
    while True:
        page = None
        # The Python worker throws error 1101 on a good fraction of requests
        # (cold starts), independent of the page contents: the same offset that
        # fails will succeed a moment later. So retry the page as-is first, and
        # only shrink it if every attempt failed (that is the CPU-limit case,
        # error 1102, which retries alone cannot fix).
        for attempt in range(1, PAGE_ATTEMPTS + 1):
            try:
                page = _fetch_page(base, token, limit, offset, completed_only)
                break
            except ExportError as exc:
                if not (exc.code == 0 or exc.code >= 500):
                    raise SystemExit(str(exc))
                last_error = exc
                if attempt < PAGE_ATTEMPTS:
                    print(f"  page offset={offset} limit={limit} attempt "
                          f"{attempt} failed (HTTP {exc.code}); retrying")
                    time.sleep(RETRY_SLEEP)
        if page is None:
            if limit > MIN_PAGE_LIMIT:
                limit = max(MIN_PAGE_LIMIT, limit // 2)
                print(f"  page offset={offset} failed {PAGE_ATTEMPTS}x; "
                      f"shrinking to limit={limit}")
                continue
            raise SystemExit(str(last_error))
        batch = page.get("rows", [])
        rows.extend(batch)
        total = page.get("total", total)
        print(f"  page offset={offset} limit={limit}  got {len(batch)}  "
              f"(running {len(rows)}/{total})")
        if not page.get("has_more") or not batch:
            break
        offset += len(batch)

    snapshot = {
        "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": base,
        "completed_only": completed_only,
        "count": len(rows),
        "total_reported": total,
        "rows": rows,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh)
    print(f"\nWrote {len(rows)} rows to {out_path}  ({time.time() - t0:.1f}s)")
    return snapshot


def main():
    ap = argparse.ArgumentParser(description="Pull the live survey DB for retraining.")
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="API base URL (or set CP_API_BASE). "
                         f"default: {DEFAULT_BASE}")
    ap.add_argument("--token", default=os.environ.get("CP_EXPORT_TOKEN"),
                    help="export token (or set CP_EXPORT_TOKEN)")
    ap.add_argument("--all", action="store_true",
                    help="include rows that have not completed all three "
                         "confirmations (default: completed only)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dump path")
    ap.add_argument("--limit", type=int, default=PAGE_LIMIT,
                    help="rows per page; halved automatically if the worker "
                         f"trips its CPU limit. default: {PAGE_LIMIT}")
    args = ap.parse_args()

    if not args.token:
        sys.exit("no export token: pass --token or set CP_EXPORT_TOKEN "
                 "(the EXPORT_TOKEN secret you set with `wrangler secret put`)")

    print(f"Pulling from {args.base}  (completed_only={not args.all}) ...")
    pull_all(args.base, args.token, completed_only=not args.all,
             out_path=args.out, page_limit=args.limit)


if __name__ == "__main__":
    main()
