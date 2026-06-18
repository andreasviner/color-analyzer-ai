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

DEFAULT_BASE = os.environ.get("CP_API_BASE", "https://api.andreaslindeman.com")
PAGE_LIMIT = 1000          # matches the worker default; capped at 5000 there
REQUEST_TIMEOUT = 60       # seconds per page request


def _fetch_page(base, token, limit, offset, completed_only):
    qs = urllib.parse.urlencode({
        "limit": limit,
        "offset": offset,
        "completed_only": "1" if completed_only else "0",
    })
    url = f"{base.rstrip('/')}/color-polygraph/export?{qs}"
    req = urllib.request.Request(url, headers={"x-export-token": token})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            pass
        raise SystemExit(
            f"export request failed: HTTP {exc.code} {exc.reason}\n  url: {url}\n  body: {body}"
        )
    except urllib.error.URLError as exc:
        raise SystemExit(f"could not reach {url}: {exc.reason}")


def pull_all(base, token, completed_only=True, out_path=DEFAULT_OUT):
    rows = []
    offset = 0
    total = None
    t0 = time.time()
    while True:
        page = _fetch_page(base, token, PAGE_LIMIT, offset, completed_only)
        batch = page.get("rows", [])
        rows.extend(batch)
        total = page.get("total", total)
        print(f"  page offset={offset}  got {len(batch)}  "
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
    args = ap.parse_args()

    if not args.token:
        sys.exit("no export token: pass --token or set CP_EXPORT_TOKEN "
                 "(the EXPORT_TOKEN secret you set with `wrangler secret put`)")

    print(f"Pulling from {args.base}  (completed_only={not args.all}) ...")
    pull_all(args.base, args.token, completed_only=not args.all, out_path=args.out)


if __name__ == "__main__":
    main()
