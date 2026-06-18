# Live-data refresh + retrain

End-to-end pipeline that pulls the live survey database, cleans it, folds it
into the training data, retrains every production model, and republishes the
leaderboards. Two scripts:

| Script | Job |
| --- | --- |
| `pull_remote.py` | Download the live DB via the authenticated worker `/export` endpoint -> `remote_dump.json`. |
| `refresh_and_retrain.py` | Ingest the dump, rebuild features, retrain the 4 models, bump the version, rewrite both leaderboards. |

## One-time Cloudflare setup

The worker (`cloudflare/worker.py`) exposes `GET /color-polygraph/export`,
gated by an `EXPORT_TOKEN` secret. Set it and deploy:

```bash
cd ../../cloudflare
wrangler secret put EXPORT_TOKEN     # paste a long random string
wrangler deploy
```

If the secret is unset the endpoint fails closed (HTTP 503), so it can never
leak the table by accident.

## Each refresh

```bash
cd training/refresh
export CP_API_BASE="https://api.andreaslindeman.com"   # host that serves the survey POSTs
export CP_EXPORT_TOKEN="<the EXPORT_TOKEN you set>"
python pull_remote.py                 # -> remote_dump.json
python refresh_and_retrain.py         # ingest + retrain + republish
```

Running `refresh_and_retrain.py` with no dump present is a safe **dry run**: it
rebuilds features and retrains on the existing data only.

## What the orchestrator does

1. **Ingest** `remote_dump.json`: each completed row is cleaned by
   `training/data_cleaning.py` (the single source of truth for validity + troll
   rules). Clean **short** rows are appended to `raw/save.ligma` (deduped by id,
   `save.ligma.bak` written first); clean **long** rows are merged into
   `raw/long_real.json`.
2. **Features**: rebuilds `features.npy` / `targets.npz` and `features_extra.npy`
   in lockstep (identical row selection + order).
3. **Train** (subprocesses):
   - `lgb-production/train_and_emit.py` - short gender/age/mood
   - `long-models/train_long.py` - long gender/age/mood (real long rows get
     `CP_REAL_LONG_WEIGHT` sample weight, default 3.0)
   - `taste-cube/train_pick.py` - short colour-pick
   - `taste-cube/train_pick_long.py` - long colour-pick

   Trees are emitted as JSON to `ai/english_html/color-polygraph/models-js/`
   (what the live survey fetches) and bucket grids to `cloudflare/`.
4. **Version**: bumps `version.json` by +0.1 (`--major` for +1.0, or
   `--version X.Y` to pin) and records the new production metrics.
5. **Publish**: regenerates the version rows and the colour-pick table in the EN
   and NO `index.html` between their `<!-- LB:... -->` markers, and updates the
   "Last refresh" date.

## Cleaning / troll rules (`training/data_cleaning.py`)

- **Age**: keep 6-80, drop the joke ages 67 and 69.
- **Gender**: keep man/woman; non-binary (DB code 2) has no raw code and is dropped.
- **Spam**: drop a session only if more than 90% of its 4-option picks land on
  one corner *and* the picking was fast (mean per-pick < `SPAM_FAST_MEAN_MS`).
  A slow, deliberate same-corner picker is kept.
- Plus the original structural checks (duration 15s-600s, bracket arrays the
  right length, numeric mood).

## Flags

```
--major                 bump the major version (+1.0)
--version X.Y           pin the new version
--real-long-weight W    sample weight for real long rows (default 3.0)
--skip-train            rebuild data + republish HTML only (no model training)
--dump PATH             ingest a specific dump file
```
