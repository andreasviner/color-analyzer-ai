# Color Polygraph - Cloudflare API

Workers Python service that powers the live survey at
[/ai/color-polygraph/survey.html](../survey.html).

## What it does

1. Browser finishes the bracket and POSTs the raw payload + client metadata.
2. Worker computes the 477-feature vector (pure Python, no numpy), persists
   the *raw payload* + IP-hash + CF geo + metadata to a D1 database, and
   returns the features to the browser. The computed features are NOT
   persisted - they are deterministic from the raw payload, and recomputing
   them from `features.py` at training time costs nothing and saves several
   KB of D1 storage per row.
3. Browser fetches `gender_trees.json` from `/ai/color-polygraph/models-js/`
   and walks the tree with `tree_walker.js`, then POSTs back the prediction +
   user-confirmed truth.
4. Mood and age confirm rounds work the same way.

The LightGBM trees themselves are static JSON shipped with the site - they
never touch the worker. That keeps the worker small (a few hundred KB
compressed) and lets the survey use the full-size leaderboard models without
fighting the 3 MB Workers free-tier limit.

## File layout

```
cloudflare/
  worker.py        # entrypoint, 4 endpoints, CORS, IP hashing
  features.py      # pure-Python feature extraction (no numpy)
  bucket_data.py   # 8x8x8 RGB bucket grids, embedded as Python lists
  schema.sql       # D1 schema (run once)
  wrangler.toml    # config; replace database_id after `wrangler d1 create`
  README.md        # this file
```

The leaderboard-grade LightGBM trees themselves live one level up at
`../models-js/` as JSON and are fetched by the browser; the worker only
runs feature extraction.

## One-time setup

You will need `wrangler` (the Cloudflare CLI). Install with:

```
npm install -g wrangler
wrangler login
```

### 1. Create the D1 database

```
wrangler d1 create color-polygraph
```

The CLI prints something like:

```
[[d1_databases]]
binding = "DB"
database_name = "color-polygraph"
database_id = "abcd1234-..."
```

Copy the `database_id` value into `wrangler.toml` (replace the placeholder
`REPLACE-WITH-D1-ID-FROM-WRANGLER-CREATE` string).

### 2. Apply the schema

Local emulator (uses a local sqlite file under `.wrangler/`):

```
wrangler d1 execute color-polygraph --local --file=./schema.sql
```

Remote (creates the tables in the real D1):

```
wrangler d1 execute color-polygraph --remote --file=./schema.sql
```

### 3. Set the IP-hash secret

```
wrangler secret put IP_HASH_SALT
```

It prompts for a value - any non-empty random string (e.g. `openssl rand -hex 32`).
This is hashed with HMAC-SHA256 against the client IP. The IP itself is never
written to the database.

### 4. (Optional) Wire up a custom domain

Edit `wrangler.toml` and uncomment the `[[routes]]` block once
`api.andreaslindeman.com` is pointed at Cloudflare. Until then, the worker is
reachable at `https://color-polygraph-api.<your-subdomain>.workers.dev`.

## Local development

```
wrangler dev
```

This starts the worker on `http://localhost:8787` against the local D1
emulator. Point the survey at it with the `?api=` query string:

```
http://localhost:8000/ai/color-polygraph/survey.html?api=http://localhost:8787
```

(Run a static server for the rest of the site, e.g.
`python -m http.server -d ai/color-polygraph 8000`.)

Hit the health endpoint as a smoke test:

```
curl http://localhost:8787/health
```

## Deploy

```
wrangler deploy
```

## Schema cheat sheet

`surveys` is a single wide table. Each row has:

- raw inputs (offered / r1 / r2 / final colors, valg, tider) as JSON blobs -
  the 477-feature vectors are NOT stored, since they're deterministic from the
  raw inputs and `features.py` recomputes them at training time
- client metadata (UA, language, screen, mobile, referrer, local time)
- server metadata (hashed IP, CF country/region/city/timezone)
- prediction + user-confirmed truth for each of gender / mood / age. Gender
  truth is stored as INTEGER (1 = girl, 0 = boy). `was_correct` flags and
  per-target `confirmed_at` timestamps are deliberately NOT stored - both are
  derivable at query time (pred vs confirmed for correctness;
  `server_received_at` is one column for the whole row)

To inspect rows:

```
wrangler d1 execute color-polygraph --local --command "SELECT id, confirmed_gender, confirmed_age, confirmed_mood FROM surveys ORDER BY server_received_at DESC LIMIT 10;"
```

## Endpoints

| Method | Path                          | Body                                            | Returns          |
| ------ | ----------------------------- | ----------------------------------------------- | ---------------- |
| POST   | `/survey`                     | `{payload, metadata}`                           | `{id, features}` |
| POST   | `/survey/:id/gender`          | `{pred_prob, confirmed_label}` ('boy' or 'girl')| `{ok}`           |
| POST   | `/survey/:id/mood`            | `{pred_value, confirmed_value}` (0..60 int)     | `{ok}`           |
| POST   | `/survey/:id/age`             | `{pred_value, confirmed_value}` (6..99 int)     | `{ok}`           |
| GET    | `/health`                     | -                                               | `{ok, service}`  |

CORS is allowed from origins listed in `ALLOWED_ORIGINS` in `wrangler.toml`.
