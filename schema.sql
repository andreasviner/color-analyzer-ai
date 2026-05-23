-- Color Polygraph survey storage. Run once with `wrangler d1 execute`.
--
-- One row per survey session. The flow is:
--   1. POST /survey            inserts the row with raw + features + metadata
--                              + the gender / mood / age PREDICTIONS run client-side
--                              are NOT in this insert; client posts them later
--   2. POST /survey/:id/gender records the gender prediction + user's truth
--   3. POST /survey/:id/mood   records mood prediction + truth
--   4. POST /survey/:id/age    records age prediction + truth (final step)
--
-- The "confirmed_*" columns become the targets for the next training run.
-- The model produced the prediction *before* the user could lie, so
-- `confirmed_*` is the ground-truth signal we trust.

CREATE TABLE IF NOT EXISTS surveys (
    -- identity & timing
    id                    TEXT    PRIMARY KEY,
    server_received_at    INTEGER NOT NULL,            -- unix ms server-side
    client_started_at     INTEGER,                     -- unix ms client-side
    client_submitted_at   INTEGER,                     -- unix ms client-side
    client_local_time     TEXT,                        -- ISO 8601 string from the browser

    -- raw inputs (JSON blobs; small enough that we keep them verbatim for retraining).
    -- The 477-feature vectors are NOT stored here because they're deterministic from
    -- the raw payload + features.py; recomputing them at training time saves several
    -- KB per row.
    offered_json          TEXT    NOT NULL,            -- 64 [r,g,b]
    r1_json               TEXT    NOT NULL,            -- 16 [r,g,b]
    r2_json               TEXT    NOT NULL,            -- 4  [r,g,b]
    final_color_json      TEXT    NOT NULL,            -- [r,g,b]
    valg                  TEXT    NOT NULL,            -- 21 ASCII digits
    tider_json            TEXT    NOT NULL,            -- 21 cumulative ms ints

    -- client metadata
    user_agent            TEXT,
    referrer              TEXT,
    language              TEXT,                        -- navigator.language
    locale                TEXT,                        -- navigator.languages joined
    is_mobile             INTEGER,                     -- 1/0
    screen_w              INTEGER,
    screen_h              INTEGER,
    viewport_w            INTEGER,
    viewport_h            INTEGER,
    timezone_client       TEXT,                        -- Intl.DateTimeFormat().resolvedOptions().timeZone

    -- server-derived metadata (from Cloudflare request.cf + headers)
    ip_hash               TEXT,                        -- SHA-256(salt + cf-connecting-ip)
    country               TEXT,
    region                TEXT,
    city                  TEXT,
    timezone_cf           TEXT,                        -- request.cf.timezone

    -- predictions + user truth. Everything else (pred_label, was_correct,
    -- confirmed_at) is derivable: pred_label = (pred_prob >= 0.5), was_correct
    -- comes from comparing pred to confirmed, and the row's submit time is
    -- already captured by server_received_at.
    pred_gender_prob      REAL,                        -- P(girl) from client-side tree walk
    confirmed_gender      INTEGER,                     -- 1 = girl, 0 = boy
    pred_mood             REAL,                        -- 0-60 prediction
    confirmed_mood        INTEGER,                     -- 0-60 truth
    pred_age              REAL,                        -- years prediction
    confirmed_age         INTEGER                      -- years truth
);

-- Filter targets: rows that completed all three confirmations are the
-- training-grade rows for next time.
CREATE INDEX IF NOT EXISTS idx_surveys_completed
    ON surveys (confirmed_age)
    WHERE confirmed_age IS NOT NULL;

-- Useful for daily counts / dashboards
CREATE INDEX IF NOT EXISTS idx_surveys_received_at ON surveys (server_received_at);
