"""
Single source of truth for "is this session a real, trainable response?"

Historically every trainer carried its own copy of `_is_valid`. They all
agreed, but the copies drifted apart easily and there was no shared place to
add troll/spam filtering. This module centralises:

  * structural validity (the original `_is_valid` checks: gender code present,
    duration in range, bracket arrays the right length, numeric mood),
  * the tightened age rule (troll ages removed), and
  * the spam / troll-pick filter.

Short raw rows (save.ligma) and converted live-DB rows share the SAME 9-field
layout, so one validator covers both:

    [id, time, ip, age, mood, gender, valg, tider, farger]
    farger = [offered(64), r1(16), r2(4), final]

Long live-DB rows use a richer payload (offered 256 / r1 64 / r2 16 / r3 4),
handled by `is_valid_long_clean`.

Age rule (per product owner): drop age < 6, age > 80, and the joke ages 67 and
69. So the valid age set is [6, 80] minus {67, 69}. (The old pipeline capped at
68; this widens the top end while punching out the two troll values.)

Spam rule: "pressing the same place" is only a troll if the picks are BOTH
heavily concentrated on one corner AND fast. Someone who genuinely loves the
same corner but deliberates (e.g. ~30 s between picks) is kept. Concretely a
session is spam when the most-common round-0 position covers MORE than
SPAM_SAME_POS_FRAC of the 4-option questions and the mean per-pick time on
those questions is under SPAM_FAST_MEAN_MS. Tune SPAM_FAST_MEAN_MS up toward
infinity for a pure position-only rule, or down to 0 to disable the spam gate.
"""

import hashlib
import json
import os
from collections import Counter

_RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
_SAVE_LIGMA = os.path.join(_RAW_DIR, "save.ligma")
_SHORT_FROM_LONG = os.path.join(_RAW_DIR, "short_from_long.json")

# ---------- structural constants (shared with the trainers) ----------

DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000
N_QUESTIONS = 21          # short survey: 16 round-0 + 4 round-1 + 1 final
SHORT_N_R0Q = 16          # short 4-option (round-0) questions
LONG_N_R0Q = 64           # long 4-option (round-0) questions

# ---------- age rule ----------

AGE_MIN = 6
AGE_MAX = 80
TROLL_AGES = frozenset({67, 69})

# ---------- spam / troll-pick rule ----------

SPAM_SAME_POS_FRAC = 0.90   # > 90 % of round-0 picks on one corner
SPAM_FAST_MEAN_MS = 2000    # ...and faster than this mean per-pick time = spam

# ---------- frozen content-hashed hold-out (stable eval split) ----------
#
# Leaderboard noise between versions came from re-deriving the train/val split
# on every retrain: a fixed RNG seed still reshuffles WHICH rows land in the
# fold once the dataset changes, so two versions were scored on different test
# rows (a few-hundred-row fold swings gender AUC by +-0.015, age MAE by +-0.45).
# Instead we assign every session to train-or-test by a hash of its IMMUTABLE
# content (offered colours + the pick string + the final colour). A given
# session therefore lands in the same fold forever; ingesting new data only
# adds new sessions to whichever side their hash already dictates, it never
# moves an existing one across the line. The split is content-addressed (no
# state file to keep in sync) and scales automatically as data grows.
#
# Two duplicate sessions (identical content) hash alike and so land on the same
# side together, which also prevents an accidental train/test leak of a repeat.

HOLDOUT_FRAC = 0.10        # short sessions reserved for eval (~ the old 710)
HOLDOUT_FRAC_LONG = 0.30   # real long sessions reserved for eval (longs scarce)


def _holdout_unit(key) -> float:
    """Deterministic value in [0, 1) from a content key (top 32 bits of SHA-1)."""
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000


def short_holdout_key(row) -> str:
    """Immutable fingerprint of a short raw row (save.ligma layout): the offered
    colours + the pick string + the final colour. The offered grid is generated
    fresh per session, so this is effectively unique and never changes."""
    farger = row[8]
    return json.dumps([farger[0], str(row[6]), farger[3]], separators=(",", ":"))


def short_is_holdout(row, frac: float = HOLDOUT_FRAC) -> bool:
    """True if this short session is permanently in the eval hold-out."""
    return _holdout_unit(short_holdout_key(row)) < frac


def long_holdout_key(payload) -> str:
    """Immutable fingerprint of a real long session payload."""
    return json.dumps([payload["offered"], str(payload["valg"]), payload["final"]],
                      separators=(",", ":"))


def long_is_holdout(payload, frac: float = HOLDOUT_FRAC_LONG) -> bool:
    """True if this real long session is permanently in the eval hold-out."""
    return _holdout_unit(long_holdout_key(payload)) < frac


def dedupe_short_rows(rows):
    """Drop exact content-duplicate short sessions, keeping the first occurrence.

    About a quarter of save.ligma rows are literal repeats: an identical offered
    grid (which is generated randomly per session, so a match is not chance) plus
    identical picks and final colour, i.e. the same survey submitted more than
    once. Duplicates over-weight the deployed model (the largest repeat group is
    18 copies) and, before the frozen content-hashed hold-out, leaked the same
    session across train and test. Keeping one representative per content key
    fixes both; which copy survives does not matter because the hold-out is keyed
    on the same content. Rows whose content cannot be keyed (malformed) are kept
    as-is so the normal validity filter can drop them.
    """
    seen = set()
    out = []
    for r in rows:
        try:
            k = short_holdout_key(r)
        except Exception:
            out.append(r)
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


# Gender codes: live DB stores 0=man, 1=woman, 2=non-binary. The raw row format
# uses "g" (gutt/boy = 0) and "j" (jente/girl = 1). Non-binary (2) has no raw
# code and is dropped per the troll rule, so the valid raw set is {"g", "j"}.
VALID_GENDER_CODES = frozenset({"g", "j"})


def age_is_valid(age) -> bool:
    try:
        a = int(age)
    except (TypeError, ValueError):
        return False
    return AGE_MIN <= a <= AGE_MAX and a not in TROLL_AGES


def _is_spam_positions(valg, tider, n_r0q) -> bool:
    """True if the round-0 picks are >90 % one corner AND fast (= spammed)."""
    picks = str(valg)[:n_r0q]
    if len(picks) < n_r0q:
        return False
    counts = Counter(picks)
    top_frac = max(counts.values()) / n_r0q
    if top_frac <= SPAM_SAME_POS_FRAC:
        return False
    # Heavily concentrated. Only call it spam if it was also fast; a slow,
    # deliberate same-corner picker is a genuine (if quirky) response.
    try:
        t = [int(x) for x in tider[:n_r0q]]
    except (TypeError, ValueError):
        return True  # unparseable timing on a concentrated session -> treat as spam
    if not t:
        return True
    deltas = [max(0, t[0])] + [max(0, t[i] - t[i - 1]) for i in range(1, len(t))]
    mean_ms = sum(deltas) / len(deltas)
    return mean_ms < SPAM_FAST_MEAN_MS


def is_valid_clean(row) -> bool:
    """Validity + troll filter for a short raw row (save.ligma layout)."""
    try:
        if row[5] not in VALID_GENDER_CODES:
            return False
        if not age_is_valid(row[3]):
            return False
        if row[8] == "no data" or len(row[8]) < 4:
            return False
        if len(row[8][0]) < 64 or len(row[8][1]) < 16 or len(row[8][2]) < 4:
            return False
        if len(row[7]) < N_QUESTIONS:
            return False
        total = int(row[7][-1])
        if total < DURATION_MIN_MS or total > DURATION_MAX_MS:
            return False
        if not str(row[4]).lstrip("-").isdigit():
            return False
        if _is_spam_positions(row[6], row[7], SHORT_N_R0Q):
            return False
        return True
    except Exception:
        return False


def is_valid_long_clean(payload, label) -> bool:
    """Validity + troll filter for a real long session.

    payload: {offered(256), r1(64), r2(16), r3(4), final, valg(85), tider(85)}
    label:   {gender(0/1), age, mood, time}
    """
    try:
        if int(label["gender"]) not in (0, 1):
            return False
        if not age_is_valid(label["age"]):
            return False
        off = payload["offered"]
        if (len(off) < 256 or len(payload["r1"]) < 64 or len(payload["r2"]) < 16
                or len(payload["r3"]) < 4 or len(payload["final"]) < 3):
            return False
        tider = [int(x) for x in payload["tider"]]
        if len(tider) < 85:
            return False
        total = tider[-1]
        if total < DURATION_MIN_MS or total > DURATION_MAX_MS:
            return False
        if _is_spam_positions(payload["valg"], tider, LONG_N_R0Q):
            return False
        return True
    except Exception:
        return False


# ---------- live-DB row -> raw row conversion ----------

def _gender_code(confirmed_gender):
    """0 (man) -> 'g', 1 (woman) -> 'j'. Anything else (e.g. 2 non-binary)
    returns None so the caller drops the row."""
    try:
        g = int(confirmed_gender)
    except (TypeError, ValueError):
        return None
    return "g" if g == 0 else "j" if g == 1 else None


def db_row_to_short_raw(db):
    """Convert one completed SHORT live-DB row into a save.ligma raw row.

    Returns the 9-field list, or None if the row is the wrong shape / non-binary
    / missing targets. Cleaning (is_valid_clean) is applied by the caller.
    """
    import json

    def _j(key):
        v = db.get(key)
        return json.loads(v) if isinstance(v, str) else v

    gender = _gender_code(db.get("confirmed_gender"))
    if gender is None:
        return None
    age = db.get("confirmed_age")
    mood = db.get("confirmed_mood")
    if age is None or mood is None:
        return None

    offered = _j("offered_json") or []
    r1 = _j("r1_json") or []
    r2 = _j("r2_json") or []
    final = _j("final_color_json") or []
    valg = db.get("valg") or ""
    tider = _j("tider_json") or []

    # unix seconds: prefer the client submit time, fall back to server receipt.
    submit_ms = db.get("client_submitted_at") or db.get("server_received_at") or 0
    try:
        time_sec = int(submit_ms) // 1000
    except (TypeError, ValueError):
        time_sec = 0

    return [
        str(db.get("id", "")),
        str(time_sec),
        "x",                       # ip is not exported; raw format keeps a placeholder
        str(int(age)),
        str(int(mood)),
        gender,
        str(valg),
        [int(x) for x in tider],
        [offered, r1, r2, final],
    ]


def db_row_to_long(db):
    """Convert one completed LONG live-DB row into (payload, label) for the
    long pipeline. Returns None if the row is the wrong shape / non-binary /
    missing targets."""
    import json

    def _j(key):
        v = db.get(key)
        return json.loads(v) if isinstance(v, str) else v

    gender = db.get("confirmed_gender")
    if _gender_code(gender) is None:
        return None
    age = db.get("confirmed_age")
    mood = db.get("confirmed_mood")
    if age is None or mood is None:
        return None

    submit_ms = db.get("client_submitted_at") or db.get("server_received_at") or 0
    try:
        time_sec = int(submit_ms) // 1000
    except (TypeError, ValueError):
        time_sec = 0

    payload = {
        "offered": _j("offered_json") or [],
        "r1": _j("r1_json") or [],
        "r2": _j("r2_json") or [],
        "r3": _j("r3_json") or [],
        "final": _j("final_color_json") or [],
        "valg": db.get("valg") or "",
        "tider": [int(x) for x in (_j("tider_json") or [])],
    }
    label = {
        "gender": 1 if int(gender) == 1 else 0,
        "age": int(age),
        "mood": int(mood),
        "time": time_sec,
    }
    return payload, label


# ---------- long -> 4 short surveys (mirror of the short -> long synthesis) ----------

def long_payload_to_shorts(payload, label, long_id):
    """Split one REAL long session into its 4 constituent short surveys.

    A long survey is structurally four short surveys stacked plus one extra
    final question (long = 4 x short + 1), assembled round-major. This reverses
    that: sub-short k takes the k-th block of every round, and the synthetic
    85th pick (the among-the-4-finalists question) is dropped.

    Unlike the short->long synthesis (which glues 4 *different* people together),
    every sub-short here is the SAME real person, so all four carry that
    person's gender / age / mood. Returns up to 4 raw rows (save.ligma layout);
    the caller still runs each through is_valid_clean.
    """
    offered = payload["offered"]
    r1 = payload["r1"]
    r2 = payload["r2"]
    r3 = payload["r3"]
    valg = str(payload["valg"])
    tider = [int(x) for x in payload["tider"]]
    if (len(offered) < 256 or len(r1) < 64 or len(r2) < 16 or len(r3) < 4
            or len(valg) < 85 or len(tider) < 85):
        return []
    deltas = [max(0, tider[0])] + [max(0, tider[i] - tider[i - 1]) for i in range(1, len(tider))]
    gender = "j" if int(label["gender"]) == 1 else "g"
    age = int(label["age"])
    mood = int(label["mood"])
    time_sec = int(label.get("time", 0) or 0)

    rows = []
    for k in range(4):
        off_k = offered[k * 64:(k + 1) * 64]                 # 64 offered
        r1_k = r1[k * 16:(k + 1) * 16]                        # 16 round-0 winners
        r2_k = r2[k * 4:(k + 1) * 4]                          # 4 round-1 winners
        final_k = r3[k]                                       # this sub-short's winner
        # picks: 16 round-0 + 4 round-1 + 1 final (the among-this-block question)
        valg_k = valg[k * 16:(k + 1) * 16] + valg[64 + k * 4:64 + (k + 1) * 4] + valg[80 + k:81 + k]
        d_k = deltas[k * 16:(k + 1) * 16] + deltas[64 + k * 4:64 + (k + 1) * 4] + [deltas[80 + k]]
        if (len(off_k) < 64 or len(r1_k) < 16 or len(r2_k) < 4
                or len(final_k) < 3 or len(valg_k) < 21 or len(d_k) < 21):
            continue
        tider_k, run = [], 0
        for d in d_k:
            run += d
            tider_k.append(int(run))
        rows.append([
            f"{long_id}#{k}", str(time_sec), "x", str(age), str(mood),
            gender, valg_k, tider_k, [off_k, r1_k, r2_k, list(final_k)],
        ])
    return rows


# ---------- short training-row loader (used by all short-model trainers) ----------

def load_short_rows(save_path=None):
    """Load the short training rows: save.ligma plus, unless
    CP_INCLUDE_DECOMPOSED=0, the shorts decomposed from real long surveys
    (raw/short_from_long.json). All short-model trainers (features.py,
    lgb-production, taste-cube/train_pick) load through here so they select the
    identical row set in the identical order. The long synthesis (train_long,
    train_pick_long) deliberately does NOT use this - it reads save.ligma only,
    so decomposed shorts are never re-synthesised back into longs."""
    import json
    with open(save_path or _SAVE_LIGMA, encoding="utf-8") as fh:
        rows = json.load(fh)
    if os.environ.get("CP_INCLUDE_DECOMPOSED", "1") != "0" and os.path.exists(_SHORT_FROM_LONG):
        with open(_SHORT_FROM_LONG, encoding="utf-8") as fh:
            rows = rows + json.load(fh)
    return dedupe_short_rows(rows)
