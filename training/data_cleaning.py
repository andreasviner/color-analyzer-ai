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

from collections import Counter

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
