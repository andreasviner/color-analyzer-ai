"""
Cloudflare Workers Python entrypoint for the Color Polygraph survey.

Endpoints, all under `https://api.andreaslindeman.com` (or whatever route
the survey.html `API_BASE` points at):

    POST /color-polygraph/survey
        Body: raw survey payload + client metadata.
        Saves the row to D1, computes the three feature vectors server-side,
        attaches IP hash + Cloudflare geo, returns `{id, features}` so the
        browser can run gender inference first.

    GET /color-polygraph/survey/:id
        Returns the stored survey result by ID for the shareable result link
        (survey-result.html?id=...). Returns winner color, confirmed gender /
        age / mood, correct count, simplified pick history.

    POST /color-polygraph/survey/:id/gender
        Body: {pred_prob, pred_label, confirmed_label}
        Saves the gender prediction + user's truth. Returns {ok}.

    POST /color-polygraph/survey/:id/mood
        Body: {pred_value, confirmed_value}
        Same shape, for mood.

    POST /color-polygraph/survey/:id/age
        Body: {pred_value, confirmed_value}
        Final step. Marks the row as completed.

    OPTIONS *
        CORS preflight. Allows requests from the static-site origin.

Notes:
  * Feature extraction lives in features.py (numpy-free). On Pyodide-on-Workers
    that keeps the bundle small.
  * The big LightGBM trees do NOT live here. The browser fetches them as JSON
    from /ai/color-polygraph/models-js/*_trees.json on the static site.
  * IP hash uses HMAC-SHA256 with the IP_HASH_SALT secret. Set it once with
    `wrangler secret put IP_HASH_SALT`.
"""

import hashlib
import hmac
import json
import time
import uuid

from workers import Response

from features import compute_features, validate_payload


# ---------- CORS ----------

# Set ALLOWED_ORIGINS via wrangler.toml [vars] (comma-separated) or fall back
# to a sensible default for the portfolio site.
DEFAULT_ALLOWED = "https://andreaslindeman.com,https://andreaslindeman.no,https://ai.andreaslindeman.com,https://ai.andreaslindeman.no"


def _allowed_origin(request, env) -> str:
    raw = getattr(env, "ALLOWED_ORIGINS", None) or DEFAULT_ALLOWED
    allowed = [o.strip() for o in raw.split(",") if o.strip()]
    origin = request.headers.get("origin", "") or ""
    if origin in allowed:
        return origin
    # During local dev / when no Origin header is sent, echo the first allowed.
    return allowed[0] if allowed else "*"


def _cors_headers(request, env):
    return {
        "access-control-allow-origin": _allowed_origin(request, env),
        "access-control-allow-methods": "GET,POST,OPTIONS",
        "access-control-allow-headers": "content-type",
        "access-control-max-age": "86400",
        "vary": "origin",
    }


def _json_response(payload, request, env, *, status=200):
    headers = _cors_headers(request, env)
    headers["content-type"] = "application/json"
    return Response(json.dumps(payload), status=status, headers=headers)


def _error(message, request, env, *, status=400):
    return _json_response({"error": message}, request, env, status=status)


# ---------- helpers ----------

def _hash_ip(ip: str, salt: str) -> str:
    if not ip:
        return ""
    return hmac.new(salt.encode("utf-8"), ip.encode("utf-8"), hashlib.sha256).hexdigest()


def _truthy(v) -> int:
    return 1 if bool(v) else 0


async def _read_json(request):
    try:
        body = await request.json()
    except Exception:
        raise ValueError("body must be JSON")
    # On Pyodide, `request.json()` typically returns a JsProxy. Convert to a
    # plain Python dict so the rest of the code can treat it normally.
    try:
        return body.to_py()
    except AttributeError:
        return body


def _get(d, key, default=None):
    """Tolerate both Python dicts and JS-proxy dict-likes."""
    try:
        v = d.get(key)
    except AttributeError:
        v = getattr(d, key, None)
    return v if v is not None else default


# ---------- /color-polygraph/survey ----------

async def _handle_submit(request, env):
    try:
        body = await _read_json(request)
    except ValueError as exc:
        return _error(str(exc), request, env, status=400)

    payload = _get(body, "payload") or {}
    metadata = _get(body, "metadata") or {}

    try:
        validate_payload(payload)
    except ValueError as exc:
        return _error(f"invalid payload: {exc}", request, env, status=400)

    submit_unix = int(_get(metadata, "client_submitted_at", 0) or time.time() * 1000) // 1000

    try:
        features = compute_features(payload, submit_unix)
    except Exception as exc:
        return _error(f"feature extraction failed: {exc}", request, env, status=500)

    survey_id = str(uuid.uuid4())
    server_received_at = int(time.time() * 1000)

    ip = request.headers.get("cf-connecting-ip", "") or ""
    salt = getattr(env, "IP_HASH_SALT", None) or "no-salt-set"
    ip_hash = _hash_ip(ip, salt)

    cf = getattr(request, "cf", None)
    country = getattr(cf, "country", None) if cf else None
    region  = getattr(cf, "region",  None) if cf else None
    city    = getattr(cf, "city",    None) if cf else None
    tz_cf   = getattr(cf, "timezone", None) if cf else None

    # Persist
    await env.DB.prepare("""
        INSERT INTO surveys (
            id, server_received_at, client_started_at, client_submitted_at, client_local_time,
            offered_json, r1_json, r2_json, final_color_json, valg, tider_json,
            user_agent, referrer, language, locale, is_mobile,
            screen_w, screen_h, viewport_w, viewport_h, timezone_client,
            ip_hash, country, region, city, timezone_cf
        ) VALUES (?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?)
    """).bind(
        survey_id, server_received_at,
        _get(metadata, "client_started_at"),
        _get(metadata, "client_submitted_at"),
        _get(metadata, "client_local_time"),
        json.dumps(payload["offered"]),
        json.dumps(payload["r1"]),
        json.dumps(payload["r2"]),
        json.dumps(payload["final"]),
        payload["valg"],
        json.dumps(payload["tider"]),
        _get(metadata, "user_agent"),
        _get(metadata, "referrer"),
        _get(metadata, "language"),
        _get(metadata, "locale"),
        _truthy(_get(metadata, "is_mobile")),
        _get(metadata, "screen_w"),
        _get(metadata, "screen_h"),
        _get(metadata, "viewport_w"),
        _get(metadata, "viewport_h"),
        _get(metadata, "timezone_client"),
        ip_hash,
        country, region, city, tz_cf,
    ).run()

    return _json_response({"id": survey_id, "features": features}, request, env)


# ---------- /color-polygraph/survey/:id/{gender,mood,age} ----------

async def _handle_gender_confirm(request, env, survey_id):
    try:
        body = await _read_json(request)
    except ValueError as exc:
        return _error(str(exc), request, env, status=400)

    confirmed = _get(body, "confirmed_label")
    if confirmed not in ("man", "woman"):
        return _error("confirmed_label must be 'man' or 'woman'", request, env, status=400)
    try:
        pred_prob_f = float(_get(body, "pred_prob"))
    except (TypeError, ValueError):
        return _error("pred_prob must be a number", request, env, status=400)
    confirmed_int = 1 if confirmed == "woman" else 0

    result = await env.DB.prepare("""
        UPDATE surveys
           SET pred_gender_prob = ?,
               confirmed_gender = ?
         WHERE id = ?
    """).bind(pred_prob_f, confirmed_int, survey_id).run()

    if not _row_was_updated(result):
        return _error("survey not found", request, env, status=404)
    return _json_response({"ok": True}, request, env)


async def _handle_mood_confirm(request, env, survey_id):
    try:
        body = await _read_json(request)
    except ValueError as exc:
        return _error(str(exc), request, env, status=400)

    try:
        pred_value = float(_get(body, "pred_value"))
        confirmed = int(_get(body, "confirmed_value"))
    except (TypeError, ValueError):
        return _error("pred_value and confirmed_value must be numbers", request, env, status=400)
    if not (0 <= confirmed <= 60):
        return _error("confirmed_value must be 0..60", request, env, status=400)

    result = await env.DB.prepare("""
        UPDATE surveys
           SET pred_mood = ?,
               confirmed_mood = ?
         WHERE id = ?
    """).bind(pred_value, confirmed, survey_id).run()

    if not _row_was_updated(result):
        return _error("survey not found", request, env, status=404)
    return _json_response({"ok": True}, request, env)


async def _handle_age_confirm(request, env, survey_id):
    try:
        body = await _read_json(request)
    except ValueError as exc:
        return _error(str(exc), request, env, status=400)

    try:
        pred_value = float(_get(body, "pred_value"))
        confirmed = int(_get(body, "confirmed_value"))
    except (TypeError, ValueError):
        return _error("pred_value and confirmed_value must be numbers", request, env, status=400)
    if not (6 <= confirmed <= 99):
        return _error("confirmed_value must be 6..99", request, env, status=400)

    result = await env.DB.prepare("""
        UPDATE surveys
           SET pred_age = ?,
               confirmed_age = ?
         WHERE id = ?
    """).bind(pred_value, confirmed, survey_id).run()

    if not _row_was_updated(result):
        return _error("survey not found", request, env, status=404)
    return _json_response({"ok": True}, request, env)


async def _handle_get_survey(request, env, survey_id):
    """Return stored survey result for shareable link (GET /survey/:id)."""
    try:
        row = await env.DB.prepare("""
            SELECT final_color_json, r1_json, r2_json, tider_json, valg,
                   confirmed_gender, pred_gender_prob,
                   confirmed_age,    pred_age,
                   confirmed_mood,   pred_mood
              FROM surveys WHERE id = ?
        """).bind(survey_id).first()
    except Exception as exc:
        return _error(f"db error: {exc}", request, env, status=500)

    if row is None:
        return _error("survey not found", request, env, status=404)

    # Pyodide returns a JsProxy — convert to plain dict.
    try:
        row = row.to_py()
    except AttributeError:
        pass

    def _parse(key):
        raw = row.get(key) if isinstance(row, dict) else getattr(row, key, None)
        if raw is None:
            return None
        if isinstance(raw, str):
            return json.loads(raw)
        return raw

    def _val(key):
        return row.get(key) if isinstance(row, dict) else getattr(row, key, None)

    final = _parse("final_color_json")
    r1    = _parse("r1_json") or []
    r2    = _parse("r2_json") or []
    tider = _parse("tider_json") or []
    valg  = _val("valg") or ""

    if not final or len(final) < 3:
        return _error("survey data incomplete", request, env, status=422)

    def _color(rgb):
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        return {"rgb": [r, g, b], "hex": "#{:02x}{:02x}{:02x}".format(r, g, b)}

    winner = _color(final)

    # Simplified history — enough to reconstruct the hue strip.
    history = [
        [{"winner": _color(c), "options": []} for c in r1 if c],
        [{"winner": _color(c), "options": []} for c in r2 if c],
        [{"winner": winner, "options": []}],
    ]

    picks = [
        {"position": int(valg[i]), "cumulativeMs": tider[i]}
        for i in range(min(len(valg), len(tider)))
    ]

    confirmed_gender_int = _val("confirmed_gender")
    confirmed_gender = (
        "woman" if confirmed_gender_int == 1
        else "man" if confirmed_gender_int == 0
        else None
    )
    confirmed_age  = _val("confirmed_age")
    confirmed_mood = _val("confirmed_mood")
    pred_gender_prob = _val("pred_gender_prob")
    pred_age   = _val("pred_age")
    pred_mood  = _val("pred_mood")

    correct = None
    if confirmed_gender is not None and confirmed_age is not None and confirmed_mood is not None:
        correct = 0
        if pred_gender_prob is not None:
            pred_label = "woman" if float(pred_gender_prob) >= 0.5 else "man"
            if confirmed_gender == pred_label:
                correct += 1
        if pred_age is not None and abs(int(confirmed_age) - float(pred_age)) <= 3:
            correct += 1
        if pred_mood is not None and abs(int(confirmed_mood) - float(pred_mood)) <= 10:
            correct += 1

    return _json_response({
        "id": survey_id,
        "winner_hex": winner["hex"],
        "winner_rgb": winner["rgb"],
        "confirmed_gender": confirmed_gender,
        "confirmed_age":    int(confirmed_age)  if confirmed_age  is not None else None,
        "confirmed_mood":   int(confirmed_mood) if confirmed_mood is not None else None,
        "correct_count": correct,
        "history": history,
        "picks": picks,
    }, request, env)


def _row_was_updated(result) -> bool:
    """D1 returns a meta object with `changes` in JS. Pyodide returns a JsProxy."""
    try:
        meta = result.meta
    except AttributeError:
        meta = None
    if meta is None:
        return True  # be permissive if the driver shape changes
    changes = getattr(meta, "changes", None) or getattr(meta, "rows_written", None)
    if changes is None:
        return True
    return int(changes) > 0


# ---------- router ----------

async def on_fetch(request, env, ctx=None):
    url = request.url
    # Cloudflare's URL object can be used directly; Pyodide exposes .pathname.
    # Fallback to string parsing if the JS URL helper is unavailable.
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path
    except Exception:
        path = url

    method = request.method.upper() if hasattr(request, "method") else "GET"

    if method == "OPTIONS":
        return Response("", status=204, headers=_cors_headers(request, env))

    if path == "/color-polygraph/survey" and method == "POST":
        return await _handle_submit(request, env)

    # /color-polygraph/survey/{id}  — GET returns stored result for share link
    if path.startswith("/color-polygraph/survey/") and method == "GET":
        parts = path.strip("/").split("/")
        if len(parts) == 3:
            survey_id = parts[2]
            return await _handle_get_survey(request, env, survey_id)

    # /color-polygraph/survey/{id}/{step}
    if path.startswith("/color-polygraph/survey/") and method == "POST":
        parts = path.strip("/").split("/")
        if len(parts) == 4:
            _, _, survey_id, step = parts
            if step == "gender":
                return await _handle_gender_confirm(request, env, survey_id)
            if step == "mood":
                return await _handle_mood_confirm(request, env, survey_id)
            if step == "age":
                return await _handle_age_confirm(request, env, survey_id)

    if path in ("/", "/health"):
        return _json_response({"ok": True, "service": "color-polygraph-api"}, request, env)

    return _error("not found", request, env, status=404)
