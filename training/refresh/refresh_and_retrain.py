"""
Color Polygraph end-to-end refresh + retrain orchestrator.

One command takes the live database snapshot pulled by pull_remote.py, folds
the clean responses into the training data, retrains every production model
(short + long gender/age/mood AND short + long colour-pick), emits the JSON
trees to the deployed models-js/ folder, bumps the version, and rewrites the
leaderboards in both languages.

Pipeline:

  1. INGEST   remote_dump.json (if present) -> clean with data_cleaning ->
              append new SHORT rows to raw/save.ligma (deduped by id, backup
              first) and merge new LONG rows into raw/long_real.json.
  2. FEATURES rebuild features.npy / targets.npz (features.py) and
              features_extra.npy in lockstep (same row selection/order).
  3. TRAIN    run the four trainers as subprocesses:
                lgb-production/train_and_emit.py     (short gender/age/mood)
                long-models/train_long.py            (long gender/age/mood)
                taste-cube/train_pick.py             (short colour-pick)
                taste-cube/train_pick_long.py        (long colour-pick)
  4. VERSION  bump version.json (+0.1, or +1.0 with --major) and record the
              new production metrics from lgb-production/summary.json.
  5. PUBLISH  regenerate the leaderboard regions (version rows + colour-pick
              table) in the EN and NO index.html between their HTML markers.

Run on the current data (no live pull needed) to validate the whole chain:

    python refresh_and_retrain.py

After a live pull:

    python pull_remote.py            # writes remote_dump.json
    python refresh_and_retrain.py    # ingests it, retrains, republishes

Flags:
    --major                 bump the major version (+1.0) instead of +0.1
    --version X.Y           set the new version explicitly
    --real-long-weight W    weight for real long rows (default 3.0)
    --skip-train            rebuild data + republish HTML only (no retrain)
    --dump PATH             ingest a specific dump (default: remote_dump.json)
"""

import argparse
import datetime
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.normpath(os.path.join(HERE, ".."))
PROJECT_ROOT = os.path.normpath(os.path.join(TRAINING_DIR, ".."))   # ai/color-polygraph
AI_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, ".."))         # ai/

RAW_SOURCE = os.path.join(TRAINING_DIR, "raw", "save.ligma")
LONG_REAL = os.path.join(TRAINING_DIR, "raw", "long_real.json")
SHORT_FROM_LONG = os.path.join(TRAINING_DIR, "raw", "short_from_long.json")
EXTRA_DIR = os.path.join(TRAINING_DIR, "extra-features")
VERSION_FILE = os.path.join(HERE, "version.json")
DEFAULT_DUMP = os.path.join(HERE, "remote_dump.json")

LGB_PROD = os.path.join(TRAINING_DIR, "lgb-production", "train_and_emit.py")
LONG_MODELS = os.path.join(TRAINING_DIR, "long-models", "train_long.py")
PICK = os.path.join(TRAINING_DIR, "taste-cube", "train_pick.py")
PICK_LONG = os.path.join(TRAINING_DIR, "taste-cube", "train_pick_long.py")

PROD_SUMMARY = os.path.join(TRAINING_DIR, "lgb-production", "summary.json")
LONG_SUMMARY = os.path.join(TRAINING_DIR, "long-models", "summary.json")
PICK_SUMMARY = os.path.join(TRAINING_DIR, "taste-cube", "pick_summary.json")
PICK_LONG_SUMMARY = os.path.join(TRAINING_DIR, "taste-cube", "pick_long_summary.json")

EN_INDEX = os.path.join(AI_ROOT, "english_html", "color-polygraph", "index.html")
NO_INDEX = os.path.join(AI_ROOT, "norwegian_html", "color-polygraph", "index.html")

sys.path.insert(0, TRAINING_DIR)
import data_cleaning as dc  # noqa: E402


# ====================== 1. INGEST ======================

def _existing_ids():
    with open(RAW_SOURCE, encoding="utf-8") as fh:
        rows = json.load(fh)
    return rows, {str(r[0]) for r in rows if r}


def ingest(dump_path):
    """Fold a live DB dump into save.ligma (short) + long_real.json (long).
    Returns a stats dict. No-op friendly: a missing dump = dry run."""
    stats = {"dump": dump_path, "new_short": 0, "new_long": 0,
             "dropped_troll": 0, "dup_skipped": 0}
    if not os.path.exists(dump_path):
        print(f"  no dump at {dump_path} -> dry run on existing data only")
        return stats

    with open(dump_path, encoding="utf-8") as fh:
        snapshot = json.load(fh)
    db_rows = snapshot.get("rows", [])
    print(f"  dump has {len(db_rows)} rows (pulled {snapshot.get('pulled_at')})")

    rows, existing = _existing_ids()

    # Existing real long rows (keyed by id for dedup across runs).
    long_by_id = {}
    if os.path.exists(LONG_REAL):
        with open(LONG_REAL, encoding="utf-8") as fh:
            for it in json.load(fh):
                if it.get("id") is not None:
                    long_by_id[str(it["id"])] = it

    new_short_rows = []
    for db in db_rows:
        rid = str(db.get("id", ""))
        is_long = bool(db.get("long_survey"))
        if is_long:
            conv = dc.db_row_to_long(db)
            if conv is None:
                stats["dropped_troll"] += 1
                continue
            payload, label = conv
            if not dc.is_valid_long_clean(payload, label):
                stats["dropped_troll"] += 1
                continue
            if rid in long_by_id:
                stats["dup_skipped"] += 1
                continue
            long_by_id[rid] = {"id": rid, "payload": payload, "label": label}
            stats["new_long"] += 1
        else:
            if rid in existing:
                stats["dup_skipped"] += 1
                continue
            raw = dc.db_row_to_short_raw(db)
            if raw is None or not dc.is_valid_clean(raw):
                stats["dropped_troll"] += 1
                continue
            new_short_rows.append(raw)
            existing.add(rid)
            stats["new_short"] += 1

    # Persist short rows (backup first).
    if new_short_rows:
        shutil.copy2(RAW_SOURCE, RAW_SOURCE + ".bak")
        rows.extend(new_short_rows)
        with open(RAW_SOURCE, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        print(f"  appended {len(new_short_rows)} short rows to save.ligma "
              f"(backup -> save.ligma.bak)")

    # Persist long rows.
    if stats["new_long"] or os.path.exists(LONG_REAL):
        with open(LONG_REAL, "w", encoding="utf-8") as fh:
            json.dump(list(long_by_id.values()), fh)
        print(f"  long_real.json now holds {len(long_by_id)} real long rows")

    print(f"  ingest: +{stats['new_short']} short, +{stats['new_long']} long, "
          f"{stats['dropped_troll']} trolls dropped, {stats['dup_skipped']} dups skipped")
    return stats


# ====================== 1b. DECOMPOSE LONG -> SHORT ======================

def decompose_longs(enabled):
    """Regenerate raw/short_from_long.json from every real long session, so the
    short models train on real shorts derived from long sessions too. Mirror of
    the short->long synthesis. Fully regenerated each run (ids are <long>#k), so
    it is idempotent and picks up longs added in earlier runs."""
    if not enabled:
        # Remove any stale file so load_short_rows() can't accidentally fold it
        # in (and so synthetic-long construction never sees real-long-derived shorts).
        if os.path.exists(SHORT_FROM_LONG):
            os.remove(SHORT_FROM_LONG)
        print("  decomposition off; short models use genuine surveys only (save.ligma)")
        return
    if not os.path.exists(LONG_REAL):
        print("  no real long rows yet -> nothing to decompose")
        return
    with open(LONG_REAL, encoding="utf-8") as fh:
        longs = json.load(fh)
    rows, kept, dropped = [], 0, 0
    for it in longs:
        payload, label = it.get("payload"), it.get("label")
        if payload is None or label is None:
            continue
        for raw in dc.long_payload_to_shorts(payload, label, it.get("id", "L")):
            if dc.is_valid_clean(raw):
                rows.append(raw)
                kept += 1
            else:
                dropped += 1
    with open(SHORT_FROM_LONG, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    print(f"  decomposed {len(longs)} longs -> {kept} short rows "
          f"({dropped} failed the troll/duration filter) -> short_from_long.json")


# ====================== 2. FEATURES ======================

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rebuild_features():
    """Rebuild features.npy/targets.npz (features.py) then features_extra.npy
    in lockstep (identical row selection + order)."""
    print("  rebuilding features.npy / targets.npz ...")
    _run([sys.executable, os.path.join(TRAINING_DIR, "features.py")], cwd=TRAINING_DIR)

    print("  rebuilding features_extra.npy (no CV) ...")
    extra = _load_module(os.path.join(EXTRA_DIR, "train.py"), "extra_train")
    raw_rows = dc.load_short_rows()   # same source + order as features.py
    X_extra, names = [], None
    for row in raw_rows:
        if not extra.is_valid(row):
            continue
        try:
            nm, vals = extra.extract_extra(row)
        except Exception:
            continue
        if names is None:
            names = nm
        elif nm != names:
            continue
        X_extra.append(vals)
    X_extra = np.array(X_extra, dtype=np.float32)
    n_base = np.load(os.path.join(TRAINING_DIR, "features.npy")).shape[0]
    if X_extra.shape[0] != n_base:
        raise SystemExit(
            f"feature row mismatch: features.npy={n_base} vs extra={X_extra.shape[0]} "
            "(the two validity passes disagree)")
    np.save(os.path.join(EXTRA_DIR, "features_extra.npy"), X_extra)
    with open(os.path.join(EXTRA_DIR, "feature_names_extra.json"), "w", encoding="utf-8") as fh:
        json.dump(names, fh, indent=2)
    print(f"  features_extra.npy {X_extra.shape}  (matches {n_base} base rows)")


# ====================== 3. TRAIN ======================

def _run(cmd, cwd=None, env=None):
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    res = subprocess.run(cmd, cwd=cwd, env=env)
    if res.returncode != 0:
        raise SystemExit(f"command failed ({res.returncode}): {' '.join(map(str, cmd))}")


def train_all(real_long_weight):
    env = dict(os.environ, CP_REAL_LONG_WEIGHT=str(real_long_weight))
    print("  [1/4] short gender/age/mood (lgb-production) ...")
    _run([sys.executable, LGB_PROD], cwd=os.path.dirname(LGB_PROD))
    print("  [2/4] long gender/age/mood (long-models) ...")
    _run([sys.executable, LONG_MODELS], cwd=os.path.dirname(LONG_MODELS), env=env)
    print("  [3/4] short colour-pick (taste-cube) ...")
    _run([sys.executable, PICK], cwd=os.path.dirname(PICK))
    print("  [4/4] long colour-pick (taste-cube) ...")
    _run([sys.executable, PICK_LONG], cwd=os.path.dirname(PICK_LONG))


# ====================== 4. VERSION ======================

def _read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def bump_version(args, n_rows):
    vinfo = _read_json(VERSION_FILE) or {"current": "1.0", "entries": []}
    cur = float(vinfo.get("current", "1.0"))
    if args.version:
        new = float(args.version)
    elif args.major:
        new = float(int(cur) + 1)
    else:
        new = round(cur + 0.1, 1)
    new_str = f"{new:.1f}"

    prod = _read_json(PROD_SUMMARY) or {}
    scores = prod.get("validation_scores", {})
    entry = {
        "version": new_str,
        "trained_at": datetime.date.today().isoformat(),
        "n_rows": int(prod.get("n_total_rows", n_rows) or n_rows),
        "gender_auc": round(float(scores.get("gender_auc", 0.0)), 3),
        "age_mae": round(float(scores.get("age_mae", 0.0)), 2),
        "mood_mae": round(float(scores.get("mood_mae", 0.0)), 2),
    }
    # Worldwide gate (5-fold CV over genuine new sessions) — the number we steer
    # by. Recorded only once there are enough worldwide rows for the gate to run,
    # so legacy-only versions stay absent from the worldwide view.
    wscores = prod.get("worldwide_scores", {})
    if wscores.get("gender_auc") is not None:
        entry["world_n_rows"] = int(wscores.get("n", 0))
        entry["world_gender_auc"] = round(float(wscores["gender_auc"]), 3)
        entry["world_age_mae"] = round(float(wscores.get("age_mae", 0.0)), 2)
        entry["world_mood_mae"] = round(float(wscores.get("mood_mae", 0.0)), 2)
    # Long-model metrics (honest real-long holdout). Recorded only when the long
    # trainer actually had real longs to hold out, so historical short-only
    # versions stay absent from the long view rather than showing fake numbers.
    lng = _read_json(LONG_SUMMARY) or {}
    lscores = lng.get("validation_scores", {})
    lval = lng.get("validation", {})
    if lscores and lval.get("n_real_total", 0) >= 30:
        entry["long_n_rows"] = int(lval.get("n_real_total", 0))
        entry["long_gender_auc"] = round(float(lscores.get("gender_auc", 0.0)), 3)
        entry["long_age_mae"] = round(float(lscores.get("age_mae", 0.0)), 2)
        entry["long_mood_mae"] = round(float(lscores.get("mood_mae", 0.0)), 2)
        sref = lval.get("short_reference")
        if sref:
            entry["short_ref_gender_auc"] = round(float(sref.get("gender_auc", 0.0)), 3)
            entry["short_ref_age_mae"] = round(float(sref.get("age_mae", 0.0)), 2)
            entry["short_ref_mood_mae"] = round(float(sref.get("mood_mae", 0.0)), 2)
    # Colour-pick metrics (short + long), recorded per version so the pick
    # leaderboard becomes a version history like gender/age/mood. Stored only
    # when the trainer actually produced a score, so a --skip-train run or a
    # missing summary leaves the version absent from the pick view rather than
    # writing zeros.
    pick, pick_long = _pick_metrics()
    if pick.get("auc") is not None:
        entry["pick_acc"] = round(float(pick.get("pick_acc") or 0.0), 3)
        entry["pick_auc"] = round(float(pick["auc"]), 3)
        entry["pick_leak_gate"] = round(float(pick.get("leak_gate") or 0.0), 3)
    if pick_long.get("auc") is not None:
        entry["pick_long_acc"] = round(float(pick_long.get("pick_acc") or 0.0), 3)
        entry["pick_long_auc"] = round(float(pick_long["auc"]), 3)
        entry["pick_long_leak_gate"] = round(float(pick_long.get("leak_gate") or 0.0), 3)
    # Replace any existing entry with this version, then prepend.
    entries = [e for e in vinfo.get("entries", []) if e.get("version") != new_str]
    entries.insert(0, entry)
    # Newest-first by numeric version.
    entries.sort(key=lambda e: float(e["version"]), reverse=True)
    vinfo = {"current": new_str, "note": vinfo.get("note", ""), "entries": entries}
    with open(VERSION_FILE, "w", encoding="utf-8") as fh:
        json.dump(vinfo, fh, indent=2)
    print(f"  version {cur:.1f} -> {new_str}   "
          f"(gender {entry['gender_auc']}, age {entry['age_mae']}, mood {entry['mood_mae']})")
    return vinfo


# ====================== 5. PUBLISH ======================

def _fmt(v, nd):
    return f"{float(v):.{nd}f}"


def _versions_rows(entries, lang):
    label_base = "LightGBM (production)" if lang == "en" else "LightGBM (produksjon)"
    best_auc = max(e["gender_auc"] for e in entries)
    best_age = min(e["age_mae"] for e in entries)
    best_mood = min(e["mood_mae"] for e in entries)
    out = []
    for e in entries:
        def cell(val, nd, is_best):
            cls = "lb-score lb-score-best" if is_best else "lb-score"
            return f'              <td class="{cls}">{_fmt(val, nd)}</td>'
        out.append(
            "            <tr>\n"
            f'              <td><a class="lb-arch-link" href="models/lgb-production">'
            f'<span class="lb-arch">{label_base} v{e["version"]}</span></a></td>\n'
            f'{cell(e["gender_auc"], 3, e["gender_auc"] == best_auc)}\n'
            f'{cell(e["age_mae"], 2, e["age_mae"] == best_age)}\n'
            f'{cell(e["mood_mae"], 2, e["mood_mae"] == best_mood)}\n'
            "            </tr>"
        )
    return "\n".join(out)


def _versions_long_rows(entries, lang):
    label_base = "LightGBM long (production)" if lang == "en" else "LightGBM lang (produksjon)"
    have = [e for e in entries if e.get("long_gender_auc") is not None]
    if not have:
        span = "Long survey models retrain on the next refresh." if lang == "en" \
            else "Lange modeller trenes på neste oppdatering."
        return ('            <tr>\n'
                f'              <td><span class="lb-arch lb-arch-plain">{span}</span></td>\n'
                '              <td class="lb-score">&mdash;</td>\n'
                '              <td class="lb-score">&mdash;</td>\n'
                '              <td class="lb-score">&mdash;</td>\n'
                '            </tr>')
    best_auc = max(e["long_gender_auc"] for e in have)
    best_age = min(e["long_age_mae"] for e in have)
    best_mood = min(e["long_mood_mae"] for e in have)
    out = []
    for e in have:
        def cell(val, nd, is_best):
            cls = "lb-score lb-score-best" if is_best else "lb-score"
            return f'              <td class="{cls}">{_fmt(val, nd)}</td>'
        out.append(
            "            <tr>\n"
            f'              <td><a class="lb-arch-link" href="models/lgb-production">'
            f'<span class="lb-arch">{label_base} v{e["version"]}</span></a></td>\n'
            f'{cell(e["long_gender_auc"], 3, e["long_gender_auc"] == best_auc)}\n'
            f'{cell(e["long_age_mae"], 2, e["long_age_mae"] == best_age)}\n'
            f'{cell(e["long_mood_mae"], 2, e["long_mood_mae"] == best_mood)}\n'
            "            </tr>"
        )
    # Reference row: the short model scored on the SAME real long surveys, so the
    # long survey's advantage on its own population is visible (long >= short).
    newest = have[0]
    if newest.get("short_ref_gender_auc") is not None:
        ref = "Short model, same people" if lang == "en" else "Kort modell, samme folk"
        out.append(
            '            <tr>\n'
            f'              <td><span class="lb-arch lb-arch-plain">{ref}</span></td>\n'
            f'              <td class="lb-score lb-na">{_fmt(newest["short_ref_gender_auc"], 3)}</td>\n'
            f'              <td class="lb-score lb-na">{_fmt(newest["short_ref_age_mae"], 2)}</td>\n'
            f'              <td class="lb-score lb-na">{_fmt(newest["short_ref_mood_mae"], 2)}</td>\n'
            "            </tr>"
        )
    return "\n".join(out)


def _versions_world_rows(entries, lang):
    """Worldwide-cohort version history (5-fold CV over genuine new sessions).
    The number we actually steer by, since the frozen hold-out is ~92% the legacy
    2020 Oslo cohort."""
    label_base = "LightGBM worldwide v" if lang == "en" else "LightGBM verden v"
    have = [e for e in entries if e.get("world_gender_auc") is not None]
    if not have:
        span = "Worldwide gate fills on the next refresh." if lang == "en" \
            else "Verdensmål fylles ved neste oppdatering."
        return ('            <tr>\n'
                f'              <td><span class="lb-arch lb-arch-plain">{span}</span></td>\n'
                '              <td class="lb-score">&mdash;</td>\n'
                '              <td class="lb-score">&mdash;</td>\n'
                '              <td class="lb-score">&mdash;</td>\n'
                '            </tr>')
    best_auc = max(e["world_gender_auc"] for e in have)
    best_age = min(e["world_age_mae"] for e in have)
    best_mood = min(e["world_mood_mae"] for e in have)
    out = []
    for e in have:
        def cell(val, nd, is_best):
            cls = "lb-score lb-score-best" if is_best else "lb-score"
            return f'              <td class="{cls}">{_fmt(val, nd)}</td>'
        out.append(
            "            <tr>\n"
            f'              <td><a class="lb-arch-link" href="models/lgb-production">'
            f'<span class="lb-arch">{label_base}{e["version"]}</span></a></td>\n'
            f'{cell(e["world_gender_auc"], 3, e["world_gender_auc"] == best_auc)}\n'
            f'{cell(e["world_age_mae"], 2, e["world_age_mae"] == best_age)}\n'
            f'{cell(e["world_mood_mae"], 2, e["world_mood_mae"] == best_mood)}\n'
            "            </tr>"
        )
    return "\n".join(out)


def _pick_rows(entries, lang):
    """Per-version colour-pick history (short + long), newest first. Mirrors the
    gender/age/mood version table: one short row and one long row per version
    that recorded pick metrics. Best pick-accuracy and best AUC across the table
    are highlighted; leak gate is left unhighlighted (it should sit near 0.25,
    so "higher/lower" is not "better")."""
    short_name = "Colour pick short" if lang == "en" else "Fargevalg kort"
    long_name = "Colour pick long" if lang == "en" else "Fargevalg lang"

    rows_data = []  # (label, acc, auc, gate)
    for e in entries:
        v = e["version"]
        if e.get("pick_auc") is not None:
            rows_data.append((f"{short_name} v{v}",
                              e.get("pick_acc"), e.get("pick_auc"), e.get("pick_leak_gate")))
        if e.get("pick_long_auc") is not None:
            rows_data.append((f"{long_name} v{v}",
                              e.get("pick_long_acc"), e.get("pick_long_auc"), e.get("pick_long_leak_gate")))

    if not rows_data:
        span = "Colour-pick models retrain on the next refresh." if lang == "en" \
            else "Fargevalgmodeller trenes på neste oppdatering."
        return ('            <tr>\n'
                f'              <td><span class="lb-arch lb-arch-plain">{span}</span></td>\n'
                '              <td class="lb-score">&mdash;</td>\n'
                '              <td class="lb-score">&mdash;</td>\n'
                '              <td class="lb-score">&mdash;</td>\n'
                '            </tr>')

    accs = [a for _, a, _, _ in rows_data if a is not None]
    aucs = [u for _, _, u, _ in rows_data if u is not None]
    best_acc = max(accs) if accs else None
    best_auc = max(aucs) if aucs else None

    def cell(v, is_best):
        if v is None:
            return '              <td class="lb-score">&mdash;</td>'
        cls = "lb-score lb-score-best" if is_best else "lb-score"
        return f'              <td class="{cls}">{_fmt(v, 3)}</td>'

    out = []
    for label, acc, auc, gate in rows_data:
        out.append(
            "            <tr>\n"
            f'              <td><span class="lb-arch lb-arch-plain">{label}</span></td>\n'
            f'{cell(acc, acc is not None and acc == best_acc)}\n'
            f'{cell(auc, auc is not None and auc == best_auc)}\n'
            f'{cell(gate, False)}\n'
            "            </tr>"
        )
    return "\n".join(out)


def _replace_region(html, name, inner):
    pat = re.compile(
        r"(<!-- LB:" + name + r":START.*?-->)(.*?)(\n\s*<!-- LB:" + name + r":END -->)",
        re.DOTALL)
    if not pat.search(html):
        raise SystemExit(f"marker LB:{name} not found in HTML")
    return pat.sub(lambda m: m.group(1) + "\n" + inner + m.group(3), html)


def _pick_metrics():
    ps = _read_json(PICK_SUMMARY) or {}
    res = ps.get("results", {}).get("B_person_cand_inter", {})
    pick = {"pick_acc": res.get("pick_acc"), "auc": res.get("auc"),
            "leak_gate": res.get("leak_gate")}
    pls = _read_json(PICK_LONG_SUMMARY) or {}
    val = pls.get("validation", {})
    pick_long = {"pick_acc": val.get("pick_accuracy"), "auc": val.get("auc"),
                 "leak_gate": val.get("leak_gate")}
    return pick, pick_long


def publish(vinfo):
    pick, pick_long = _pick_metrics()
    today = datetime.date.today().isoformat()
    for path, lang, label in ((EN_INDEX, "en", "Last refresh"),
                              (NO_INDEX, "no", "Sist oppdatert")):
        with open(path, encoding="utf-8") as fh:
            html = fh.read()
        html = _replace_region(html, "VERSIONS", _versions_rows(vinfo["entries"], lang))
        html = _replace_region(html, "VERSIONS_WORLD", _versions_world_rows(vinfo["entries"], lang))
        html = _replace_region(html, "VERSIONS_LONG", _versions_long_rows(vinfo["entries"], lang))
        html = _replace_region(html, "PICK", _pick_rows(vinfo["entries"], lang))
        html = re.sub(rf"({label}: )\d{{4}}-\d{{2}}-\d{{2}}", rf"\g<1>{today}", html)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"  updated {os.path.relpath(path, AI_ROOT)}")
    print(f"  pick (short): {pick}")
    print(f"  pick (long):  {pick_long}")


# ====================== main ======================

def main():
    ap = argparse.ArgumentParser(description="Refresh data + retrain + republish.")
    ap.add_argument("--dump", default=DEFAULT_DUMP, help="live DB dump to ingest")
    ap.add_argument("--major", action="store_true", help="bump major version (+1.0)")
    ap.add_argument("--version", help="set new version explicitly, e.g. 1.2")
    ap.add_argument("--real-long-weight", default="3.0",
                    help="sample weight for real long rows (default 3.0)")
    ap.add_argument("--decompose-long", dest="decompose_long",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Split real long sessions into short rows (4 per long) and "
                         "feed them to the short models as extra TRAINING data "
                         "(held train-only; short_is_holdout rejects them, so the val "
                         "folds stay genuine). Default ON: the long respondents match "
                         "the current WORLDWIDE population, and a leak-free era-sliced "
                         "ablation showed this measurably improves the worldwide cohort "
                         "(age MAE 7.01->6.29, gender 0.858->0.876) -- it only looked "
                         "worse before because the metric was dominated by the legacy "
                         "2020 Oslo cohort. Use --no-decompose-long for genuine-only.")
    ap.add_argument("--skip-train", action="store_true",
                    help="rebuild data + republish only (no model training)")
    args = ap.parse_args()

    # The short-model trainers (subprocesses) inherit this; it gates whether
    # data_cleaning.load_short_rows() folds in the decomposed long->short rows.
    os.environ["CP_INCLUDE_DECOMPOSED"] = "1" if args.decompose_long else "0"

    print("== 1. INGEST ==")
    ingest(args.dump)
    decompose_longs(args.decompose_long)

    print("== 2. FEATURES ==")
    rebuild_features()
    n_rows = int(np.load(os.path.join(TRAINING_DIR, "features.npy")).shape[0])
    print(f"  training rows: {n_rows}")

    if not args.skip_train:
        print("== 3. TRAIN ==")
        train_all(args.real_long_weight)

    print("== 4. VERSION ==")
    vinfo = bump_version(args, n_rows)

    print("== 5. PUBLISH ==")
    publish(vinfo)

    print(f"\nDone. Production is now v{vinfo['current']} "
          f"({n_rows} short training rows).")


if __name__ == "__main__":
    main()
