"""
Train the *long* (256-color) Color Polygraph models from synthetic long
sessions built out of real short sessions.

A long survey is structurally four short surveys stacked plus one extra final
question (long = 4 x short + 1). So we manufacture long training rows by joining
four short sessions that share gender + age + (closely) mood:

    long round 0 (64 q)  = the four shorts' round-0 questions   (offered 256, r1 64)
    long round 1 (16 q)  = the four shorts' round-1 questions   (r2 16)
    long round 2 (4 q)   = the four shorts' final questions     (r3 4 = the 4 short winners)
    long round 3 (1 q)   = a synthetic pick among those 4 winners (final)

The pick order is kept round-major so the assembled valg / tider / colour lists
match exactly what the live long survey emits, which lets the same
`features_long.py` extractor run at train and serve time.

Label per long row: gender (shared), age (shared), mood = mean of the 4 moods.

Grouping ("ca the same mood"): within each (gender, age) cell, sort by mood and
take consecutive groups of 4. Cells whose size is not a multiple of 4 are padded
by duplicating (sampling with replacement) sessions from the same cell — i.e.
"if there isn't enough data for that age/gender, duplicate the data".

Pipeline mirrors `lgb-production/train_and_emit.py`: champion LightGBM params,
RGB bucket-score totals, a two-pass eval-then-refit, and a bit-exact JSON tree
emit verified against LightGBM.

Emits:
    models-js/{gender,age,mood}_long_trees.json
    cloudflare/bucket_data_long.py
    training/long-models/summary.json   (metrics, next to this script)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.normpath(os.path.join(HERE, ".."))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
CF_DIR = os.path.join(PROJECT_ROOT, "cloudflare")
# Trees ship from the static site, alongside the short trees + tree_walker.js
# that survey-result.html fetches via ./models-js/.
JS_OUT_DIR = os.path.normpath(
    os.path.join(PROJECT_ROOT, "..", "english_html", "color-polygraph", "models-js"))
RAW_SOURCE = os.path.join(TRAINING_DIR, "raw", "save.ligma")
os.makedirs(JS_OUT_DIR, exist_ok=True)

# Use the SAME extractor the worker serves with (single source of truth).
sys.path.insert(0, CF_DIR)
import features_long as fl  # noqa: E402
import features as sf       # noqa: E402  short extractor (for the short reference)
import math                 # noqa: E402

# Shared validity + troll filter for the short sessions we synthesise from,
# and for any real long sessions folded in.
sys.path.insert(0, TRAINING_DIR)
from data_cleaning import (  # noqa: E402
    is_valid_clean, is_valid_long_clean, long_payload_to_shorts, long_is_holdout,
    dedupe_short_rows)


def _load_short_trees():
    """Load the deployed short gender/age/mood trees (None if not yet built)."""
    out = {}
    for name in ("gender", "age", "mood"):
        path = os.path.join(JS_OUT_DIR, f"{name}_trees.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            out[name] = json.load(fh)["trees"]
    return out


def _walk(trees, f):
    total = 0.0
    for tr in trees:
        i = 0
        while tr[i * 4] != -1:
            i = tr[i * 4 + 2] if f[tr[i * 4]] <= tr[i * 4 + 1] else tr[i * 4 + 3]
        total += tr[i * 4 + 1]
    return total


def _short_reference(test_payloads, test_labels, age_true, mood_true):
    """How well the SHORT model does on the same held-out real longs, run on
    each long's 4 sub-surveys and averaged. This is the honest same-population
    comparison the leaderboard shows next to the long model. The short trees are
    trained on genuine shorts only (decomposition off), so they never saw these
    longs -> leakage-free. Returns None if the short trees aren't built yet."""
    trees = _load_short_trees()
    if trees is None:
        return None
    probs, ages, moods = [], [], []
    for p, lab in zip(test_payloads, test_labels):
        gp, ap, mp = [], [], []
        for row in long_payload_to_shorts(p, lab, "r"):
            pay = {"offered": row[8][0], "r1": row[8][1], "r2": row[8][2],
                   "final": row[8][3], "valg": row[6], "tider": row[7]}
            fv = sf.compute_features(pay, lab["time"])
            gp.append(1.0 / (1.0 + math.exp(-_walk(trees["gender"], fv["gender"]))))
            ap.append(_walk(trees["age"], fv["age"]))
            mp.append(_walk(trees["mood"], fv["mood"]))
        probs.append(float(np.mean(gp)))
        ages.append(float(np.mean(ap)))
        moods.append(float(np.mean(mp)))
    yv = np.array([int(l["gender"]) for l in test_labels])
    return {
        "kind": "short model, mean of the 4 sub-surveys, on the same held-out real longs",
        "gender_auc": float(roc_auc_score(yv, probs)),
        "age_mae": float(mean_absolute_error(age_true, ages)),
        "mood_mae": float(mean_absolute_error(mood_true, moods)),
    }

# Real long sessions pulled from the live DB (written by the refresh
# orchestrator). They are rare, so we still synthesise the bulk from short
# quads, but real rows get a heavier sample weight so the models focus on
# genuine long behaviour. Override the weight with CP_REAL_LONG_WEIGHT.
REAL_LONG_SOURCE = os.path.join(TRAINING_DIR, "raw", "long_real.json")
REAL_LONG_WEIGHT = float(os.environ.get("CP_REAL_LONG_WEIGHT", "3.0"))

# Honest long eval: hold out a fraction of the REAL long surveys as the test
# set (so the reported metrics reflect genuine long performance, not the
# synthetic short-quad rows whose duplicated padding would otherwise leak into
# the test fold). We keep most real longs in training so they are not
# "sacrificed" to the test set. Needs a minimum count to be meaningful;
# below it we fall back to the old stratified-over-all split.
LONG_TEST_REAL_FRAC = float(os.environ.get("CP_LONG_TEST_FRAC", "0.30"))
MIN_REAL_FOR_HOLDOUT = 30

SEED = 42
SHORT_N_R1 = 16   # short round-0 questions
SHORT_N_R2 = 4    # short round-1 questions
QUAD = 4          # four shorts -> one long
N_BUCKETS = fl.N_BUCKETS

DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000
VAL_SIZE = 0.10   # fraction of long rows held out for metrics

CHAMPION_CLF = dict(
    n_estimators=800, num_leaves=63, learning_rate=0.03,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=20, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbosity=-1,
)
CHAMPION_REG = dict(
    n_estimators=1000, num_leaves=63, learning_rate=0.03,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=20, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbosity=-1,
)


# ---------- load + validate short sessions ----------

def _is_valid(row):
    # Shared filter for the short sessions the synthetic long rows are built from.
    return is_valid_clean(row)


def _load_real_long():
    """Load real long sessions pulled from the live DB, if any.

    Returns (payloads, labels) for rows that pass the long troll filter. The
    file is a JSON list of {"payload": {...}, "label": {...}} written by the
    refresh orchestrator; absent (e.g. before the first live pull) -> empty.
    """
    if not os.path.exists(REAL_LONG_SOURCE):
        return [], []
    with open(REAL_LONG_SOURCE, encoding="utf-8") as fh:
        items = json.load(fh)
    payloads, labels = [], []
    for it in items:
        payload, label = it.get("payload"), it.get("label")
        if payload is None or label is None:
            continue
        if not is_valid_long_clean(payload, label):
            continue
        payloads.append(payload)
        labels.append(label)
    return payloads, labels


def _parse_short(row):
    """Pull the bits we need out of a raw short session row."""
    tider = [int(x) for x in row[7][:21]]
    deltas = [max(0, tider[0])] + [max(0, tider[i] - tider[i - 1]) for i in range(1, 21)]
    return {
        "time": int(row[1]) if str(row[1]).lstrip("-").isdigit() else 0,
        "gender": 1 if row[5] == "j" else 0,
        "age": int(row[3]),
        "mood": int(row[4]),
        "offered": [list(c) for c in row[8][0][:64]],   # round-0 options (64)
        "r1": [list(c) for c in row[8][1][:16]],         # round-0 winners (16)
        "r2": [list(c) for c in row[8][2][:4]],          # round-1 winners (4)
        "final": list(row[8][3]),                        # round-2 winner
        "valg": row[6][:21],                             # 16 + 4 + 1 picks
        "deltas": deltas,                                # per-question ms (21)
    }


# ---------- assemble one long session from a quad of shorts ----------

def _assemble_long(quad, rng):
    """Stack four shorts into one long session, round-major, matching the live
    long survey's payload layout."""
    offered, r1, r2, r3 = [], [], [], []
    v0, v1, v2 = [], [], []          # round-0/1/2 pick digits
    d0, d1, d2 = [], [], []          # round-0/1/2 per-question deltas

    for s in quad:
        offered.extend(s["offered"])            # 4 x 64 -> 256
        r1.extend(s["r1"])                       # 4 x 16 -> 64
        r2.extend(s["r2"])                       # 4 x 4  -> 16
        r3.append(s["final"])                    # 4 x 1  -> 4

        v0.append(s["valg"][0:SHORT_N_R1])                       # 16 round-0 picks
        v1.append(s["valg"][SHORT_N_R1:SHORT_N_R1 + SHORT_N_R2]) # 4 round-1 picks
        v2.append(s["valg"][SHORT_N_R1 + SHORT_N_R2])            # 1 final pick

        d0.extend(s["deltas"][0:SHORT_N_R1])
        d1.extend(s["deltas"][SHORT_N_R1:SHORT_N_R1 + SHORT_N_R2])
        d2.append(s["deltas"][SHORT_N_R1 + SHORT_N_R2])

    # The synthetic +1 question: pick one of the 4 finalists.
    final_idx = int(rng.randint(0, QUAD))
    final = r3[final_idx]
    final_delta = float(np.mean(d2))   # plausible dwell on the last screen

    valg = "".join(v0) + "".join(v1) + "".join(v2) + str(final_idx)   # 64+16+4+1
    deltas = d0 + d1 + d2 + [final_delta]                             # 85
    tider, run = [], 0.0
    for d in deltas:
        run += d
        tider.append(int(run))

    payload = {
        "offered": offered, "r1": r1, "r2": r2, "r3": r3, "final": final,
        "valg": valg, "tider": tider,
    }
    label = {
        "gender": quad[0]["gender"],
        "age": quad[0]["age"],
        "mood": float(np.mean([s["mood"] for s in quad])),
        "time": quad[0]["time"],
    }
    return payload, label


def _build_long_sessions(shorts, rng):
    """Group shorts by (gender, age), mood-sort, pad to multiples of 4, chunk."""
    cells = {}
    for s in shorts:
        cells.setdefault((s["gender"], s["age"]), []).append(s)

    payloads, labels = [], []
    n_padded = 0
    for key, group in sorted(cells.items()):
        group = sorted(group, key=lambda s: s["mood"])
        pad = (-len(group)) % QUAD
        if pad:
            extra = [group[i] for i in rng.randint(0, len(group), size=pad)]
            group = sorted(group + extra, key=lambda s: s["mood"])
            n_padded += pad
        for i in range(0, len(group), QUAD):
            quad = group[i:i + QUAD]
            p, l = _assemble_long(quad, rng)
            payloads.append(p)
            labels.append(l)
    return payloads, labels, n_padded


# ---------- duplicate-short synthesis (chosen over blending) ----------
# Blending four DIFFERENT people into one synthetic long makes an unrealistic
# "four-person" long that classifies too easily and transfers poorly to real
# (single-person) longs. Duplicating ONE short four times keeps the synthetic
# long single-person, matching what a real long actually is. On a leakage-free
# 30%-real-long holdout this beat blending on gender AUC (0.775 vs 0.766).

def _dup_short_to_long(s):
    """Replicate one short session 4x into a single-person long payload+label."""
    v0, v1, vf = s["valg"][0:16], s["valg"][16:20], s["valg"][20]
    d0, d1, df = s["deltas"][0:16], s["deltas"][16:20], s["deltas"][20]
    payload = {
        "offered": [list(c) for c in s["offered"]] * 4,   # 256
        "r1": [list(c) for c in s["r1"]] * 4,               # 64
        "r2": [list(c) for c in s["r2"]] * 4,               # 16
        "r3": [list(s["final"]) for _ in range(4)],         # 4 finalists (all same)
        "final": list(s["final"]),
        "valg": v0 * 4 + v1 * 4 + vf * 4 + "0",             # 64 + 16 + 4 + 1
    }
    deltas = d0 * 4 + d1 * 4 + [df] * 4 + [float(df)]
    tider, run = [], 0.0
    for d in deltas:
        run += d
        tider.append(int(run))
    payload["tider"] = tider
    label = {"gender": s["gender"], "age": s["age"], "mood": s["mood"], "time": s["time"]}
    return payload, label


def _build_dup_long_sessions(shorts):
    """One synthetic long per short (duplicate-short). No padding, no blending."""
    payloads, labels = [], []
    for s in shorts:
        p, l = _dup_short_to_long(s)
        payloads.append(p)
        labels.append(l)
    return payloads, labels, 0


# ---------- LGB -> flat JSON tree emit (copied from lgb-production) ----------

def _flatten_tree_quads(node):
    nodes_flat = [None, None, None, None]
    stack = [(node, 0)]
    while stack:
        n, slot = stack.pop()
        base = slot * 4
        if "leaf_index" in n:
            nodes_flat[base] = -1
            nodes_flat[base + 1] = float(n["leaf_value"])
            nodes_flat[base + 2] = 0
            nodes_flat[base + 3] = 0
            continue
        left_idx = len(nodes_flat) // 4
        nodes_flat.extend([None, None, None, None])
        right_idx = len(nodes_flat) // 4
        nodes_flat.extend([None, None, None, None])
        nodes_flat[base] = int(n["split_feature"])
        nodes_flat[base + 1] = float(n["threshold"])
        nodes_flat[base + 2] = left_idx
        nodes_flat[base + 3] = right_idx
        stack.append((n["right_child"], right_idx))
        stack.append((n["left_child"], left_idx))
    return nodes_flat


def _emit_tree_json(booster, out_path, objective):
    dump = booster.dump_model()
    n_features = int(dump["max_feature_idx"]) + 1
    trees, total_nodes = [], 0
    for tree_info in dump["tree_info"]:
        flat = _flatten_tree_quads(tree_info["tree_structure"])
        trees.append(flat)
        total_nodes += len(flat) // 4
    payload = {
        "objective": objective, "n_features": n_features,
        "n_trees": len(trees), "n_nodes": total_nodes, "trees": trees,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return len(trees), total_nodes


def _verify_json(out_path, booster, X_sample):
    with open(out_path, encoding="utf-8") as fh:
        model = json.load(fh)

    def js_like_score(features):
        total = 0.0
        for tree in model["trees"]:
            i = 0
            while tree[i * 4] != -1:
                feat = tree[i * 4]
                thr = tree[i * 4 + 1]
                i = tree[i * 4 + 2] if features[feat] <= thr else tree[i * 4 + 3]
            total += tree[i * 4 + 1]
        return total

    raw_lgb = booster.predict(X_sample, raw_score=True)
    js_scores = np.array([js_like_score(list(map(float, row))) for row in X_sample],
                         dtype=np.float64)
    return float(np.max(np.abs(raw_lgb - js_scores)))


def _emit_bucket_data_long(out_path, grids):
    girly, masc, age_g, mood_g = grids

    def lst(a):
        return repr(np.asarray(a, dtype=float).ravel().tolist())

    parts = [
        '"""Auto-generated long-survey bucket-score grids (8x8x8 = 512 buckets).\n',
        'Built from the synthetic long training rows in train_long.py."""\n\n',
        f'GIRLY_GRID_LONG = {lst(girly)}\n\n',
        f'MASC_GRID_LONG  = {lst(masc)}\n\n',
        f'AGE_GRID_LONG   = {lst(age_g)}\n\n',
        f'MOOD_GRID_LONG  = {lst(mood_g)}\n',
    ]
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))


# ---------- main ----------

def main():
    t_total = time.time()
    rng = np.random.RandomState(SEED)

    print("Loading short sessions...")
    with open(RAW_SOURCE, encoding="utf-8") as fh:
        rows = json.load(fh)
    rows = dedupe_short_rows(rows)  # one row per unique survey (same as the short models)
    shorts = [_parse_short(r) for r in rows if _is_valid(r)]
    print(f"  {len(shorts)} valid short sessions")

    print("Synthesising long sessions (duplicate-short: one short replicated 4x)...")
    payloads, labels, n_padded = _build_dup_long_sessions(shorts)
    n_synth = len(payloads)

    # Fold in real long sessions from the live DB (heavier sample weight).
    real_payloads, real_labels = _load_real_long()
    n_real = len(real_payloads)
    payloads = payloads + real_payloads
    labels = labels + real_labels

    N = len(payloads)
    g = np.array([l["gender"] for l in labels], dtype=np.int8)
    a = np.array([l["age"] for l in labels], dtype=np.float32)
    m = np.array([l["mood"] for l in labels], dtype=np.float32)
    # Synthetic rows weigh 1.0; real long rows weigh REAL_LONG_WEIGHT so the
    # models lean on genuine long behaviour despite being outnumbered.
    weight = np.ones(N, dtype=np.float32)
    weight[n_synth:] = REAL_LONG_WEIGHT
    print(f"  {N} long rows = {n_synth} synthetic ({n_padded} padded) + "
          f"{n_real} real (weight {REAL_LONG_WEIGHT:g})  "
          f"girls={int((g == 1).sum())} boys={int((g == 0).sum())}  "
          f"age {a.mean():.1f}  mood {m.mean():.1f}")

    print("Extracting long features (shared serve/train extractor)...")
    t0 = time.time()
    X_static = np.array(
        [fl._extract_static_long(p, labels[i]["time"]) for i, p in enumerate(payloads)],
        dtype=np.float32,
    )
    discrete = np.zeros((N, N_BUCKETS), dtype=np.float32)
    smooth = np.zeros((N, N_BUCKETS), dtype=np.float32)
    for i, p in enumerate(payloads):
        d, s = fl.compute_bucket_delta_long(p)
        discrete[i] = d
        smooth[i] = s
    print(f"  X_static {X_static.shape}  ({time.time() - t0:.1f}s)")

    # ---- bucket grid helpers ----
    def build_grids(idx):
        d_sub, g_sub, a_sub, m_sub = discrete[idx], g[idx], a[idx], m[idx]
        girly = d_sub[g_sub == 1].mean(axis=0)
        masc = d_sub[g_sub == 0].mean(axis=0)
        age_g = ((a_sub - a_sub.mean())[:, None] * d_sub).mean(axis=0)
        mood_g = ((m_sub - m_sub.mean())[:, None] * d_sub).mean(axis=0)
        return girly, masc, age_g, mood_g

    def stack(idx, grids):
        girly, masc, age_g, mood_g = grids
        sm = smooth[idx]
        girly_t = sm @ girly
        masc_t = sm @ masc
        age_t = sm @ age_g
        mood_t = sm @ mood_g
        X_gen = np.concatenate(
            [X_static[idx], girly_t[:, None], masc_t[:, None], (girly_t - masc_t)[:, None]], axis=1)
        X_age = np.concatenate([X_static[idx], age_t[:, None]], axis=1)
        X_mood = np.concatenate([X_static[idx], mood_t[:, None]], axis=1)
        return X_gen, X_age, X_mood

    # ---- Pass 1: eval split ----
    all_idx = np.arange(N)
    synth_idx = np.arange(0, n_synth)
    real_idx = np.arange(n_synth, N)
    if n_real >= MIN_REAL_FOR_HOLDOUT:
        # Honest test = a FROZEN ~30% slice of the REAL long surveys, chosen by
        # a hash of each session's content so the same longs stay in test every
        # retrain (comparable metrics across versions). Train = all synthetic +
        # the remaining real. No synthetic (hence no duplicated padding) ever
        # reaches the test fold, so the metrics are not inflated by leakage, and
        # ~70% of the real longs still train the model.
        real_hold = np.array([long_is_holdout(p) for p in real_payloads])
        r_te = real_idx[real_hold]
        r_tr = real_idx[~real_hold]
        # Degenerate fold (too few, or one gender only) -> fall back to the old
        # seeded stratified slice so the eval still produces a usable AUC.
        if len(r_te) < 5 or len(set(g[r_te].tolist())) < 2:
            r_tr, r_te = train_test_split(
                real_idx, test_size=LONG_TEST_REAL_FRAC, random_state=SEED, stratify=g[real_idx])
        tr = np.sort(np.concatenate([synth_idx, r_tr]))
        va = np.sort(r_te)
        eval_kind = (f"frozen content-hashed real-long holdout: ~{int(LONG_TEST_REAL_FRAC * 100)}% "
                     f"of {n_real} real longs as test, train = all synthetic + remaining real")
    else:
        tr, va = train_test_split(all_idx, test_size=VAL_SIZE, random_state=SEED, stratify=g)
        tr, va = np.sort(tr), np.sort(va)
        eval_kind = (f"stratified split over all rows (only {n_real} real longs, "
                     f"< {MIN_REAL_FOR_HOLDOUT} needed for a real holdout)")
    print(f"\nPass 1 - eval ({eval_kind})  train={len(tr)}  val={len(va)}")
    grids_eval = build_grids(tr)
    Xg_tr, Xa_tr, Xm_tr = stack(tr, grids_eval)
    Xg_va, Xa_va, Xm_va = stack(va, grids_eval)

    clf = lgb.LGBMClassifier(**CHAMPION_CLF).fit(Xg_tr, g[tr], sample_weight=weight[tr])
    p_g = clf.predict_proba(Xg_va)[:, 1]
    auc_g = roc_auc_score(g[va], p_g)
    pred = (p_g >= 0.5).astype(int)
    acc_g, f1_g = accuracy_score(g[va], pred), f1_score(g[va], pred)
    print(f"  GENDER  AUC={auc_g:.4f}  acc={acc_g:.4f}  F1={f1_g:.4f}")

    reg_a = lgb.LGBMRegressor(**CHAMPION_REG).fit(Xa_tr, a[tr], sample_weight=weight[tr])
    p_a = reg_a.predict(Xa_va)
    mae_a, r2_a = mean_absolute_error(a[va], p_a), r2_score(a[va], p_a)
    print(f"  AGE     MAE={mae_a:.3f}  R2={r2_a:+.3f}")

    reg_m = lgb.LGBMRegressor(**CHAMPION_REG).fit(Xm_tr, m[tr], sample_weight=weight[tr])
    p_m = reg_m.predict(Xm_va)
    mae_m, r2_m = mean_absolute_error(m[va], p_m), r2_score(m[va], p_m)
    print(f"  MOOD    MAE={mae_m:.3f}  R2={r2_m:+.3f}")

    # Short-model reference on the SAME held-out real longs (only meaningful
    # when the test fold is real longs, i.e. the real-long holdout path).
    short_ref = None
    if n_real >= MIN_REAL_FOR_HOLDOUT:
        short_ref = _short_reference([payloads[i] for i in va],
                                     [labels[i] for i in va], a[va], m[va])
        if short_ref:
            print(f"  SHORT-REF (same real longs)  gender AUC={short_ref['gender_auc']:.4f}  "
                  f"age MAE={short_ref['age_mae']:.2f}  mood MAE={short_ref['mood_mae']:.2f}")

    # ---- Pass 2: refit on all rows + emit ----
    print("\nPass 2 - refit on all rows + emit ...")
    grids_full = build_grids(all_idx)
    Xg, Xa, Xm = stack(all_idx, grids_full)
    prod_clf = lgb.LGBMClassifier(**CHAMPION_CLF).fit(Xg, g, sample_weight=weight)
    prod_reg_a = lgb.LGBMRegressor(**CHAMPION_REG).fit(Xa, a, sample_weight=weight)
    prod_reg_m = lgb.LGBMRegressor(**CHAMPION_REG).fit(Xm, m, sample_weight=weight)

    sample_idx = rng.choice(N, min(16, N), replace=False)
    cfg = [
        ("gender_long", prod_clf.booster_, "binary", Xg[sample_idx]),
        ("age_long", prod_reg_a.booster_, "regression", Xa[sample_idx]),
        ("mood_long", prod_reg_m.booster_, "regression", Xm[sample_idx]),
    ]
    emit_stats = {}
    print("Emitting JSON trees (models-js/) ...")
    for name, booster, objective, X_sample in cfg:
        out = os.path.join(JS_OUT_DIR, f"{name}_trees.json")
        n_trees, n_nodes = _emit_tree_json(booster, out, objective)
        delta = _verify_json(out, booster, X_sample)
        if delta > 1e-5:
            raise SystemExit(f"{name} JSON diverged from LightGBM by {delta:.3e}")
        kb = os.path.getsize(out) / 1024
        emit_stats[name] = {"json_kb": round(kb, 2), "n_trees": n_trees,
                            "n_nodes": n_nodes, "max_emit_delta": float(delta)}
        print(f"  {name:11s}  {kb:7.1f} KB  trees={n_trees} nodes={n_nodes}  |delta|<{delta:.1e}")

    bucket_path = os.path.join(CF_DIR, "bucket_data_long.py")
    _emit_bucket_data_long(bucket_path, grids_full)
    print(f"  bucket_data_long.py  {os.path.getsize(bucket_path) / 1024:.1f} KB")

    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "kind": "long survey (256 colors) models",
        "data": {
            "source": "duplicate-short synthetic longs (one short replicated 4x) + real long DB rows (heavier weight)",
            "n_short_sessions": len(shorts),
            "n_long_rows": int(N),
            "n_synthetic": int(n_synth),
            "n_real": int(n_real),
            "real_long_weight": REAL_LONG_WEIGHT,
            "n_padded_shorts": int(n_padded),
            "quad_size": QUAD,
            "label_mood": "mean of the 4 short moods",
        },
        "validation": {"kind": eval_kind, "test_real_frac": LONG_TEST_REAL_FRAC,
                       "n_train": int(len(tr)), "n_val": int(len(va)),
                       "n_real_total": int(n_real), "seed": SEED,
                       "short_reference": short_ref},
        "n_features": {"gender": int(Xg.shape[1]), "age": int(Xa.shape[1]),
                       "mood": int(Xm.shape[1])},
        "validation_scores": {
            "gender_auc": float(auc_g), "gender_accuracy": float(acc_g), "gender_f1": float(f1_g),
            "age_mae": float(mae_a), "age_r2": float(r2_a),
            "mood_mae": float(mae_m), "mood_r2": float(r2_m),
        },
        "champion_clf_params": CHAMPION_CLF,
        "champion_reg_params": CHAMPION_REG,
        "emit_stats": emit_stats,
        "seed": SEED,
    }
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nValidation (stratified {int((1-VAL_SIZE)*100)}/{int(VAL_SIZE*100)} split, seed={SEED})")
    print(f"  GENDER AUC = {auc_g:.4f}  acc = {acc_g:.4f}  F1 = {f1_g:.4f}")
    print(f"  AGE    MAE = {mae_a:.3f}  R2 = {r2_a:+.3f}")
    print(f"  MOOD   MAE = {mae_m:.3f}  R2 = {r2_m:+.3f}")
    print(f"\nArtifacts: {JS_OUT_DIR} (trees), {CF_DIR}/bucket_data_long.py, {HERE}/summary.json")
    print(f"Total wall time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
