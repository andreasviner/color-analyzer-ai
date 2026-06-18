"""
Train the production LightGBM + bucket-scores model and emit the artifacts
the live survey actually needs:

    models-js/{gender,age,mood}_trees.json   JSON flat-tree the browser
                                             reads via tree_walker.js
    cloudflare/bucket_data.py                per-pool 8x8x8 grid lookups
                                             consumed by the worker's
                                             features.py
    summary.json                             leaderboard-row metrics
                                             (lives next to this script)

Hyperparameters match `color-buckets/train.py` exactly (the leaderboard
champion: 800 trees, num_leaves=63, lr=0.03 for the classifier, 1000
trees for the regressors). The trees ship client-side as JavaScript, so
Cloudflare's bundle limit doesn't bind.

Two-pass procedure:

  Pass 1 — EVALUATION
    Shuffle the 6,710 valid sessions (stratified on gender, seed=42) and
    split 6,000 / 710. Build per-pool bucket grids from the 6,000-row
    training fold, train all three heads, and report gender AUC / age MAE /
    mood MAE on the 710-row validation fold. These are the numbers that go
    on the leaderboard row — random shuffle (not temporal order) so the
    validation distribution matches the training distribution, avoiding the
    cohort drift the previous temporal hold-out caught on age and mood.

  Pass 2 — DEPLOYMENT
    Refit all three heads on the FULL 6,710 rows, this time with bucket
    grids built from every row. These are the boosters that get emitted
    to `models-js/` and the grids that get emitted to
    `cloudflare/bucket_data.py`. The deployed model sees strictly more
    training data than the eval model, so its true generalisation should
    be no worse than what pass 1 reports.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    r2_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.normpath(os.path.join(HERE, ".."))
EXTRA_DIR = os.path.join(TRAINING_DIR, "extra-features")
BUCKETS_DIR = os.path.join(TRAINING_DIR, "color-buckets")
RAW_SOURCE = os.path.join(TRAINING_DIR, "raw", "save.ligma")

PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
CF_DIR = os.path.join(PROJECT_ROOT, "cloudflare")
# Production trees ship from the static site that survey-result.html fetches
# via ./models-js/ (same dir as the long + pick trees). PROJECT_ROOT is
# ai/color-polygraph, the deployed site is ai/english_html/color-polygraph.
JS_OUT_DIR = os.path.normpath(
    os.path.join(PROJECT_ROOT, "..", "english_html", "color-polygraph", "models-js"))
os.makedirs(JS_OUT_DIR, exist_ok=True)

# Shared validity + troll filter (must match features.py so the re-parsed
# bucket vectors line up row-for-row with features.npy).
sys.path.insert(0, TRAINING_DIR)
from data_cleaning import is_valid_clean  # noqa: E402

SEED = 42
N_R1 = 16
N_QUESTIONS = 21
GRID = 8
N_BUCKETS = GRID ** 3
BUCKET_WIDTH = 256 / GRID
BUCKET_CENTER_OFFSET = BUCKET_WIDTH / 2
PICK_VALUE = 0.1 * 16 * 3 / N_QUESTIONS
NOT_PICK_VALUE = -0.1

DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000

VAL_SIZE = 710  # last N rows = temporal validation hold-out

# Champion hyperparameters — identical to color-buckets/train.py
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


# ---------- Validation + bucket helpers (same as color-buckets/train.py) ----------

def _is_valid(row):
    # Shared filter — identical selection to features.py / features.npy.
    return is_valid_clean(row)


def _bucket_id(r, g, b):
    return r * GRID * GRID + g * GRID + b


def _discrete_bucket(rgb):
    r = min(GRID - 1, int(rgb[0]) // int(BUCKET_WIDTH))
    g = min(GRID - 1, int(rgb[1]) // int(BUCKET_WIDTH))
    b = min(GRID - 1, int(rgb[2]) // int(BUCKET_WIDTH))
    return _bucket_id(r, g, b)


def _trilinear_weights(rgb):
    fr = max(0.0, min(GRID - 1, (rgb[0] - BUCKET_CENTER_OFFSET) / BUCKET_WIDTH))
    fg = max(0.0, min(GRID - 1, (rgb[1] - BUCKET_CENTER_OFFSET) / BUCKET_WIDTH))
    fb = max(0.0, min(GRID - 1, (rgb[2] - BUCKET_CENTER_OFFSET) / BUCKET_WIDTH))
    ir, ig, ib = int(fr), int(fg), int(fb)
    dr, dg, db = fr - ir, fg - ig, fb - ib
    out = []
    for ox, wr in ((0, 1.0 - dr), (1, dr)):
        if wr == 0:
            continue
        br = min(GRID - 1, ir + ox)
        for oy, wg in ((0, 1.0 - dg), (1, dg)):
            if wg == 0:
                continue
            bg = min(GRID - 1, ig + oy)
            for oz, wb in ((0, 1.0 - db), (1, db)):
                if wb == 0:
                    continue
                bb = min(GRID - 1, ib + oz)
                out.append((_bucket_id(br, bg, bb), wr * wg * wb))
    return out


def _compute_deltas(row):
    offered = row[8][0]
    r1 = row[8][1]
    r2 = row[8][2]
    final = row[8][3]
    valg = row[6]
    picked = set()
    for q in range(N_R1):
        try:
            idx = int(valg[q])
            if 0 <= idx <= 3:
                picked.add(q * 4 + idx)
        except (ValueError, IndexError):
            pass
    discrete = np.zeros(N_BUCKETS, dtype=np.float32)
    smooth = np.zeros(N_BUCKETS, dtype=np.float32)

    def add_event(c, v):
        discrete[_discrete_bucket(c)] += v
        for b, w in _trilinear_weights(c):
            smooth[b] += v * w

    for i in range(64):
        if i not in picked:
            add_event(offered[i], NOT_PICK_VALUE)
    for c in r1: add_event(c, PICK_VALUE)
    for c in r2: add_event(c, PICK_VALUE)
    add_event(final, PICK_VALUE)
    return discrete, smooth


# ---------- LGB -> flat-tree emitter (JSON only — Python source is no
# longer emitted because the worker doesn't run inference; the browser does) ----------

def _flatten_tree_quads(node):
    """Same flatten but emits a single flat list of 4*N entries — what the
    JS tree walker expects (positional layout, 4 entries per node)."""
    nodes_flat = [None, None, None, None]
    stack = [(node, 0)]
    while stack:
        n, slot = stack.pop()
        base = slot * 4
        if "leaf_index" in n:
            nodes_flat[base]     = -1
            nodes_flat[base + 1] = float(n["leaf_value"])
            nodes_flat[base + 2] = 0
            nodes_flat[base + 3] = 0
            continue
        left_idx = len(nodes_flat) // 4
        nodes_flat.extend([None, None, None, None])
        right_idx = len(nodes_flat) // 4
        nodes_flat.extend([None, None, None, None])
        nodes_flat[base]     = int(n["split_feature"])
        nodes_flat[base + 1] = float(n["threshold"])
        nodes_flat[base + 2] = left_idx
        nodes_flat[base + 3] = right_idx
        stack.append((n["right_child"], right_idx))
        stack.append((n["left_child"], left_idx))
    return nodes_flat


def _emit_tree_json(booster: lgb.Booster, out_path: str, objective: str):
    dump = booster.dump_model()
    n_features = int(dump["max_feature_idx"]) + 1
    trees = []
    total_nodes = 0
    for tree_info in dump["tree_info"]:
        flat = _flatten_tree_quads(tree_info["tree_structure"])
        trees.append(flat)
        total_nodes += len(flat) // 4
    payload = {
        "objective": objective,
        "n_features": n_features,
        "n_trees": len(trees),
        "n_nodes": total_nodes,
        "trees": trees,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return len(trees), total_nodes


def _verify_json(out_path: str, booster: lgb.Booster, X_sample: np.ndarray):
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


# ---------- bucket_data.py emission (matches old layout) ----------

def _emit_bucket_data(out_path, grids):
    girly_grid, masc_grid, age_grid, mood_grid = grids
    def lst(a):
        return repr(a.astype(float).ravel().tolist())
    parts = []
    parts.append('"""Auto-generated bucket-score grids (8x8x8 = 512 buckets each).\n')
    parts.append('Computed from the first 6,000 of 6,710 valid sessions (the same\n')
    parts.append('training pool the production booster was fit on)."""\n\n')
    parts.append(f'GRID = {GRID}\n')
    parts.append(f'N_BUCKETS = {N_BUCKETS}\n')
    parts.append(f'BUCKET_WIDTH = {int(BUCKET_WIDTH)}\n')
    parts.append(f'BUCKET_CENTER_OFFSET = {int(BUCKET_CENTER_OFFSET)}\n')
    parts.append(f'PICK_VALUE = {PICK_VALUE}\n')
    parts.append(f'NOT_PICK_VALUE = {NOT_PICK_VALUE}\n\n')
    parts.append(f'GIRLY_GRID = {lst(girly_grid)}\n\n')
    parts.append(f'MASC_GRID  = {lst(masc_grid)}\n\n')
    parts.append(f'AGE_GRID   = {lst(age_grid)}\n\n')
    parts.append(f'MOOD_GRID  = {lst(mood_grid)}\n')
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))


# ---------- main ----------

def main():
    t_total = time.time()

    print("Loading static features + targets...")
    X_base  = np.load(os.path.join(TRAINING_DIR, "features.npy"))
    X_extra = np.load(os.path.join(EXTRA_DIR, "features_extra.npy"))
    X_static = np.concatenate([X_base, X_extra], axis=1).astype(np.float32)
    targets = np.load(os.path.join(TRAINING_DIR, "targets.npz"))
    g = targets["gender"].astype(np.int8)
    a = targets["age"].astype(np.int32)
    m = targets["mood"].astype(np.int32)
    N = X_static.shape[0]
    print(f"  X_static {X_static.shape}  gender boys/girls "
          f"{int((g == 0).sum())}/{int((g == 1).sum())}")

    print("Computing per-session bucket vectors (8x8x8 RGB)...")
    t0 = time.time()
    with open(RAW_SOURCE, encoding="utf-8") as fh:
        rows = json.load(fh)
    discrete_list, smooth_list = [], []
    for row in rows:
        if not _is_valid(row):
            continue
        d, s = _compute_deltas(row)
        discrete_list.append(d)
        smooth_list.append(s)
    discrete = np.array(discrete_list, dtype=np.float32)
    smooth   = np.array(smooth_list,   dtype=np.float32)
    print(f"  shapes {discrete.shape}, {smooth.shape}  ({time.time()-t0:.1f}s)")
    assert discrete.shape[0] == N

    # ---------- Pass 1: random shuffled eval split ----------
    # Stratified-by-gender 6000 / 710 split so the val fold matches the
    # train fold's gender ratio. Used purely for reporting metrics.
    all_idx = np.arange(N)
    train_idx, val_idx = train_test_split(
        all_idx, test_size=VAL_SIZE, random_state=SEED, stratify=g,
    )
    train_idx = np.sort(train_idx)
    val_idx   = np.sort(val_idx)
    print(f"\nPass 1 - random stratified eval split")
    print(f"  train: {len(train_idx)} rows  ({g[train_idx].mean():.3f} girls, "
          f"age {a[train_idx].mean():.1f}, mood {m[train_idx].mean():.1f})")
    print(f"  val:   {len(val_idx)} rows  ({g[val_idx].mean():.3f} girls, "
          f"age {a[val_idx].mean():.1f}, mood {m[val_idx].mean():.1f})")

    def _build_grids_from(indices):
        d_sub  = discrete[indices]
        sm_sub = smooth[indices]
        g_sub  = g[indices]
        a_sub  = a[indices]
        m_sub  = m[indices]
        girly = d_sub[g_sub == 1].mean(axis=0)
        masc  = d_sub[g_sub == 0].mean(axis=0)
        age_g = ((a_sub - a_sub.mean()).astype(np.float32)[:, None] * d_sub).mean(axis=0)
        mood_g = ((m_sub - m_sub.mean()).astype(np.float32)[:, None] * d_sub).mean(axis=0)
        return girly, masc, age_g, mood_g

    def _stack_features(indices, girly, masc, age_g, mood_g):
        sm_sub = smooth[indices]
        girly_t = sm_sub @ girly
        masc_t  = sm_sub @ masc
        signed_t = girly_t - masc_t
        age_t   = sm_sub @ age_g
        mood_t  = sm_sub @ mood_g
        X_gen = np.concatenate([X_static[indices],
                                girly_t[:, None], masc_t[:, None], signed_t[:, None]], axis=1)
        X_age = np.concatenate([X_static[indices], age_t[:, None]], axis=1)
        X_mood = np.concatenate([X_static[indices], mood_t[:, None]], axis=1)
        return X_gen, X_age, X_mood

    # Build grids from the eval-train fold only (no leakage on eval-val).
    girly_grid_eval, masc_grid_eval, age_grid_eval, mood_grid_eval = _build_grids_from(train_idx)
    X_g_tr,  X_a_tr,  X_m_tr  = _stack_features(train_idx,
                                                girly_grid_eval, masc_grid_eval,
                                                age_grid_eval,  mood_grid_eval)
    X_g_val, X_a_val, X_m_val = _stack_features(val_idx,
                                                girly_grid_eval, masc_grid_eval,
                                                age_grid_eval,  mood_grid_eval)

    print("\nFitting eval models (gender / age / mood, train n=6000)...")
    t0 = time.time()
    eval_clf = lgb.LGBMClassifier(**CHAMPION_CLF)
    eval_clf.fit(X_g_tr, g[train_idx])
    p_g = eval_clf.predict_proba(X_g_val)[:, 1]
    auc_g = roc_auc_score(g[val_idx], p_g)
    pred  = (p_g >= 0.5).astype(int)
    acc_g = accuracy_score(g[val_idx], pred)
    f1_g  = f1_score(g[val_idx], pred)
    print(f"  GENDER  AUC={auc_g:.4f}  acc={acc_g:.4f}  F1={f1_g:.4f}  "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    eval_reg_a = lgb.LGBMRegressor(**CHAMPION_REG)
    eval_reg_a.fit(X_a_tr, a[train_idx])
    p_a = eval_reg_a.predict(X_a_val)
    mae_a = mean_absolute_error(a[val_idx], p_a)
    r2_a  = r2_score(a[val_idx], p_a)
    print(f"  AGE     MAE={mae_a:.3f}  R2={r2_a:+.3f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    eval_reg_m = lgb.LGBMRegressor(**CHAMPION_REG)
    eval_reg_m.fit(X_m_tr, m[train_idx])
    p_m = eval_reg_m.predict(X_m_val)
    mae_m = mean_absolute_error(m[val_idx], p_m)
    r2_m  = r2_score(m[val_idx], p_m)
    print(f"  MOOD    MAE={mae_m:.3f}  R2={r2_m:+.3f}  ({time.time()-t0:.1f}s)")

    # ---------- Pass 2: refit on full data + emit ----------
    print("\nPass 2 - refitting on FULL 6,710 rows for deployment ...")
    all_indices = np.arange(N)
    girly_grid, masc_grid, age_grid, mood_grid = _build_grids_from(all_indices)
    X_gen_full, X_age_full, X_mood_full = _stack_features(
        all_indices, girly_grid, masc_grid, age_grid, mood_grid,
    )

    t0 = time.time()
    prod_clf = lgb.LGBMClassifier(**CHAMPION_CLF)
    prod_clf.fit(X_gen_full, g)
    print(f"  GENDER  fit on {len(g)} rows  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    prod_reg_a = lgb.LGBMRegressor(**CHAMPION_REG)
    prod_reg_a.fit(X_age_full, a)
    print(f"  AGE     fit on {len(a)} rows  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    prod_reg_m = lgb.LGBMRegressor(**CHAMPION_REG)
    prod_reg_m.fit(X_mood_full, m)
    print(f"  MOOD    fit on {len(m)} rows  ({time.time()-t0:.1f}s)")

    # ---- Emit JSON to models-js/ (production trees, full-data fit) ----
    print("\nEmitting JSON (models-js/) ...")
    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(N, 16, replace=False)

    targets_cfg = [
        ("gender", prod_clf.booster_,   "binary",     X_gen_full[sample_idx]),
        ("age",    prod_reg_a.booster_, "regression", X_age_full[sample_idx]),
        ("mood",   prod_reg_m.booster_, "regression", X_mood_full[sample_idx]),
    ]

    emit_stats = {}
    for name, booster, objective, X_sample in targets_cfg:
        json_out = os.path.join(JS_OUT_DIR, f"{name}_trees.json")
        t0 = time.time()
        n_trees, n_nodes = _emit_tree_json(booster, json_out, objective)
        delta = _verify_json(json_out, booster, X_sample)
        json_kb = os.path.getsize(json_out) / 1024
        if delta > 1e-5:
            raise SystemExit(f"{name} JSON emission diverged from LightGBM by {delta:.3e}")
        print(f"  {name:6s}  json={json_kb:7.1f} KB  trees={n_trees} nodes={n_nodes}  "
              f"|delta|<{delta:.1e}  ({time.time()-t0:.1f}s)")
        emit_stats[name] = {
            "json_kb": round(json_kb, 2),
            "n_trees": n_trees,
            "n_nodes": n_nodes,
            "max_emit_delta": float(delta),
        }

    # bucket_data.py — full-data grids, written into cloudflare/ so the
    # worker's features.py picks up the lookups the deployed trees expect.
    bucket_path = os.path.join(CF_DIR, "bucket_data.py")
    _emit_bucket_data(bucket_path, (girly_grid, masc_grid, age_grid, mood_grid))
    print(f"  bucket_data.py  {os.path.getsize(bucket_path)/1024:.1f} KB")

    # ---- Save metadata + summary ----
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_total_rows":  int(N),
        "validation": {
            "kind":     "stratified random split",
            "n_train":  int(len(train_idx)),
            "n_val":    int(len(val_idx)),
            "stratify": "gender",
            "seed":     SEED,
        },
        "deployment": {
            "fit_on_rows": int(N),
            "notes":       "Final boosters re-fit on the full dataset after metrics were captured on the eval split; bucket grids are built from all rows.",
        },
        "champion_clf_params": CHAMPION_CLF,
        "champion_reg_params": CHAMPION_REG,
        "n_features": {
            "gender": int(X_gen_full.shape[1]),
            "age":    int(X_age_full.shape[1]),
            "mood":   int(X_mood_full.shape[1]),
        },
        "target_stats": {
            "gender_girl_frac": float((g == 1).mean()),
            "age_min":  int(a.min()), "age_max":  int(a.max()), "age_mean":  float(a.mean()),
            "mood_min": int(m.min()), "mood_max": int(m.max()), "mood_mean": float(m.mean()),
        },
        "validation_scores": {
            "gender_auc":      float(auc_g),
            "gender_accuracy": float(acc_g),
            "gender_f1":       float(f1_g),
            "age_mae":         float(mae_a),
            "age_r2":          float(r2_a),
            "mood_mae":        float(mae_m),
            "mood_r2":         float(r2_m),
        },
        "emit_stats": emit_stats,
        "seed": SEED,
    }
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nValidation (random stratified 6,000 / 710 split, seed=42)")
    print(f"  GENDER AUC = {auc_g:.4f}   acc = {acc_g:.4f}   F1 = {f1_g:.4f}")
    print(f"  AGE    MAE = {mae_a:.3f}   R2 = {r2_a:+.3f}")
    print(f"  MOOD   MAE = {mae_m:.3f}   R2 = {r2_m:+.3f}")
    print(f"\nDeployed boosters fit on all {N} rows.")
    print(f"Artifacts:")
    print(f"  {JS_OUT_DIR}     (model trees)")
    print(f"  {CF_DIR}         (bucket_data.py)")
    print(f"  {HERE}           (summary.json)")
    print(f"Total wall time: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
