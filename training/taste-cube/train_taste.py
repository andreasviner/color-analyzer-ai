"""
Train the personal "taste cube" model (short survey).

Pipeline:
  1. Load + validate short sessions from raw/save.ligma.
  2. Turn each person into a handful of leakage-safe "probe" rows
     (taste_features.build_probe_rows): given the person's taste fingerprint and
     one candidate colour, the label is 1 if they picked that colour.
  3. Pass 1 - evaluate with a SESSION-level split (no person in both train and
     val): row AUC, holdout pick-accuracy (argmax desirability over the real
     offered quad vs the real pick), and a fingerprint-only baseline as a
     leakage gate (must sit at chance ~0.25).
  4. Pass 2 - refit on every row, emit the flat-tree JSON the browser walks
     (models-js/taste_trees.json), a JS parity fixture (taste_parity.json), and
     summary.json.

Run:  python train_taste.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import taste_features as tf  # noqa: E402

TRAINING_DIR = os.path.normpath(os.path.join(HERE, ".."))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RAW_SOURCE = os.path.join(TRAINING_DIR, "raw", "save.ligma")
# Trees ship from the static site next to the short/long trees + tree_walker.js.
JS_OUT_DIR = os.path.normpath(
    os.path.join(PROJECT_ROOT, "..", "english_html", "color-polygraph", "models-js"))
os.makedirs(JS_OUT_DIR, exist_ok=True)

SEED = 42
VAL_SIZE = 0.10
DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000
FP_LEN = tf.feature_layout()["fingerprint"]

# Tuned via tune_taste.py (probes=8, fixed session-level val): deeper + slower
# learning lifted holdout pick-accuracy from 0.49 to 0.50.
CHAMPION_CLF = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.015,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=100, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbosity=-1,
)


# ---------- load + validate (same shape checks as long-models/train_long.py) ----------

def _is_valid(row):
    try:
        if row[5] not in ("g", "j"):
            return False
        if row[8] == "no data" or len(row[8]) < 4:
            return False
        if len(row[8][0]) < 64 or len(row[8][1]) < 16 or len(row[8][2]) < 4:
            return False
        if len(row[7]) < 21 or len(row[6]) < 21:
            return False
        total = int(row[7][-1])
        if total < DURATION_MIN_MS or total > DURATION_MAX_MS:
            return False
        return True
    except Exception:
        return False


def _parse(row):
    return {
        "id": row[0],
        "offered": [list(c) for c in row[8][0][:64]],
        "r1": [list(c) for c in row[8][1][:16]],
        "r2": [list(c) for c in row[8][2][:4]],
        "final": list(row[8][3]),
        "valg": row[6][:21],
    }


# ---------- flat-tree JSON emit (copied from long-models/train_long.py) ----------

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


# ---------- pick-accuracy over held-out probe groups ----------

def _pick_accuracy(model, X, y, groups, fingerprint_only=False):
    """For each probe group of 4 candidates, did argmax model score land on the
    colour the person actually picked?"""
    Xs = X.copy()
    if fingerprint_only:
        Xs = Xs.copy()
        Xs[:, FP_LEN:] = 0.0  # blank candidate + interaction blocks
    raw = model.predict(Xs, raw_score=True)
    by_group = {}
    for i, gid in enumerate(groups):
        by_group.setdefault(gid, []).append(i)
    correct = 0
    total = 0
    for gid, idxs in by_group.items():
        if len(idxs) < 2:
            continue
        scores = [raw[i] for i in idxs]
        true_local = next((k for k, i in enumerate(idxs) if y[i] == 1), None)
        if true_local is None:
            continue
        pred_local = int(np.argmax(scores))
        correct += int(pred_local == true_local)
        total += 1
    return correct / total if total else 0.0


# ---------- main ----------

def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    print("Loading short sessions...")
    with open(RAW_SOURCE, encoding="utf-8") as fh:
        raw = json.load(fh)
    sessions = [_parse(r) for r in raw if _is_valid(r)]
    print(f"  {len(sessions)} valid sessions")

    print(f"Building probe rows ({tf.PROBES_PER_SESSION} probes/person)...")
    X_list, y_list, grp_list, sess_list = [], [], [], []
    for s in sessions:
        for row, label, gid in tf.build_probe_rows(s):
            X_list.append(row)
            y_list.append(label)
            grp_list.append(gid)
            sess_list.append(s["id"])
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int8)
    groups = grp_list                          # list of (session_id, question) tuples
    sess_ids = sess_list                        # list of session ids, row-aligned
    print(f"  X {X.shape}  positives {int(y.sum())} ({y.mean()*100:.1f}%)")

    # ---- Pass 1: session-level split ----
    uniq = sorted(set(sess_ids))
    tr_sess, va_sess = train_test_split(uniq, test_size=VAL_SIZE, random_state=SEED)
    tr_set, va_set = set(tr_sess), set(va_sess)
    tr = np.array([i for i in range(len(X)) if sess_ids[i] in tr_set])
    va = np.array([i for i in range(len(X)) if sess_ids[i] in va_set])
    print(f"\nPass 1 - session split  train_rows={len(tr)}  val_rows={len(va)}")

    clf = lgb.LGBMClassifier(**CHAMPION_CLF).fit(X[tr], y[tr])
    p_va = clf.predict_proba(X[va])[:, 1]
    auc = roc_auc_score(y[va], p_va)
    groups_va = [groups[i] for i in va]
    acc = _pick_accuracy(clf, X[va], y[va], groups_va)
    base = _pick_accuracy(clf, X[va], y[va], groups_va, fingerprint_only=True)
    print(f"  row AUC               = {auc:.4f}")
    print(f"  holdout pick-accuracy = {acc:.4f}   (chance = 0.25)")
    print(f"  fingerprint-only base = {base:.4f}   (leakage gate, want ~0.25)")

    # ---- Pass 2: refit on all rows + emit ----
    print("\nPass 2 - refit on all rows + emit ...")
    prod = lgb.LGBMClassifier(**CHAMPION_CLF).fit(X, y)
    out = os.path.join(JS_OUT_DIR, "taste_trees.json")
    n_trees, n_nodes = _emit_tree_json(prod.booster_, out, "binary")
    sample_idx = rng.choice(len(X), min(64, len(X)), replace=False)
    delta = _verify_json(out, prod.booster_, X[sample_idx])
    if delta > 1e-5:
        raise SystemExit(f"taste_trees.json diverged from LightGBM by {delta:.3e}")
    kb = os.path.getsize(out) / 1024
    print(f"  taste_trees.json  {kb:.1f} KB  trees={n_trees} nodes={n_nodes}  |delta|<{delta:.1e}")

    # ---- parity fixture for the JS mirror (serve mode: full real winners) ----
    parity = []
    test_colors = [[230, 40, 40], [40, 80, 220], [240, 230, 60],
                   [30, 170, 80], [200, 120, 200], [25, 25, 25]]
    for s in sessions[:8]:
        ctx = tf.session_context(s["r1"], s["r2"], s["final"])
        parity.append({
            "r1": s["r1"], "r2": s["r2"], "final": s["final"],
            "fingerprint": tf.fingerprint_vector(ctx),
            "candidates": [{"rgb": c, "row": tf.feature_row(ctx, c)} for c in test_colors],
        })
    with open(os.path.join(HERE, "taste_parity.json"), "w", encoding="utf-8") as fh:
        json.dump({"layout": tf.feature_layout(), "samples": parity}, fh)
    print(f"  taste_parity.json  {len(parity)} samples")

    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "kind": "personal taste cube model (short survey)",
        "data": {
            "source": "raw/save.ligma short sessions",
            "n_sessions": len(sessions),
            "probes_per_session": tf.PROBES_PER_SESSION,
            "n_rows": int(len(X)),
            "positive_rate": float(y.mean()),
        },
        "feature_layout": tf.feature_layout(),
        "validation": {
            "kind": "session-level split", "val_frac": VAL_SIZE, "seed": SEED,
            "row_auc": float(auc),
            "holdout_pick_accuracy": float(acc),
            "fingerprint_only_baseline": float(base),
            "chance": 0.25,
        },
        "champion_clf_params": CHAMPION_CLF,
        "emit": {"json_kb": round(kb, 2), "n_trees": n_trees, "n_nodes": n_nodes,
                 "max_emit_delta": float(delta)},
        "seed": SEED,
    }
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nArtifacts: {out}")
    print(f"           {os.path.join(HERE, 'taste_parity.json')}")
    print(f"           {os.path.join(HERE, 'summary.json')}")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
