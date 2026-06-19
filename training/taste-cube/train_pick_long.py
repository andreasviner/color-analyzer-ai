"""
Train the LONG-survey colour-pick model.

Long surveys are rare in the wild, so training rows are SYNTHETIC long
sessions assembled from quads of real short sessions -- the exact same
construction (and code) the long gender/age/mood models use
(training/long-models/train_long.py). Each synthetic long row then yields
probe rows via the same overwrite scheme as the short pick model.

Mirrors train_pick.py: session-level split, leak gate, holdout pick-accuracy,
pass-2 refit + emit.

Run:  python train_pick_long.py [--smoke]

Emits:
    models-js/pick_long_trees.json
    pick_long_parity.json (JS mirror check on a long-shaped context)
    pick_long_summary.json
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
LONG_MODELS_DIR = os.path.normpath(os.path.join(HERE, "..", "long-models"))
sys.path.insert(0, HERE)
sys.path.insert(0, LONG_MODELS_DIR)

import pick_features as pf            # noqa: E402  (shared candidate block)
import pick_features_long as pfl      # noqa: E402
import taste_features as tfeat        # noqa: E402
import train_long as tl               # noqa: E402  (synthetic long assembly)
from train_taste import _emit_tree_json, _verify_json  # noqa: E402

TRAINING_DIR = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, TRAINING_DIR)
import data_cleaning as dc             # noqa: E402  (frozen hold-out helper)
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RAW_SOURCE = os.path.join(TRAINING_DIR, "raw", "save.ligma")
JS_OUT_DIR = os.path.normpath(
    os.path.join(PROJECT_ROOT, "..", "english_html", "color-polygraph", "models-js"))
os.makedirs(JS_OUT_DIR, exist_ok=True)

SEED = 42
VAL_SIZE = 0.10

# Same tuned params as the short pick model.
PARAMS = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.015,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=100, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbosity=-1,
)

LAYOUT = pfl.layout(with_interactions=True)
PERSON = LAYOUT["person"]
TOTAL = LAYOUT["total"]


def pick_accuracy(model, X, y, groups, blank_from=None):
    Xs = X
    if blank_from is not None:
        Xs = X.copy()
        Xs[:, blank_from:] = 0.0
    raw = model.predict(Xs, raw_score=True)
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    ok = tot = 0
    for g, idxs in by.items():
        true_local = next((k for k, i in enumerate(idxs) if y[i] == 1), None)
        if true_local is None or len(idxs) < 2:
            continue
        ok += int(int(np.argmax([raw[i] for i in idxs])) == true_local)
        tot += 1
    return ok / tot if tot else 0.0


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    print("Loading short sessions (served as duplicate-short longs)...")
    with open(RAW_SOURCE, encoding="utf-8") as fh:
        raw = json.load(fh)
    raw = dc.dedupe_short_rows(raw)  # one row per unique survey (same as the short models)
    valid_raw = [r for r in raw if tl._is_valid(r)]
    shorts = [tl._parse_short(r) for r in valid_raw]
    if smoke:
        shorts = shorts[:120]
        valid_raw = valid_raw[:120]
    # Duplicate-short: each short is served as one coherent person's long (the
    # short replicated 4x), same as the long gender/age/mood model. The probe
    # builder removes the probed question from the short BEFORE duplicating, so
    # the answer cannot leak through the duplicate blocks.
    for i, s in enumerate(shorts):
        s["id"] = i
    # Frozen content-hashed hold-out on the SOURCE short (same hold-out the
    # short pick model uses), so the long pick metric is stable across versions
    # and scored on the same people as the rest of the short leaderboard.
    sess_holdout = [dc.short_is_holdout(r) for r in valid_raw]
    sessions = shorts
    print(f"  {len(shorts)} short sessions -> duplicate-short longs (built per probe)")

    print(f"Building probe rows ({pfl.PROBES_PER_SESSION_LONG} probes/row, "
          f"long prod features per probe)...   layout {LAYOUT}")
    t1 = time.time()
    cap = len(sessions) * pfl.PROBES_PER_SESSION_LONG * 4
    X = np.zeros((cap, TOTAL), dtype=np.float32)
    y = np.zeros(cap, dtype=np.int8)
    sid, gid, hid = [], [], []
    n = 0
    for k, s in enumerate(sessions):
        for row, label, g in pfl.build_probe_rows(s):
            X[n] = row
            y[n] = label
            sid.append(s["id"])
            gid.append(g)
            hid.append(sess_holdout[k])
            n += 1
        if (k + 1) % 200 == 0:
            rate = (k + 1) / (time.time() - t1)
            print(f"  {k+1}/{len(sessions)} rows  ({rate:.0f}/s, "
                  f"eta {(len(sessions)-k-1)/rate:.0f}s)")
    X, y = X[:n], y[:n]
    print(f"  X {X.shape}  positives {int(y.sum())} ({y.mean()*100:.1f}%)  "
          f"build {time.time()-t1:.0f}s")

    # ---- Pass 1: frozen content-hashed session-level split ----
    tr = np.array([i for i in range(n) if not hid[i]])
    va = np.array([i for i in range(n) if hid[i]])
    gva = [gid[i] for i in va]
    print(f"\nPass 1 - frozen hold-out split  train_rows={len(tr)}  val_rows={len(va)}")

    clf = lgb.LGBMClassifier(**PARAMS).fit(X[tr], y[tr])
    auc = roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1])
    acc = pick_accuracy(clf, X[va], y[va], gva)
    gate = pick_accuracy(clf, X[va], y[va], gva, blank_from=PERSON)
    print(f"  AUC={auc:.4f}  pick-accuracy={acc:.4f}  leak-gate={gate:.4f} (want ~0.25)")

    # ---- Pass 2: refit on all rows + emit ----
    emit_stats = {}
    if not smoke:
        print("\nPass 2 - refit on all rows + emit ...")
        prod_clf = lgb.LGBMClassifier(**PARAMS).fit(X, y)
        out = os.path.join(JS_OUT_DIR, "pick_long_trees.json")
        n_trees, n_nodes = _emit_tree_json(prod_clf.booster_, out, "binary")
        sample = rng.choice(n, min(64, n), replace=False)
        delta = _verify_json(out, prod_clf.booster_, X[sample])
        if delta > 1e-5:
            raise SystemExit(f"pick_long_trees.json diverged from LightGBM by {delta:.3e}")
        kb = os.path.getsize(out) / 1024
        emit_stats = {"json_kb": round(kb, 2), "n_trees": n_trees, "n_nodes": n_nodes,
                      "max_emit_delta": float(delta)}
        print(f"  pick_long_trees.json  {kb:.1f} KB  trees={n_trees} nodes={n_nodes}  |delta|<{delta:.1e}")

        # Parity fixture with a LONG-shaped context (r1 64-wide, r2 16-wide):
        # same JS functions as the short fixture, different input lengths.
        test_colors = [[230, 40, 40], [40, 80, 220], [240, 230, 60],
                       [30, 170, 80], [200, 120, 200], [25, 25, 25]]
        parity = []
        for s in sessions[:4]:
            long_pay, _ = tl._dup_short_to_long(s)   # the served long shape
            ctx = tfeat.session_context(long_pay["r1"], long_pay["r2"], long_pay["final"])
            parity.append({
                "r1": long_pay["r1"], "r2": long_pay["r2"], "final": long_pay["final"],
                "candidates": [{
                    "rgb": c,
                    "cand": pf.candidate_vector(c),
                    "inter": tfeat.interaction_vector(c, ctx),
                } for c in test_colors],
            })
        with open(os.path.join(HERE, "pick_long_parity.json"), "w", encoding="utf-8") as fh:
            json.dump({"layout": LAYOUT, "samples": parity}, fh)
        print(f"  pick_long_parity.json  {len(parity)} samples")

    with open(os.path.join(HERE, "pick_long_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "kind": "LONG-survey colour-pick model (synthetic long rows from short quads)",
            "smoke": smoke,
            "n_long_rows": len(sessions),
            "probes_per_session": pfl.PROBES_PER_SESSION_LONG,
            "n_rows": int(n),
            "layout": LAYOUT,
            "params": PARAMS,
            "validation": {"kind": "frozen content-hashed session-level hold-out",
                           "val_frac": VAL_SIZE, "chance": 0.25,
                           "auc": float(auc), "pick_accuracy": float(acc),
                           "leak_gate": float(gate)},
            "emit": emit_stats,
        }, fh, indent=2)

    print(f"\nSummary -> pick_long_summary.json   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
