"""
Train the colour-pick model: person (prod features, probe overwritten) +
4 new colours -> which one would they pick.

The model is a binary LightGBM scoring one candidate at a time; the prediction
for "4 new colours" is the argmax of the 4 scores. Two variants are evaluated:

    A: person(479 prod) + candidate descriptors
    B: person(479 prod) + candidate descriptors + interaction block

Both are scored on the same fixed session-level validation split with
holdout pick-accuracy (chance = 0.25) plus a leakage gate (candidate columns
blanked -> the model must fall to chance, proving the overwritten prod
features contain no trace of the probe's answer).

Run:  python train_pick.py [--smoke]
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
import pick_features as pf  # noqa: E402
import taste_features as tfeat  # noqa: E402
# Flat-tree JSON emit + bit-exact verification, shared with train_taste.py.
from train_taste import _emit_tree_json, _verify_json  # noqa: E402

TRAINING_DIR = os.path.normpath(os.path.join(HERE, ".."))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RAW_SOURCE = os.path.join(TRAINING_DIR, "raw", "save.ligma")

sys.path.insert(0, TRAINING_DIR)
from data_cleaning import is_valid_clean  # noqa: E402
JS_OUT_DIR = os.path.normpath(
    os.path.join(PROJECT_ROOT, "..", "english_html", "color-polygraph", "models-js"))
os.makedirs(JS_OUT_DIR, exist_ok=True)

SEED = 42
VAL_SIZE = 0.10
DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000

# Tuned on the fingerprint model (tune_taste.py); good starting point here.
PARAMS = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.015,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=100, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbosity=-1,
)

LAYOUT = pf.layout(with_interactions=True)
PERSON = LAYOUT["person"]
CAND = LAYOUT["candidate"]
INTER = LAYOUT["interaction"]
TOTAL = LAYOUT["total"]


def _is_valid(row):
    # Shared validity + troll filter, plus the pick model's stricter need for a
    # full 21-char valg (round-1 16 + round-2 4 + final).
    if not is_valid_clean(row):
        return False
    try:
        return len(row[6]) >= 21
    except Exception:
        return False


def _parse(row):
    return {
        "id": row[0],
        "time": int(row[1]) if str(row[1]).lstrip("-").isdigit() else 0,
        "offered": [list(c) for c in row[8][0][:64]],
        "r1": [list(c) for c in row[8][1][:16]],
        "r2": [list(c) for c in row[8][2][:4]],
        "final": list(row[8][3]),
        "valg": row[6][:21],
        "tider": [int(x) for x in row[7][:21]],
    }


def pick_accuracy(model, X, y, groups, cols=None, blank_from=None):
    """Per probe group of 4, does argmax score match the actual pick?
    cols: optional column slice (use only the first `cols` columns).
    blank_from: zero out columns >= blank_from (leakage gate)."""
    Xs = X[:, :cols] if cols else X
    if blank_from is not None:
        Xs = Xs.copy()
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

    print("Loading sessions...")
    with open(RAW_SOURCE, encoding="utf-8") as fh:
        raw = json.load(fh)
    sessions = [_parse(r) for r in raw if _is_valid(r)]
    if smoke:
        sessions = sessions[:300]
    print(f"  {len(sessions)} sessions   layout {LAYOUT}")

    print(f"Building rows ({pf.PROBES_PER_SESSION} probes/person, prod features per probe)...")
    t1 = time.time()
    # Preallocate: rows = sessions * probes * 4 upper bound.
    cap = len(sessions) * pf.PROBES_PER_SESSION * 4
    X = np.zeros((cap, TOTAL), dtype=np.float32)
    y = np.zeros(cap, dtype=np.int8)
    sid, gid = [], []
    n = 0
    for k, s in enumerate(sessions):
        for row, label, g in pf.build_probe_rows(s):
            X[n] = row
            y[n] = label
            sid.append(s["id"])
            gid.append(g)
            n += 1
        if (k + 1) % 500 == 0:
            rate = (k + 1) / (time.time() - t1)
            print(f"  {k+1}/{len(sessions)} sessions  ({rate:.0f}/s, "
                  f"eta {(len(sessions)-k-1)/rate:.0f}s)")
    X, y = X[:n], y[:n]
    print(f"  X {X.shape}  positives {int(y.sum())} ({y.mean()*100:.1f}%)  "
          f"build {time.time()-t1:.0f}s")

    # Fixed session-level split.
    uniq = sorted(set(sid))
    tr_s, va_s = train_test_split(uniq, test_size=VAL_SIZE, random_state=SEED)
    tr_set, va_set = set(tr_s), set(va_s)
    tr = np.array([i for i in range(n) if sid[i] in tr_set])
    va = np.array([i for i in range(n) if sid[i] in va_set])
    gva = [gid[i] for i in va]
    print(f"  train_rows={len(tr)}  val_rows={len(va)}")

    results = {}

    # ---- Variant A: person + candidate descriptors ----
    print("\nVariant A: prod person + candidate descriptors")
    colsA = PERSON + CAND
    clfA = lgb.LGBMClassifier(**PARAMS).fit(X[tr][:, :colsA], y[tr])
    aucA = roc_auc_score(y[va], clfA.predict_proba(X[va][:, :colsA])[:, 1])
    accA = pick_accuracy(clfA, X[va], y[va], gva, cols=colsA)
    gateA = pick_accuracy(clfA, X[va], y[va], gva, cols=colsA, blank_from=PERSON)
    print(f"  AUC={aucA:.4f}  pick-accuracy={accA:.4f}  leak-gate={gateA:.4f} (want ~0.25)")
    results["A_person_cand"] = dict(auc=float(aucA), pick_acc=float(accA), leak_gate=float(gateA))

    # ---- Variant B: + interaction block ----
    print("\nVariant B: + interaction block (candidate vs the person's winners)")
    clfB = lgb.LGBMClassifier(**PARAMS).fit(X[tr], y[tr])
    aucB = roc_auc_score(y[va], clfB.predict_proba(X[va])[:, 1])
    accB = pick_accuracy(clfB, X[va], y[va], gva)
    gateB = pick_accuracy(clfB, X[va], y[va], gva, blank_from=PERSON)
    print(f"  AUC={aucB:.4f}  pick-accuracy={accB:.4f}  leak-gate={gateB:.4f} (want ~0.25)")
    results["B_person_cand_inter"] = dict(auc=float(aucB), pick_acc=float(accB), leak_gate=float(gateB))

    # ---- Pass 2: refit variant B (the winner) on ALL rows + emit ----
    emit_stats = {}
    if not smoke:
        print("\nPass 2 - refit variant B on all rows + emit ...")
        prod_clf = lgb.LGBMClassifier(**PARAMS).fit(X, y)
        out = os.path.join(JS_OUT_DIR, "pick_trees.json")
        n_trees, n_nodes = _emit_tree_json(prod_clf.booster_, out, "binary")
        rng = np.random.RandomState(SEED)
        sample = rng.choice(n, min(64, n), replace=False)
        delta = _verify_json(out, prod_clf.booster_, X[sample])
        if delta > 1e-5:
            raise SystemExit(f"pick_trees.json diverged from LightGBM by {delta:.3e}")
        kb = os.path.getsize(out) / 1024
        emit_stats = {"json_kb": round(kb, 2), "n_trees": n_trees, "n_nodes": n_nodes,
                      "max_emit_delta": float(delta)}
        print(f"  pick_trees.json  {kb:.1f} KB  trees={n_trees} nodes={n_nodes}  |delta|<{delta:.1e}")

        # Parity fixture for the JS mirror: only the candidate + interaction
        # blocks are computed in the browser (the person vector arrives from
        # the worker), so that is what the fixture covers.
        test_colors = [[230, 40, 40], [40, 80, 220], [240, 230, 60],
                       [30, 170, 80], [200, 120, 200], [25, 25, 25]]
        parity = []
        for s in sessions[:8]:
            ctx = tfeat.session_context(s["r1"], s["r2"], s["final"])
            parity.append({
                "r1": s["r1"], "r2": s["r2"], "final": s["final"],
                "candidates": [{
                    "rgb": c,
                    "cand": pf.candidate_vector(c),
                    "inter": pf.interaction_vector(c, ctx),
                } for c in test_colors],
            })
        with open(os.path.join(HERE, "pick_parity.json"), "w", encoding="utf-8") as fh:
            json.dump({"layout": LAYOUT, "samples": parity}, fh)
        print(f"  pick_parity.json  {len(parity)} samples")

    with open(os.path.join(HERE, "pick_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "kind": "colour-pick model (prod person features + client-side candidate features)",
            "smoke": smoke,
            "n_sessions": len(sessions),
            "probes_per_session": pf.PROBES_PER_SESSION,
            "n_rows": int(n),
            "layout": LAYOUT,
            "params": PARAMS,
            "validation": {"kind": "session-level split", "val_frac": VAL_SIZE,
                           "seed": SEED, "chance": 0.25},
            "results": results,
            "emit": emit_stats,
        }, fh, indent=2)

    print(f"\nSummary -> pick_summary.json   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
