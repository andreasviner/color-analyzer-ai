"""
Train the LONG-survey colour-pick model on REAL long surveys.

Previously this trained on SYNTHETIC "duplicate-short longs" (each short
replicated 4x into a fake long) because real longs were rare. We now have 1326
real longs, and a leak-free era-honest experiment showed real beats synthetic on
the worldwide cohort (pick-acc 0.534 vs 0.507, AUC 0.759 vs 0.742) and that
adding the synthetic data back in slightly HURTS (legacy population mismatch) --
so we train on real longs only.

Construction: the long profile must be a full long (all 256 colours), so the
leak-free target uses the overwrite scheme ON the real long -- pick a round-0
question whose winner did not advance (absent from r2, safe to remove), clone a
donor loser over its offered quad / r1 winner / valg digit, compute the long
prod features on the modified long, and the original quad becomes the "4 new
colours" with the real pick as the label. ALL eligible round-0 losers per long
are used (~48/long); unlike the short model the person vector is recomputed per
probe (the overwrite changes the profile), so density costs compute.

Hold-out is per real long (long_is_holdout) so a person's profile and targets
never straddle train/val, and the held-out longs ARE the worldwide cohort -- the
reported pick accuracy is the worldwide long-pick gate.

Run:  python train_pick_long.py [--smoke]
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pick_features as pf            # noqa: E402  (shared candidate descriptors)
import pick_features_long as pfl      # noqa: E402  (long prod person vector)
import taste_features as tfeat        # noqa: E402  (interactions + colour math)
from train_taste import _emit_tree_json, _verify_json  # noqa: E402

TRAINING_DIR = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, TRAINING_DIR)
import data_cleaning as dc            # noqa: E402
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
LONG_REAL = os.path.join(TRAINING_DIR, "raw", "long_real.json")
JS_OUT_DIR = os.path.normpath(
    os.path.join(PROJECT_ROOT, "..", "english_html", "color-polygraph", "models-js"))
os.makedirs(JS_OUT_DIR, exist_ok=True)

SEED = 42
N_R0_LONG = 64

PARAMS = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.015,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=100, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbosity=-1,
)

LAYOUT = pfl.layout(with_interactions=True)
PERSON = LAYOUT["person"]
TOTAL = LAYOUT["total"]


def long_eligible(p):
    """Round-0 questions (of 64) whose winner did not advance to r2."""
    r1, r2 = p["r1"], [list(c) for c in p["r2"]]
    out = []
    for q in range(N_R0_LONG):
        if tfeat._color_in(r1[q], r2):
            continue
        try:
            pk = int(p["valg"][q])
        except (ValueError, IndexError):
            continue
        if 0 <= pk <= 3:
            out.append(q)
    return out


def long_modified(p, q, donor):
    """The real long as if round-0 question q never happened (its offered quad,
    r1 winner and valg digit overwritten with a donor loser's)."""
    offered = [list(c) for c in p["offered"]]
    r1 = [list(c) for c in p["r1"]]
    valg = list(p["valg"])
    offered[q * 4:(q + 1) * 4] = [list(c) for c in p["offered"][donor * 4:(donor + 1) * 4]]
    r1[q] = list(p["r1"][donor])
    valg[q] = p["valg"][donor]
    return {"offered": offered, "r1": r1, "r2": [list(c) for c in p["r2"]],
            "r3": [list(c) for c in p["r3"]], "final": list(p["final"]),
            "valg": "".join(valg), "tider": p["tider"]}


def build_real_long_groups(p, submit_unix):
    """All eligible-loser probe groups for one real long: list of (rows4, pick)."""
    probes = long_eligible(p)
    if len(probes) < 2:
        return []
    groups = []
    for i, q in enumerate(probes):
        donor = probes[(i + 1) % len(probes)]
        if donor == q:
            continue
        mod = long_modified(p, q, donor)
        person = pfl.person_vector(mod, submit_unix)
        ctx = tfeat.session_context(mod["r1"], mod["r2"], mod["final"])
        pick = int(p["valg"][q])
        quad = p["offered"][q * 4:(q + 1) * 4]
        if len(quad) < 4:
            continue
        rows4 = np.empty((4, TOTAL), dtype=np.float32)
        for idx in range(4):
            rows4[idx] = person + pf.candidate_vector(quad[idx]) + \
                tfeat.interaction_vector(quad[idx], ctx)
        groups.append((rows4, pick))
    return groups


def _assemble(group_recs):
    ng = len(group_recs)
    X = np.empty((ng * 4, TOTAL), dtype=np.float32)
    y = np.zeros(ng * 4, dtype=np.int8)
    gid = np.empty(ng * 4, dtype=np.int32)
    for gi, (rows4, pick) in enumerate(group_recs):
        X[gi * 4:gi * 4 + 4] = rows4
        y[gi * 4 + pick] = 1
        gid[gi * 4:gi * 4 + 4] = gi
    return X, y, gid


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

    print("Loading real long surveys...")
    longs = [it for it in json.load(open(LONG_REAL, encoding="utf-8"))
             if it.get("payload") and it.get("label")]
    if smoke:
        longs = longs[:80]
    print(f"  {len(longs)} real longs   layout {LAYOUT}")

    print("Building overwrite probe groups (all eligible round-0 losers per long)...")
    t1 = time.time()
    train_recs, val_recs, n_tr, n_va = [], [], 0, 0
    for n, it in enumerate(longs):
        g = build_real_long_groups(it["payload"], int(it["label"].get("time", 0) or 0))
        if not g:
            continue
        if dc.long_is_holdout(it["payload"]):
            val_recs.extend(g)
            n_va += 1
        else:
            train_recs.extend(g)
            n_tr += 1
        if (n + 1) % 200 == 0:
            rate = (n + 1) / (time.time() - t1)
            print(f"  {n+1}/{len(longs)} longs  ({rate:.0f}/s, eta {(len(longs)-n-1)/rate:.0f}s)")
    Xtr, ytr, _ = _assemble(train_recs)
    Xva, yva, gva = _assemble(val_recs)
    print(f"  train: {n_tr} longs, {len(train_recs)} groups, {len(ytr)} rows")
    print(f"  val:   {n_va} longs, {len(val_recs)} groups, {len(yva)} rows  "
          f"(build {time.time()-t1:.0f}s)")

    # ---- Pass 1: worldwide long-pick gate (held-out real longs) ----
    print("\nPass 1 - worldwide gate (held-out real longs)")
    clf = lgb.LGBMClassifier(**PARAMS).fit(Xtr, ytr)
    auc = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
    acc = pick_accuracy(clf, Xva, yva, gva)
    gate = pick_accuracy(clf, Xva, yva, gva, blank_from=PERSON)
    print(f"  AUC={auc:.4f}  pick-accuracy={acc:.4f}  leak-gate={gate:.4f} (want ~0.25)")
    n_rows_total = int(len(ytr) + len(yva))
    n_tr_groups, n_va_groups = len(train_recs), len(val_recs)

    # ---- Pass 2: refit on ALL groups + emit ----
    emit_stats = {}
    del Xtr, ytr, Xva, yva, gva, clf
    if not smoke:
        print("\nPass 2 - refit on all groups + emit ...")
        Xall, yall, _ = _assemble(train_recs + val_recs)
        train_recs, val_recs = [], []
        prod_clf = lgb.LGBMClassifier(**PARAMS).fit(Xall, yall)
        out = os.path.join(JS_OUT_DIR, "pick_long_trees.json")
        n_trees, n_nodes = _emit_tree_json(prod_clf.booster_, out, "binary")
        rng = np.random.RandomState(SEED)
        sample = rng.choice(len(yall), min(64, len(yall)), replace=False)
        delta = _verify_json(out, prod_clf.booster_, Xall[sample])
        if delta > 1e-5:
            raise SystemExit(f"pick_long_trees.json diverged from LightGBM by {delta:.3e}")
        kb = os.path.getsize(out) / 1024
        emit_stats = {"json_kb": round(kb, 2), "n_trees": n_trees, "n_nodes": n_nodes,
                      "max_emit_delta": float(delta)}
        print(f"  pick_long_trees.json  {kb:.1f} KB  trees={n_trees} nodes={n_nodes}  |delta|<{delta:.1e}")

        # Parity fixture (long-shaped context: r1 64-wide, r2 16-wide) from a real long.
        test_colors = [[230, 40, 40], [40, 80, 220], [240, 230, 60],
                       [30, 170, 80], [200, 120, 200], [25, 25, 25]]
        parity = []
        for it in longs[:4]:
            p = it["payload"]
            ctx = tfeat.session_context(p["r1"], p["r2"], p["final"])
            parity.append({
                "r1": p["r1"], "r2": p["r2"], "final": p["final"],
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
            "kind": "LONG-survey colour-pick model (REAL longs, overwrite probes, no synthetic)",
            "smoke": smoke,
            "n_train_longs": n_tr,
            "n_val_longs": n_va,
            "n_train_groups": n_tr_groups,
            "n_val_groups": n_va_groups,
            "n_rows": n_rows_total,
            "layout": LAYOUT,
            "params": PARAMS,
            "validation": {"kind": "per-long hold-out (worldwide cohort); long_is_holdout, "
                                   "chance=0.25",
                           "auc": float(auc), "pick_accuracy": float(acc),
                           "leak_gate": float(gate)},
            "emit": emit_stats,
        }, fh, indent=2)

    print(f"\nSummary -> pick_long_summary.json   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
