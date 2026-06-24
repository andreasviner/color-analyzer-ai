"""
Train the colour-pick model: person (prod features) + 4 new colours -> which
one would they pick. Binary LightGBM scoring one candidate at a time; the
prediction for "4 new colours" is the argmax of the 4 scores.

TRAINING DATA -- real long surveys, no synthetic overwrite (the previous scheme
cloned a "loser" question over another within ONE short survey to fabricate a
leak-free probe). A real long survey is four short sub-surveys from the SAME
person, so we get authentic probes for free:

    PROFILE = one sub-survey, prod-features computed UNTOUCHED (= a real short
              survey, exactly what the deployed model sees at inference)
    TARGET  = a round-0 question from a DIFFERENT sub-survey of the same long:
              its 4 offered colours are the candidates, the person's actual
              pick is the label

Because the target question lives in a different sub-survey, the profile's
features structurally cannot contain it -> leak-free with no overwrite, the
profile is undistorted, and every round-0 question of the other sub-surveys is
usable (not just losers). We use ALL available probe groups per long
(profile k x other sub-survey j x 16 round-0 questions ~= 192/long); a
data-volume sweep showed accuracy plateaus by ~32 groups/long but more is free
and never hurts (leak-gate stays at chance throughout).

The hold-out is per LONG (long_is_holdout) so a person's profile and targets
never straddle train/val. The held-out longs ARE the worldwide cohort, so the
reported pick accuracy is the worldwide gate -- the number to steer by, not the
legacy-2020-Oslo-dominated short hold-out.

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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pick_features as pf  # noqa: E402
import taste_features as tfeat  # noqa: E402
from train_taste import _emit_tree_json, _verify_json  # noqa: E402

TRAINING_DIR = os.path.normpath(os.path.join(HERE, ".."))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
LONG_REAL = os.path.join(TRAINING_DIR, "raw", "long_real.json")

sys.path.insert(0, TRAINING_DIR)
import data_cleaning as dc  # noqa: E402
JS_OUT_DIR = os.path.normpath(
    os.path.join(PROJECT_ROOT, "..", "english_html", "color-polygraph", "models-js"))
os.makedirs(JS_OUT_DIR, exist_ok=True)

SEED = 42
N_R0 = 16  # round-0 questions per sub-survey

PARAMS = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.015,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=100, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbosity=-1,
)

LAYOUT = pf.layout(with_interactions=True)
PERSON = LAYOUT["person"]
TOTAL = LAYOUT["total"]


def sub_payloads(it):
    """The long's valid sub-surveys (save.ligma layout -> dicts)."""
    rows = dc.long_payload_to_shorts(it["payload"], it["label"], it.get("id", "L"))
    subs = []
    for r in rows:
        if not dc.is_valid_clean(r) or len(r[6]) < 21:
            continue
        subs.append({
            "offered": [list(c) for c in r[8][0][:64]],
            "r1": [list(c) for c in r[8][1][:16]],
            "r2": [list(c) for c in r[8][2][:4]],
            "final": list(r[8][3]),
            "valg": r[6][:21],
            "tider": [int(x) for x in r[7][:21]],
            "time": int(r[1]) if str(r[1]).lstrip("-").isdigit() else 0,
        })
    return subs


def _person_ctx(sub):
    pay = {"offered": sub["offered"], "r1": sub["r1"], "r2": sub["r2"],
           "final": sub["final"], "valg": sub["valg"], "tider": sub["tider"]}
    pv = np.asarray(pf.person_vector(pay, sub["time"]), dtype=np.float32)
    ctx = tfeat.session_context(sub["r1"], sub["r2"], sub["final"])
    return pv, ctx


def build_long_groups(subs):
    """All probe groups for one long: list of (rows4[4,TOTAL], pick).
    profile sub-survey k (pv+ctx) x other sub-survey j x round-0 question q."""
    cache = [_person_ctx(s) for s in subs]
    groups = []
    for k in range(len(subs)):
        pv, ctx = cache[k]
        pv_list = pv.tolist()
        for j in range(len(subs)):
            if j == k:
                continue
            tj = subs[j]
            for q in range(N_R0):
                try:
                    pick = int(tj["valg"][q])
                except (ValueError, IndexError):
                    continue
                if not (0 <= pick <= 3):
                    continue
                quad = tj["offered"][q * 4:(q + 1) * 4]
                if len(quad) < 4:
                    continue
                rows4 = np.empty((4, TOTAL), dtype=np.float32)
                for idx in range(4):
                    rows4[idx] = pv_list + pf.candidate_vector(quad[idx]) + \
                        pf.interaction_vector(quad[idx], ctx)
                groups.append((rows4, pick))
    return groups


def pick_accuracy(model, X, y, groups, blank_from=None):
    """Per probe group of 4, does argmax score match the actual pick?
    blank_from: zero out columns >= blank_from (leakage gate)."""
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


def _assemble(group_recs):
    """group_recs: list of (rows4, pick). Returns X, y, group-id array."""
    ng = len(group_recs)
    X = np.empty((ng * 4, TOTAL), dtype=np.float32)
    y = np.zeros(ng * 4, dtype=np.int8)
    gid = np.empty(ng * 4, dtype=np.int32)
    for gi, (rows4, pick) in enumerate(group_recs):
        X[gi * 4:gi * 4 + 4] = rows4
        y[gi * 4 + pick] = 1
        gid[gi * 4:gi * 4 + 4] = gi
    return X, y, gid


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()

    print("Loading real long surveys...")
    longs = [it for it in json.load(open(LONG_REAL, encoding="utf-8"))
             if it.get("payload") and it.get("label")]
    if smoke:
        longs = longs[:60]
    print(f"  {len(longs)} longs   layout {LAYOUT}")

    print("Building probe groups (profile sub-survey + cross-sub-survey round-0 targets)...")
    t1 = time.time()
    train_recs, val_recs, n_tr_long, n_va_long = [], [], 0, 0
    for n, it in enumerate(longs):
        subs = sub_payloads(it)
        if len(subs) < 2:
            continue
        g = build_long_groups(subs)
        if not g:
            continue
        if dc.long_is_holdout(it["payload"]):
            val_recs.extend(g)
            n_va_long += 1
        else:
            train_recs.extend(g)
            n_tr_long += 1
        if (n + 1) % 200 == 0:
            rate = (n + 1) / (time.time() - t1)
            print(f"  {n+1}/{len(longs)} longs  ({rate:.0f}/s, "
                  f"eta {(len(longs)-n-1)/rate:.0f}s)")
    Xtr, ytr, _ = _assemble(train_recs)
    Xva, yva, gva = _assemble(val_recs)
    print(f"  train: {n_tr_long} longs, {len(train_recs)} groups, {len(ytr)} rows")
    print(f"  val:   {n_va_long} longs, {len(val_recs)} groups, {len(yva)} rows  "
          f"(build {time.time()-t1:.0f}s)")

    # ---- Pass 1: worldwide gate (held-out longs) ----
    print("\nPass 1 - worldwide gate (held-out longs)")
    clf = lgb.LGBMClassifier(**PARAMS).fit(Xtr, ytr)
    auc = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
    acc = pick_accuracy(clf, Xva, yva, gva)
    gate = pick_accuracy(clf, Xva, yva, gva, blank_from=PERSON)
    print(f"  AUC={auc:.4f}  pick-accuracy={acc:.4f}  leak-gate={gate:.4f} (want ~0.25)")

    # ---- Pass 2: refit on ALL groups + emit ----
    emit_stats = {}
    n_tr_groups, n_va_groups = len(train_recs), len(val_recs)
    n_rows_total = int(len(ytr) + len(yva))
    del Xtr, ytr, Xva, yva, gva, clf   # free Pass-1 matrices before the full-data one
    if not smoke:
        print("\nPass 2 - refit on all groups + emit ...")
        Xall, yall, _ = _assemble(train_recs + val_recs)
        train_recs, val_recs = [], []   # release the per-group record lists
        prod_clf = lgb.LGBMClassifier(**PARAMS).fit(Xall, yall)
        out = os.path.join(JS_OUT_DIR, "pick_trees.json")
        n_trees, n_nodes = _emit_tree_json(prod_clf.booster_, out, "binary")
        rng = np.random.RandomState(SEED)
        sample = rng.choice(len(yall), min(64, len(yall)), replace=False)
        delta = _verify_json(out, prod_clf.booster_, Xall[sample])
        if delta > 1e-5:
            raise SystemExit(f"pick_trees.json diverged from LightGBM by {delta:.3e}")
        kb = os.path.getsize(out) / 1024
        emit_stats = {"json_kb": round(kb, 2), "n_trees": n_trees, "n_nodes": n_nodes,
                      "max_emit_delta": float(delta)}
        print(f"  pick_trees.json  {kb:.1f} KB  trees={n_trees} nodes={n_nodes}  |delta|<{delta:.1e}")

        # Parity fixture for the JS mirror: candidate + interaction blocks only
        # (the person vector arrives from the worker). Profile sub-surveys are
        # short-shaped, same as deployment.
        test_colors = [[230, 40, 40], [40, 80, 220], [240, 230, 60],
                       [30, 170, 80], [200, 120, 200], [25, 25, 25]]
        parity = []
        for it in longs[:8]:
            subs = sub_payloads(it)
            if not subs:
                continue
            s = subs[0]
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

    # Summary (keeps the B_person_cand_inter key the orchestrator reads; the
    # numbers are now the worldwide per-long gate, not the legacy short fold).
    with open(os.path.join(HERE, "pick_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "kind": "colour-pick model (real-long-derived: untouched profile sub-survey "
                    "+ cross-sub-survey round-0 targets, no overwrite)",
            "smoke": smoke,
            "n_train_longs": n_tr_long,
            "n_val_longs": n_va_long,
            "n_train_groups": n_tr_groups,
            "n_val_groups": n_va_groups,
            "n_rows": n_rows_total,
            "layout": LAYOUT,
            "params": PARAMS,
            "validation": {"kind": "per-long hold-out (worldwide cohort); "
                                   "long_is_holdout, chance=0.25"},
            "results": {"B_person_cand_inter": dict(
                auc=float(auc), pick_acc=float(acc), leak_gate=float(gate))},
            "emit": emit_stats,
        }, fh, indent=2)

    print(f"\nSummary -> pick_summary.json   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
