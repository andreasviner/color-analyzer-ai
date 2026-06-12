"""
Tuning harness for the taste-cube model. Not part of the build; run by hand to
explore the effect of (a) how many probes we draw per survey-taker, (b) which
feature blocks matter, and (c) LightGBM hyperparameters -- all on a FIXED
session-level validation set so the numbers are comparable.

Speed trick: every eligible probe row is independent of how many probes we keep
per person (a probe row for question q only depends on q and its donor), so we
build ALL eligible rows once and subset in memory across the sweep.

Run:  python tune_taste.py
"""

import json
import os
import sys
import time

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import taste_features as tf  # noqa: E402
import train_taste as tt      # noqa: E402  (reuse _is_valid, _parse)

SEED = 42
VAL_SIZE = 0.10
FP, CAND, INTER = (tf.feature_layout()["fingerprint"],
                   tf.feature_layout()["candidate"],
                   tf.feature_layout()["interaction"])


def build_all():
    with open(tt.RAW_SOURCE, encoding="utf-8") as fh:
        raw = json.load(fh)
    sessions = [tt._parse(r) for r in raw if tt._is_valid(r)]
    X, y, sid, gid = [], [], [], []
    for s in sessions:
        for row, label, g in tf.build_probe_rows(s, n_probes=0):  # 0 = all eligible
            X.append(row); y.append(label); sid.append(s["id"]); gid.append(g)
    return (np.asarray(X, np.float32), np.asarray(y, np.int8), sid, gid, len(sessions))


def even_subset_mask(sid, gid, keep_sessions, n_probes):
    """Mask selecting rows whose session is in keep_sessions and whose probe is
    among an evenly spread subset of up to n_probes per session (0 = keep all)."""
    # ordered distinct groups per session (rows come in session/q order)
    groups_by_sess = {}
    for i, s in enumerate(sid):
        if s not in keep_sessions:
            continue
        g = gid[i]
        lst = groups_by_sess.setdefault(s, [])
        if not lst or lst[-1] != g:
            lst.append(g)
    chosen = set()
    for s, gl in groups_by_sess.items():
        if n_probes and len(gl) > n_probes:
            step = len(gl) / n_probes
            gl = [gl[int(k * step)] for k in range(n_probes)]
        chosen.update(gl)
    return np.array([sid[i] in keep_sessions and gid[i] in chosen for i in range(len(sid))])


def pick_acc(model, X, y, gid_sub, blocks="all"):
    Xs = X
    if blocks != "all":
        Xs = X.copy()
        if "fp" not in blocks:    Xs[:, :FP] = 0.0
        if "cand" not in blocks:  Xs[:, FP:FP + CAND] = 0.0
        if "inter" not in blocks: Xs[:, FP + CAND:] = 0.0
    raw = model.predict(Xs, raw_score=True)
    by = {}
    for i, g in enumerate(gid_sub):
        by.setdefault(g, []).append(i)
    ok = tot = 0
    for g, idxs in by.items():
        true_local = next((k for k, i in enumerate(idxs) if y[i] == 1), None)
        if true_local is None or len(idxs) < 2:
            continue
        ok += int(int(np.argmax([raw[i] for i in idxs])) == true_local); tot += 1
    return ok / tot if tot else 0.0


def main():
    t0 = time.time()
    print("Building all eligible probe rows once...")
    X, y, sid, gid, n_sess = build_all()
    print(f"  {X.shape}  from {n_sess} sessions")

    uniq = sorted(set(sid))
    tr_sess, va_sess = train_test_split(uniq, test_size=VAL_SIZE, random_state=SEED)
    tr_set, va_set = set(tr_sess), set(va_sess)

    # Fixed val set: ALL eligible probes of the val sessions.
    va_mask = np.array([s in va_set for s in sid])
    Xva, yva = X[va_mask], y[va_mask]
    gva = [gid[i] for i in range(len(sid)) if va_mask[i]]
    print(f"  fixed val rows {len(yva)}  groups {len(set(gva))}")

    base = dict(n_estimators=400, num_leaves=63, learning_rate=0.03,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                min_child_samples=40, reg_lambda=1.0,
                random_state=SEED, n_jobs=-1, verbosity=-1)

    def fit_eval(params, n_probes, label):
        m = even_subset_mask(sid, gid, tr_set, n_probes)
        clf = lgb.LGBMClassifier(**params).fit(X[m], y[m])
        auc = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        acc = pick_acc(clf, Xva, yva, gva)
        print(f"  {label:30s} train_rows={int(m.sum()):>7}  AUC={auc:.4f}  pickacc={acc:.4f}")
        return acc, clf

    print("\n[1] PROBES_PER_SESSION sweep (base hyperparams):")
    for n in (2, 3, 5, 8, 12, 0):
        fit_eval(base, n, f"probes={n if n else 'all'}")

    print("\n[2] Feature-block ablation (probes=8):")
    m = even_subset_mask(sid, gid, tr_set, 8)
    clf = lgb.LGBMClassifier(**base).fit(X[m], y[m])
    for blocks in (("all",), ("fp",), ("cand",), ("cand", "inter"), ("fp", "cand")):
        bl = "all" if blocks == ("all",) else set(blocks)
        acc = pick_acc(clf, Xva, yva, gva, blocks=bl)
        print(f"  blocks={str(blocks):24s} pickacc={acc:.4f}")

    print("\n[3] Hyperparameter sweep (probes=8):")
    grid = [
        dict(n_estimators=400, num_leaves=63, learning_rate=0.03, min_child_samples=40),
        dict(n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=40),
        dict(n_estimators=800, num_leaves=127, learning_rate=0.02, min_child_samples=60),
        dict(n_estimators=1200, num_leaves=63, learning_rate=0.02, min_child_samples=80),
        dict(n_estimators=600, num_leaves=31, learning_rate=0.05, min_child_samples=40),
        dict(n_estimators=1500, num_leaves=63, learning_rate=0.015, min_child_samples=100),
    ]
    best = (0, None)
    for g in grid:
        p = dict(base); p.update(g)
        acc, _ = fit_eval(p, 8, str(g))
        if acc > best[0]:
            best = (acc, g)
    print(f"\nBest hyperparams: pickacc={best[0]:.4f}  {best[1]}")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
