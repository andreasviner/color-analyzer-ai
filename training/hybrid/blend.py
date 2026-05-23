"""
Hybrid blend of all available architectures on the color-polygraph dataset.

The idea: the GBM stack, the transformer, and the LSTM are very different
inductive biases. The trees consume 441 engineered tabular features; the
sequence models read the raw 21-step per-question stream. Each one almost
certainly captures signal the others miss.

This script loads every available OOF prediction file and blends them with a
simple regularised meta-learner. For gender we use LogisticRegressionCV over
the OOF probs (all base models share the same StratifiedKFold(5, seed=42)
fold definition, so blending is clean). For age and mood we use RidgeCV; the
GBM stack uses KFold and the sequence models use StratifiedKFold, so the
meta-CV has a tiny amount of fold-mismatch leakage, but the resulting MAE is
still a reasonable estimate of what the blend would achieve in production
(the per-model OOFs themselves are individually unbiased).

Inputs (loaded if present):
    ../overnight_out/oof_gender.npz   -> "stacked" key (or family bests)
    ../overnight_out/oof_age.npz
    ../overnight_out/oof_mood.npz
    ../transformer/transformer_oof.npz
    ../seq/seq_oof.npz

Outputs:
    summary.json    final per-target metrics
    blend_oof.npz   blended OOF prediction arrays
"""

import json
import os

import numpy as np
from scipy.optimize import minimize, nnls
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.normpath(os.path.join(HERE, ".."))
SEED = 42
N_FOLDS = 5


def load_targets():
    t = np.load(os.path.join(TRAIN_DIR, "targets.npz"))
    return t["gender"], t["age"], t["mood"]


def load_base_oofs():
    """Return {target: {model_name: oof_array}}."""
    out = {"gender": {}, "age": {}, "mood": {}}

    gpath = os.path.join(TRAIN_DIR, "overnight_out", "oof_gender.npz")
    if os.path.exists(gpath):
        d = np.load(gpath)
        if "stacked" in d.files:
            out["gender"]["gbm_stack"] = d["stacked"]
        # If the stack file doesn't exist, fall back to the LGB OOF
        elif "lgb" in d.files:
            out["gender"]["gbm_lgb"] = d["lgb"]

    for tgt in ("age", "mood"):
        p = os.path.join(TRAIN_DIR, "overnight_out", f"oof_{tgt}.npz")
        if os.path.exists(p):
            d = np.load(p)
            if "stacked" in d.files:
                out[tgt]["gbm_stack"] = d["stacked"]
            elif "lgb" in d.files:
                out[tgt][f"gbm_lgb"] = d["lgb"]

    tpath = os.path.join(TRAIN_DIR, "transformer", "transformer_oof.npz")
    if os.path.exists(tpath):
        d = np.load(tpath)
        for tgt in ("gender", "age", "mood"):
            if tgt in d.files:
                out[tgt]["transformer"] = d[tgt]

    spath = os.path.join(TRAIN_DIR, "seq", "seq_oof.npz")
    if os.path.exists(spath):
        d = np.load(spath)
        for tgt in ("gender", "age", "mood"):
            if tgt in d.files:
                out[tgt]["lstm"] = d[tgt]

    upath = os.path.join(TRAIN_DIR, "gru", "gru_oof.npz")
    if os.path.exists(upath):
        d = np.load(upath)
        for tgt in ("gender", "age", "mood"):
            if tgt in d.files:
                out[tgt]["bigru"] = d[tgt]

    return out


def nnls_blend_oof(X, y, cv):
    """Non-negative least squares blend, evaluated by holdout-CV.

    For each fold, fit weights w >= 0 minimising ||y_tr - X_tr @ w||^2 on the
    training rows, then predict val rows as X_va @ w. The resulting OOF
    prediction is honest (no train-set rows used at their own meta-prediction
    time). NNLS gives a positive-weighted blend, which is the right inductive
    bias when every base model has positive predictive power and probabilities
    must compose linearly.
    """
    oof = np.zeros(len(y), dtype=np.float32)
    weights = []
    for tr, va in cv:
        w, _ = nnls(X[tr], y[tr].astype(np.float64))
        oof[va] = (X[va] @ w).astype(np.float32)
        weights.append(w)
    return oof, np.array(weights).mean(axis=0)


def blend_gender(base, y):
    """Two evaluations: positive-weighted NNLS (additive) and logistic CV.
    Report whichever produces the higher OOF AUC.
    """
    names = list(base.keys())
    X = np.column_stack([base[n] for n in names])

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, y))

    # NNLS blend
    oof_nnls, w_nnls = nnls_blend_oof(X, y, splits)

    # Logistic CV blend (more flexible — can downweight noisy models via L2)
    oof_lr = np.zeros(len(y), dtype=np.float32)
    coefs_lr = []
    for tr, va in splits:
        m = LogisticRegressionCV(
            Cs=[0.01, 0.05, 0.2, 1.0, 5.0, 25.0],
            cv=3, max_iter=4000,
            scoring="roc_auc", n_jobs=-1,
        )
        m.fit(X[tr], y[tr])
        oof_lr[va] = m.predict_proba(X[va])[:, 1]
        coefs_lr.append(m.coef_.ravel())

    auc_nnls = float(roc_auc_score(y, oof_nnls))
    auc_lr   = float(roc_auc_score(y, oof_lr))

    if auc_nnls >= auc_lr:
        oof = oof_nnls
        method = "nnls"
        avg_coef = w_nnls.tolist()
    else:
        oof = oof_lr
        method = "logistic_cv"
        avg_coef = np.mean(coefs_lr, axis=0).tolist()

    thr, acc_t = tune_threshold(y, oof)
    pred = (oof >= thr).astype(int)
    return {
        "models":     names,
        "method":     method,
        "avg_coef":   [float(c) for c in avg_coef],
        "auc":        float(roc_auc_score(y, oof)),
        "auc_nnls":   auc_nnls,
        "auc_lr":     auc_lr,
        "acc_at_0.5": float(((oof >= 0.5).astype(int) == y).mean()),
        "acc_tuned":  acc_t,
        "threshold":  thr,
        "f1":         float(f1_score(y, pred)),
        "oof":        oof,
    }


def l1_blend_oof(X, y, splits):
    """Non-negative L1-loss blend, evaluated by holdout-CV.
    Minimises sum |X@w - y| over w >= 0 per fold. Preserves the median
    property of MAE-trained base regressors, where Ridge stacking fails.
    """
    n_feat = X.shape[1]
    oof = np.zeros(len(y), dtype=np.float32)
    weights = []
    for tr, va in splits:
        def loss(w):
            return np.abs(X[tr] @ w - y[tr]).mean()
        w0 = np.full(n_feat, 1.0 / n_feat)
        res = minimize(loss, x0=w0, bounds=[(0.0, 1.0)] * n_feat,
                       method="L-BFGS-B")
        oof[va] = (X[va] @ res.x).astype(np.float32)
        weights.append(res.x)
    return oof, np.array(weights).mean(axis=0)


def blend_reg(base, y):
    """L1-blend vs NNLS vs RidgeCV; pick whichever has lower OOF MAE."""
    names = list(base.keys())
    X = np.column_stack([base[n] for n in names])

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(cv.split(X))

    oof_l1, w_l1 = l1_blend_oof(X, y, splits)
    oof_nnls, w_nnls = nnls_blend_oof(X, y, splits)

    oof_ridge = np.zeros(len(y), dtype=np.float32)
    coefs_ridge = []
    for tr, va in splits:
        m = RidgeCV(alphas=[0.01, 0.1, 1.0, 5.0, 25.0, 100.0], cv=3)
        m.fit(X[tr], y[tr])
        oof_ridge[va] = m.predict(X[va])
        coefs_ridge.append(m.coef_)

    mae_l1    = float(mean_absolute_error(y, oof_l1))
    mae_nnls  = float(mean_absolute_error(y, oof_nnls))
    mae_ridge = float(mean_absolute_error(y, oof_ridge))

    candidates = [
        ("l1", mae_l1, oof_l1, w_l1),
        ("nnls", mae_nnls, oof_nnls, w_nnls),
        ("ridge_cv", mae_ridge, oof_ridge, np.mean(coefs_ridge, axis=0)),
    ]
    method, _, oof, coef = min(candidates, key=lambda c: c[1])

    return {
        "models":    names,
        "method":    method,
        "avg_coef":  [float(c) for c in coef.tolist()],
        "mae":       float(mean_absolute_error(y, oof)),
        "mae_l1":    mae_l1,
        "mae_nnls":  mae_nnls,
        "mae_ridge": mae_ridge,
        "r2":        float(r2_score(y, oof)),
        "oof":       oof,
    }


def tune_threshold(y, prob):
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(0.20, 0.80, 241):
        acc = ((prob >= t).astype(int) == y).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, float(best_acc)


def age_bucket(y):
    out = np.full_like(y, -1)
    out[(y >= 6)  & (y <= 9)]  = 0
    out[(y >= 10) & (y <= 12)] = 1
    out[(y >= 13) & (y <= 15)] = 2
    out[(y >= 16) & (y <= 18)] = 3
    out[(y >= 19) & (y <= 25)] = 4
    out[(y >= 26) & (y <= 68)] = 5
    return out


def main():
    g, a, m = load_targets()
    print(f"targets: gender={g.shape}, age={a.shape}, mood={m.shape}")
    print(f"  boys/girls={int((g==0).sum())}/{int((g==1).sum())}")

    base = load_base_oofs()
    print(f"\nbase OOFs available:")
    for tgt, models in base.items():
        print(f"  {tgt}: {list(models.keys())}")

    summary = {}
    oof_to_save = {}

    if base["gender"]:
        print("\n--- GENDER blend ---")
        res = blend_gender(base["gender"], g)
        print(f"  models: {res['models']}")
        print(f"  method: {res['method']}  (nnls AUC={res['auc_nnls']:.4f}, lr AUC={res['auc_lr']:.4f})")
        print(f"  avg_coef: {[round(c, 3) for c in res['avg_coef']]}")
        print(f"  AUC = {res['auc']:.4f}")
        print(f"  acc@0.5 = {res['acc_at_0.5']:.4f}")
        print(f"  acc@{res['threshold']:.3f} = {res['acc_tuned']:.4f}")
        print(f"  F1  = {res['f1']:.4f}")
        oof_to_save["gender"] = res["oof"]
        summary["gender"] = {k: v for k, v in res.items() if k != "oof"}

    if base["age"]:
        print("\n--- AGE blend ---")
        res = blend_reg(base["age"], a)
        print(f"  models: {res['models']}")
        print(f"  method: {res['method']}  (l1 MAE={res['mae_l1']:.3f}, nnls MAE={res['mae_nnls']:.3f}, ridge MAE={res['mae_ridge']:.3f})")
        print(f"  avg_coef: {[round(c, 3) for c in res['avg_coef']]}")
        print(f"  MAE = {res['mae']:.3f}")
        print(f"  R2  = {res['r2']:+.4f}")
        ba = (age_bucket(res["oof"].round().clip(6, 68).astype(int)) == age_bucket(a)).mean()
        print(f"  bucket acc = {ba:.4f}")
        oof_to_save["age"] = res["oof"]
        summary["age"] = {k: v for k, v in res.items() if k != "oof"}
        summary["age"]["bucket_acc"] = float(ba)

    if base["mood"]:
        print("\n--- MOOD blend ---")
        res = blend_reg(base["mood"], m)
        print(f"  models: {res['models']}")
        print(f"  method: {res['method']}  (l1 MAE={res['mae_l1']:.3f}, nnls MAE={res['mae_nnls']:.3f}, ridge MAE={res['mae_ridge']:.3f})")
        print(f"  avg_coef: {[round(c, 3) for c in res['avg_coef']]}")
        print(f"  MAE = {res['mae']:.3f}")
        print(f"  R2  = {res['r2']:+.4f}")
        oof_to_save["mood"] = res["oof"]
        summary["mood"] = {k: v for k, v in res.items() if k != "oof"}

    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    np.savez(os.path.join(HERE, "blend_oof.npz"), **oof_to_save)
    print(f"\nwrote summary.json and blend_oof.npz to {HERE}")


if __name__ == "__main__":
    main()
