"""
Rescue partial results from a killed overnight_search.py run.

Given a target (gender / age / mood) and a set of families that completed at
least a few trials in trial_results.csv, this script:

    1. Reads trial_results.csv, picks the top-K configurations per family
       for that target (by best CV score).
    2. Refits each one with the same 5-fold CV to regenerate its OOF preds.
    3. Stacks the top-K from each family with a regularized meta-learner.
    4. For gender, tunes the threshold; for age/mood, reports MAE / R2.
    5. Writes the same outputs run_target() in overnight_search.py would:
           summary_<target>.json
           oof_<target>.npz
           best_params_<target>.json
           importance_<target>.json   (LGB only)

Usage:
    python rescue_target.py --target gender
    python rescue_target.py --target gender --families lgb,xgb
    python rescue_target.py --target gender --top-lgb 5 --top-xgb 4 --top-cat 3

A refit of 5-fold CV for a single LGB or XGB config takes a few minutes; the
whole rescue typically finishes in 20-60 min depending on how many families
you stack.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    f1_score, mean_absolute_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAVE_CATBOOST = True
except ImportError:
    HAVE_CATBOOST = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, ".."))   # ai/ root: shared features.npy / targets.npz
OUT_DIR = os.path.join(DATA_DIR, "overnight_out")
TRIALS_CSV = os.path.join(OUT_DIR, "trial_results.csv")

N_FOLDS = 5
SEED = 42

ALL_FAMILIES = ["lgb", "xgb", "cat", "hgb", "et"]


def log(msg):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def load_data():
    X = np.load(os.path.join(DATA_DIR, "features.npy"))
    t = np.load(os.path.join(DATA_DIR, "targets.npz"))
    return X, {"gender": t["gender"], "age": t["age"], "mood": t["mood"]}


def read_top_trials(target, family, k):
    """Return list of (score, params_dict), sorted best -> worst, length <= k."""
    if not os.path.exists(TRIALS_CSV):
        sys.exit(f"missing: {TRIALS_CSV}")
    rows = []
    with open(TRIALS_CSV, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r["target"] != target or r["family"] != family:
                continue
            try:
                score = float(r["score"])
                params = json.loads(r["params"])
            except (ValueError, json.JSONDecodeError):
                continue
            rows.append((score, params))
    # gender wants AUC (higher better); age/mood want MAE (lower better)
    reverse = (target == "gender")
    rows.sort(key=lambda x: x[0], reverse=reverse)
    return rows[:k]


def build_model(family, params, task):
    if family == "lgb":
        return (lgb.LGBMClassifier(**params) if task == "gender"
                else lgb.LGBMRegressor(**params))
    if family == "xgb":
        return (xgb.XGBClassifier(**params) if task == "gender"
                else xgb.XGBRegressor(**params))
    if family == "cat":
        if not HAVE_CATBOOST:
            raise RuntimeError("catboost not installed")
        return (CatBoostClassifier(**params) if task == "gender"
                else CatBoostRegressor(**params))
    if family == "hgb":
        return (HistGradientBoostingClassifier(**params) if task == "gender"
                else HistGradientBoostingRegressor(**params))
    if family == "et":
        return (ExtraTreesClassifier(**params) if task == "gender"
                else ExtraTreesRegressor(**params))
    raise ValueError(family)


def fit_with_es(family, m, X_tr, y_tr, X_va, y_va):
    if family == "lgb":
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(0)])
    elif family == "xgb":
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif family == "cat":
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    elif family in ("hgb", "et"):
        m.fit(X_tr, y_tr)
    else:
        raise ValueError(family)


def cv_oof(family, params, X, y, task, splits):
    oof = np.zeros(len(y), dtype=np.float32)
    fitted = []
    for tr, va in splits:
        m = build_model(family, params, task)
        fit_with_es(family, m, X[tr], y[tr], X[va], y[va])
        if task == "gender":
            oof[va] = m.predict_proba(X[va])[:, 1]
        else:
            oof[va] = m.predict(X[va])
        fitted.append(m)
    return oof, fitted


def stack_gender(oofs, y):
    X_meta = np.column_stack(oofs)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    Cs = [0.01, 0.05, 0.2, 1.0, 5.0, 25.0]
    for tr, va in cv.split(X_meta, y):
        m = LogisticRegressionCV(Cs=Cs, cv=3, max_iter=4000,
                                 scoring="roc_auc", n_jobs=-1)
        m.fit(X_meta[tr], y[tr])
        oof[va] = m.predict_proba(X_meta[va])[:, 1]
    return oof


def stack_reg(oofs, y):
    """Stack regression OOFs with a non-negative L1-loss blend.

    Ridge / L2 stacking is wrong for MAE-trained base learners: each base
    OOF is approximately the conditional median, and a weighted L2 fit
    pulls the blend toward the conditional mean, which inflates MAE. We
    minimise sum |X@w - y| with non-negative weights, which preserves the
    median property.
    """
    X_meta = np.column_stack(oofs)
    n_feat = X_meta.shape[1]
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, va in cv.split(X_meta):
        def loss(w):
            return np.abs(X_meta[tr] @ w - y[tr]).mean()
        # warm-start with equal weights normalised so blend predicts ~y
        w0 = np.full(n_feat, 1.0 / n_feat)
        res = minimize(loss, x0=w0, bounds=[(0.0, 1.0)] * n_feat,
                       method="L-BFGS-B")
        oof[va] = (X_meta[va] @ res.x).astype(np.float32)
    return oof


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


def lgb_importance(models):
    if not models:
        return None
    gains = [m.booster_.feature_importance(importance_type="gain") for m in models]
    return np.mean(np.stack(gains, axis=0), axis=0).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["gender", "age", "mood"])
    parser.add_argument("--families", default="lgb,xgb,cat",
                        help="Comma-separated subset of lgb,xgb,cat,hgb,et")
    parser.add_argument("--top-lgb", type=int, default=5)
    parser.add_argument("--top-xgb", type=int, default=4)
    parser.add_argument("--top-cat", type=int, default=3)
    parser.add_argument("--top-hgb", type=int, default=3)
    parser.add_argument("--top-et",  type=int, default=2)
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    top_k = {
        "lgb": args.top_lgb, "xgb": args.top_xgb, "cat": args.top_cat,
        "hgb": args.top_hgb, "et":  args.top_et,
    }

    target = args.target
    log(f"Rescue target={target}, families={families}")
    log(f"Top-K per family: {top_k}")

    X, ys = load_data()
    y = ys[target]
    log(f"X={X.shape}, y={y.shape}")

    # Build CV splits ONCE so every refit uses the exact same folds
    if target == "gender":
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(cv.split(X, y))
    else:
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(cv.split(X))

    base_oofs = []
    per_family_best_oof = {}
    per_family_best_score = {}
    per_family_best_params = {}
    lgb_models_best = None

    for family in families:
        if family == "cat" and not HAVE_CATBOOST:
            log(f"--- skipping cat (catboost not installed) ---")
            continue
        k = top_k.get(family, 0)
        if k <= 0:
            continue
        top = read_top_trials(target, family, k)
        if not top:
            log(f"--- {family}: no trials in CSV, skipping ---")
            continue
        log(f"--- {family}: refitting top {len(top)} trials ---")

        family_oofs = []
        family_models = []
        t_fam = time.time()
        for i, (score, params) in enumerate(top):
            t0 = time.time()
            try:
                oof, models = cv_oof(family, params, X, y, target, splits)
            except Exception as exc:
                log(f"  rank {i+1}: FAILED {type(exc).__name__}: {exc}")
                continue
            dt = time.time() - t0
            if target == "gender":
                refit_score = float(roc_auc_score(y, oof))
                metric = "AUC"
            else:
                refit_score = float(mean_absolute_error(y, oof))
                metric = "MAE"
            log(f"  rank {i+1}/{len(top)}  csv_score={score:.4f}  refit_{metric}={refit_score:.4f}  ({dt:.1f}s)")
            family_oofs.append(oof)
            family_models.append(models)
            if i == 0:
                per_family_best_oof[family] = oof
                per_family_best_score[family] = refit_score
                per_family_best_params[family] = params
                if family == "lgb":
                    lgb_models_best = models

        log(f"--- {family}: done in {(time.time()-t_fam)/60:.1f} min, "
            f"{len(family_oofs)} OOFs ---")
        base_oofs.extend(family_oofs)

        # Write incremental per-family file
        if family_oofs:
            np.savez(os.path.join(OUT_DIR, f"oof_{target}_{family}.npz"),
                     best_oof=family_oofs[0],
                     top_oofs=np.column_stack(family_oofs))

    if not base_oofs:
        sys.exit("no families produced OOFs, nothing to stack")

    log(f"=== stacking {len(base_oofs)} base OOFs ===")

    if target == "gender":
        stacked = stack_gender(base_oofs, y)
        thr, acc_t = tune_threshold(y, stacked)
        summary = {
            "stacked_auc":        float(roc_auc_score(y, stacked)),
            "stacked_acc_at_0.5": float(((stacked >= 0.5).astype(int) == y).mean()),
            "stacked_acc_tuned":  acc_t,
            "stacked_threshold":  thr,
            "stacked_f1":         float(f1_score(y, (stacked >= thr).astype(int))),
            "n_base_models_in_stack": len(base_oofs),
        }
        for f, s in per_family_best_score.items():
            summary[f"best_{f}_auc"] = s
        np_kwargs = {"stacked": stacked, "threshold": np.array(thr, dtype=np.float32)}
        for f, o in per_family_best_oof.items():
            np_kwargs[f] = o
        np.savez(os.path.join(OUT_DIR, f"oof_{target}.npz"), **np_kwargs)
    else:
        stacked = stack_reg(base_oofs, y)
        summary = {
            "stacked_mae": float(mean_absolute_error(y, stacked)),
            "stacked_r2":  float(r2_score(y, stacked)),
            "n_base_models_in_stack": len(base_oofs),
        }
        for f, s in per_family_best_score.items():
            summary[f"best_{f}_mae"] = s
        if target == "age":
            summary["stacked_bucket_acc"] = float(
                (age_bucket(stacked.round().clip(6, 68).astype(int))
                 == age_bucket(y)).mean()
            )
        np_kwargs = {"stacked": stacked}
        for f, o in per_family_best_oof.items():
            np_kwargs[f] = o
        np.savez(os.path.join(OUT_DIR, f"oof_{target}.npz"), **np_kwargs)

    with open(os.path.join(OUT_DIR, f"summary_{target}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(OUT_DIR, f"best_params_{target}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(per_family_best_params, fh, indent=2, default=str)
    if lgb_models_best is not None:
        imp = lgb_importance(lgb_models_best)
        if imp is not None:
            with open(os.path.join(OUT_DIR, f"importance_{target}.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"gain": imp}, fh, indent=2)

    log("=== rescue summary ===")
    for k, v in summary.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
