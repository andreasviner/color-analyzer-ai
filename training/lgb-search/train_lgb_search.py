"""
Random search over LightGBM hyperparameters for the three targets, plus
threshold tuning on the gender out-of-fold predictions.

For each target (gender / age / mood):
    1. Sample N_TRIALS hyperparameter configurations.
    2. Score each by 5-fold OOF metric (AUC for gender, MAE for regressions).
    3. Refit the winner with the same 5-fold CV to produce final OOF preds.
    4. For gender, search the OOF threshold that maximises accuracy.

Run features.py first.
"""

import json
import os
import time
import warnings
from copy import deepcopy

import numpy as np
import lightgbm as lgb

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, ".."))   # ai/ root: shared features.npy / targets.npz
N_FOLDS = 5
SEED = 42
N_TRIALS = 30  # per target


def load():
    X = np.load(os.path.join(DATA_DIR, "features.npy"))
    t = np.load(os.path.join(DATA_DIR, "targets.npz"))
    with open(os.path.join(DATA_DIR, "feature_names.json"), encoding="utf-8") as fh:
        names = json.load(fh)
    return X, t["gender"], t["age"], t["mood"], names


def sample_params(rng, task):
    """Sample one LightGBM hyperparameter configuration."""
    p = {
        "n_estimators":    int(rng.choice([400, 600, 800, 1200, 1600, 2000])),
        "learning_rate":   float(rng.choice([0.01, 0.02, 0.03, 0.05])),
        "num_leaves":      int(rng.choice([15, 31, 47, 63, 95, 127])),
        "max_depth":       int(rng.choice([-1, 5, 7, 9])),
        "min_child_samples": int(rng.choice([5, 10, 20, 40, 80])),
        "feature_fraction": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        "bagging_fraction": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        "bagging_freq":    int(rng.choice([0, 3, 5, 7])),
        "reg_alpha":       float(rng.choice([0.0, 0.05, 0.2, 1.0])),
        "reg_lambda":      float(rng.choice([0.0, 0.5, 1.0, 3.0])),
        "min_split_gain":  float(rng.choice([0.0, 0.01, 0.05])),
        "random_state":    SEED,
        "n_jobs":          -1,
        "verbosity":       -1,
    }
    if task == "gender":
        p["objective"] = "binary"
        p["metric"]    = "auc"
    else:
        p["objective"] = "regression"
        p["metric"]    = "mae"
    return p


def cv_score(params, X, y, task):
    """Return (score, oof). For gender, higher is better (AUC). For regression, lower is better (MAE)."""
    if task == "gender":
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof = np.zeros(len(y), dtype=np.float32)
        for tr, va in cv.split(X, y):
            m = lgb.LGBMClassifier(**params)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])
            oof[va] = m.predict_proba(X[va])[:, 1]
        return roc_auc_score(y, oof), oof
    else:
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof = np.zeros(len(y), dtype=np.float32)
        for tr, va in cv.split(X):
            m = lgb.LGBMRegressor(**params)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])
            oof[va] = m.predict(X[va])
        return mean_absolute_error(y, oof), oof


def search(X, y, task, n_trials):
    rng = np.random.default_rng(SEED)
    best_score = -np.inf if task == "gender" else np.inf
    best_params = None
    best_oof = None

    print(f"\n--- LightGBM random search: {task}  ({n_trials} trials) ---")
    for trial in range(n_trials):
        params = sample_params(rng, task)
        t0 = time.time()
        score, oof = cv_score(params, X, y, task)
        dt = time.time() - t0
        better = (score > best_score) if task == "gender" else (score < best_score)
        if better:
            best_score = score
            best_params = params
            best_oof = oof
            marker = "  *"
        else:
            marker = ""
        metric_label = "AUC" if task == "gender" else "MAE"
        print(f"  trial {trial+1:2d}/{n_trials}  {metric_label}={score:.4f}  "
              f"lr={params['learning_rate']:.2f}  leaves={params['num_leaves']:3d}  "
              f"mcs={params['min_child_samples']:3d}  ff={params['feature_fraction']:.1f}  "
              f"l2={params['reg_lambda']:.1f}  n={params['n_estimators']:4d}  "
              f"({dt:.1f}s){marker}")
    return best_score, best_params, best_oof


def tune_threshold(y, prob):
    """Find the threshold that maximises accuracy on the OOF probs."""
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(0.30, 0.70, 81):
        acc = ((prob >= t).astype(int) == y).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def header(text):
    print()
    print("=" * 64)
    print(text)
    print("=" * 64)


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
    X, g, a, m, names = load()
    print(f"X: {X.shape}  features: {len(names)}")
    print(f"trials per target: {N_TRIALS}   folds: {N_FOLDS}   seed: {SEED}")

    # -------- GENDER --------
    g_auc, g_params, g_oof = search(X, g, "gender", N_TRIALS)
    g_thr, g_acc_tuned = tune_threshold(g, g_oof)
    g_pred = (g_oof >= g_thr).astype(int)

    header("GENDER (best LightGBM, OOF, tuned threshold)")
    print(f"  AUC={roc_auc_score(g, g_oof):.4f}")
    print(f"  acc@0.5            = {((g_oof >= 0.5).astype(int) == g).mean():.4f}")
    print(f"  acc@{g_thr:.3f} (tuned) = {g_acc_tuned:.4f}")
    print(f"  F1                 = {f1_score(g, g_pred):.4f}")
    print(f"  best params: {{k: v for k, v in g_params.items() if k not in {{'random_state','n_jobs','verbosity','objective','metric'}}}}".replace("'", '"'))
    for k, v in g_params.items():
        if k in {"random_state", "n_jobs", "verbosity", "objective", "metric"}:
            continue
        print(f"    {k:20s} = {v}")

    # -------- AGE --------
    a_mae, a_params, a_oof = search(X, a, "age", N_TRIALS)
    header("AGE (best LightGBM, OOF)")
    print(f"  MAE = {mean_absolute_error(a, a_oof):.3f}")
    print(f"  R2  = {r2_score(a, a_oof):+.4f}")
    bucket_acc = (age_bucket(a_oof.round().clip(6, 68).astype(int)) == age_bucket(a)).mean()
    print(f"  bucket acc (6 buckets) = {bucket_acc:.4f}")
    for k, v in a_params.items():
        if k in {"random_state", "n_jobs", "verbosity", "objective", "metric"}:
            continue
        print(f"    {k:20s} = {v}")

    # -------- MOOD --------
    m_mae, m_params, m_oof = search(X, m, "mood", N_TRIALS)
    header("MOOD (best LightGBM, OOF)")
    print(f"  MAE = {mean_absolute_error(m, m_oof):.3f}")
    print(f"  R2  = {r2_score(m, m_oof):+.4f}")
    for k, v in m_params.items():
        if k in {"random_state", "n_jobs", "verbosity", "objective", "metric"}:
            continue
        print(f"    {k:20s} = {v}")

    # Save OOF predictions and best params
    np.savez(
        os.path.join(HERE, "lgb_search_oof.npz"),
        gender=g_oof, age=a_oof, mood=m_oof,
        gender_threshold=np.array(g_thr, dtype=np.float32),
    )
    with open(os.path.join(HERE, "lgb_best_params.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"gender": g_params, "age": a_params, "mood": m_params,
             "gender_threshold": g_thr},
            fh, indent=2,
        )
    print("\nwrote lgb_search_oof.npz and lgb_best_params.json")


if __name__ == "__main__":
    main()
