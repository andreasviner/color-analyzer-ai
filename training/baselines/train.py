"""
Train baselines on the color-polygraph features with 5-fold CV.

Three heads, all from the same per-session feature vector:
    gender  - binary  (boy / girl)
    age     - regress (6..68, also reported as bucket accuracy)
    mood    - regress (0 = happy, 60 = glum)

Models compared, each via 5-fold CV (stratified for gender):
    - logistic / ridge baseline
    - histogram gradient boosting (sklearn)
    - LightGBM
    - XGBoost
    - MLP (sklearn, scaled inputs)
    - stacked ensemble (logistic / ridge on out-of-fold predictions)

Run features.py first to produce features.npy / targets.npz.
"""

import json
import os
import time

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, ".."))   # ai/ root: features.npy, targets.npz live here
N_FOLDS = 5
SEED = 42


def load():
    X = np.load(os.path.join(DATA_DIR, "features.npy"))
    t = np.load(os.path.join(DATA_DIR, "targets.npz"))
    with open(os.path.join(DATA_DIR, "feature_names.json"), encoding="utf-8") as fh:
        names = json.load(fh)
    return X, t["gender"], t["age"], t["mood"], names


def header(text):
    print()
    print("=" * 64)
    print(text)
    print("=" * 64)


def age_bucket(y):
    buckets = np.full_like(y, -1)
    buckets[(y >= 6)  & (y <= 9)]  = 0
    buckets[(y >= 10) & (y <= 12)] = 1
    buckets[(y >= 13) & (y <= 15)] = 2
    buckets[(y >= 16) & (y <= 18)] = 3
    buckets[(y >= 19) & (y <= 25)] = 4
    buckets[(y >= 26) & (y <= 68)] = 5
    return buckets


# ---------- Cross-validation runners ----------

def cv_clf(name, build_model, X, y):
    """5-fold stratified CV for a binary classifier. Returns OOF probs."""
    t0 = time.time()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, te in skf.split(X, y):
        m = build_model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    pred = (oof >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, oof)
    f1  = f1_score(y, pred)
    print(f"  {name:30s}  acc={acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}   ({time.time()-t0:.1f}s)")
    return oof


def cv_reg(name, build_model, X, y, span):
    """5-fold CV for a regressor. Returns OOF predictions."""
    t0 = time.time()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, te in kf.split(X):
        m = build_model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    mae = mean_absolute_error(y, oof)
    r2  = r2_score(y, oof)
    print(f"  {name:30s}  MAE={mae:5.2f} ({mae / span * 100:.1f}% of range)  R2={r2:+.3f}   ({time.time()-t0:.1f}s)")
    return oof


# ---------- Model builders ----------

def mk_log():
    return Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def mk_ridge():
    return Pipeline([("sc", StandardScaler()), ("rg", Ridge(alpha=1.0))])


def mk_hgb_clf():
    return HistGradientBoostingClassifier(
        max_iter=500, max_depth=6, learning_rate=0.05,
        l2_regularization=0.5, random_state=SEED,
    )


def mk_hgb_reg():
    return HistGradientBoostingRegressor(
        max_iter=600, max_depth=7, learning_rate=0.05,
        l2_regularization=0.5, random_state=SEED,
    )


def mk_lgb_clf():
    return lgb.LGBMClassifier(
        n_estimators=800, num_leaves=63, learning_rate=0.03,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=20, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )


def mk_lgb_reg():
    return lgb.LGBMRegressor(
        n_estimators=1000, num_leaves=63, learning_rate=0.03,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=20, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )


def mk_xgb_clf():
    return xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, eval_metric="logloss",
        tree_method="hist", verbosity=0,
    )


def mk_xgb_reg():
    return xgb.XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0,
    )


def mk_mlp_clf():
    return Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=300,
            early_stopping=True, validation_fraction=0.1,
            alpha=1e-3, random_state=SEED,
        )),
    ])


def mk_mlp_reg():
    return Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), max_iter=400,
            early_stopping=True, validation_fraction=0.1,
            alpha=1e-3, random_state=SEED,
        )),
    ])


# ---------- Stacking ----------

def stack_clf(name, base_oofs, y):
    """Logistic regression over base OOF probs."""
    X_meta = np.column_stack(base_oofs)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, te in skf.split(X_meta, y):
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(X_meta[tr], y[tr])
        oof[te] = m.predict_proba(X_meta[te])[:, 1]
    pred = (oof >= 0.5).astype(int)
    print(f"  {name:30s}  acc={accuracy_score(y, pred):.3f}  "
          f"AUC={roc_auc_score(y, oof):.3f}  F1={f1_score(y, pred):.3f}")
    return oof


def stack_reg(name, base_oofs, y, span):
    X_meta = np.column_stack(base_oofs)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, te in kf.split(X_meta):
        m = Ridge(alpha=1.0)
        m.fit(X_meta[tr], y[tr])
        oof[te] = m.predict(X_meta[te])
    mae = mean_absolute_error(y, oof)
    r2  = r2_score(y, oof)
    print(f"  {name:30s}  MAE={mae:5.2f} ({mae / span * 100:.1f}% of range)  R2={r2:+.3f}")
    return oof


# ---------- Feature importance ----------

def lgb_gain_importance(model, names, top=25):
    """Average gain of splits using each feature (LightGBM-native)."""
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    order = np.argsort(gains)[::-1]
    print(f"  {'feature':30s}  {'gain':>10s}  {'splits':>7s}")
    for i in order[:top]:
        print(f"  {names[i]:30s}  {gains[i]:10.1f}  {splits[i]:7d}")


# ---------- Main ----------

def main():
    X, gender, age, mood, names = load()
    print(f"X: {X.shape}  features: {len(names)}")
    print(f"folds: {N_FOLDS}  seed: {SEED}")

    # ============ GENDER ============
    header("GENDER  (5-fold stratified CV)")
    g_majority = max(np.bincount(gender)) / len(gender)
    print(f"  baseline (majority class)      acc={g_majority:.3f}")
    oof_log = cv_clf("logistic regression",   mk_log,     X, gender)
    oof_hgb = cv_clf("hist gradient boosting", mk_hgb_clf, X, gender)
    oof_lgb = cv_clf("LightGBM",               mk_lgb_clf, X, gender)
    oof_xgb = cv_clf("XGBoost",                mk_xgb_clf, X, gender)
    oof_mlp = cv_clf("MLP (256,128,64)",       mk_mlp_clf, X, gender)
    stack_clf("stack(log+HGB+LGB+XGB+MLP)",
              [oof_log, oof_hgb, oof_lgb, oof_xgb, oof_mlp], gender)

    # ============ AGE ============
    header("AGE  (5-fold CV, regression on 6..68)")
    span_a = float(age.max() - age.min())
    naive_mae = mean_absolute_error(age, np.full_like(age, age.mean(), dtype=np.float32))
    print(f"  baseline (predict mean)        MAE={naive_mae:5.2f} ({naive_mae / span_a * 100:.1f}% of range)")
    oof_a_ridge = cv_reg("ridge regression",       mk_ridge,   X, age, span_a)
    oof_a_hgb   = cv_reg("hist gradient boosting", mk_hgb_reg, X, age, span_a)
    oof_a_lgb   = cv_reg("LightGBM",               mk_lgb_reg, X, age, span_a)
    oof_a_xgb   = cv_reg("XGBoost",                mk_xgb_reg, X, age, span_a)
    oof_a_mlp   = cv_reg("MLP (256,128,64)",       mk_mlp_reg, X, age, span_a)
    oof_a_stack = stack_reg("stack(ridge+HGB+LGB+XGB+MLP)",
                            [oof_a_ridge, oof_a_hgb, oof_a_lgb, oof_a_xgb, oof_a_mlp],
                            age, span_a)
    pred_buckets = age_bucket(oof_a_stack.round().clip(6, 68).astype(int))
    true_buckets = age_bucket(age)
    bucket_acc = (pred_buckets == true_buckets).mean()
    print(f"  stack age-bucket accuracy      {bucket_acc:.3f}  (6 buckets)")

    # ============ MOOD ============
    header("MOOD  (5-fold CV, regression on 0..60)")
    span_m = 60.0
    naive_mae = mean_absolute_error(mood, np.full_like(mood, mood.mean(), dtype=np.float32))
    print(f"  baseline (predict mean)        MAE={naive_mae:5.2f} ({naive_mae / span_m * 100:.1f}% of range)")
    oof_m_ridge = cv_reg("ridge regression",       mk_ridge,   X, mood, span_m)
    oof_m_hgb   = cv_reg("hist gradient boosting", mk_hgb_reg, X, mood, span_m)
    oof_m_lgb   = cv_reg("LightGBM",               mk_lgb_reg, X, mood, span_m)
    oof_m_xgb   = cv_reg("XGBoost",                mk_xgb_reg, X, mood, span_m)
    oof_m_mlp   = cv_reg("MLP (256,128,64)",       mk_mlp_reg, X, mood, span_m)
    stack_reg("stack(ridge+HGB+LGB+XGB+MLP)",
              [oof_m_ridge, oof_m_hgb, oof_m_lgb, oof_m_xgb, oof_m_mlp],
              mood, span_m)

    # ============ Feature importance ============
    header("FEATURE IMPORTANCE  (gender LightGBM, gain-based, top 25)")
    full = mk_lgb_clf()
    full.fit(X, gender)
    lgb_gain_importance(full, names, top=25)


if __name__ == "__main__":
    print("Training color-polygraph baselines (5-fold CV)...")
    main()
