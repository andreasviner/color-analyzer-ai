"""
Overnight hyperparameter search + stacking ensemble.

For each target (gender / age / mood):
    Phase 1: random search over LightGBM
    Phase 2: random search over XGBoost
    Phase 3: random search over CatBoost
    Phase 4: random search over sklearn HistGradientBoosting
    Phase 5: random search over sklearn ExtraTrees
    Phase 6: stack top-K OOFs from each family with a regularized meta-learner
             (LogisticRegression for gender, Ridge for regression), inner-CV
             tuned regularization strength.
    Phase 7: for gender, tune the decision threshold on the stacked OOF probs.

Outputs (written to ./overnight_out/):
    log.txt                       incremental console log
    summary.json                  final metrics + best params per family
    trial_results.csv             one row per trial: family/target/score/params
    oof_<target>.npz              stacked + best-per-family OOF preds (+ thr for gender)
    best_params_<target>.json     best hyperparam dict per family, easy to refit from
    importance_<target>.json      gain importances of the best LGB model on this target

Run with:
    python overnight_search.py                          # full overnight run
    python overnight_search.py --quick                  # 5 trials per family, smoke test
    python overnight_search.py --skip-existing          # resume: skip targets whose
                                                        #   summary file already exists
    python overnight_search.py --targets gender         # only gender
    python overnight_search.py --families lgb,xgb       # only those families

Wall time on a recent laptop, default config, all three targets:
    roughly 8-12 hours.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
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
    accuracy_score, f1_score, mean_absolute_error,
    r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAVE_CATBOOST = True
except ImportError:
    HAVE_CATBOOST = False

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*does not have valid feature names.*",
    category=UserWarning,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, ".."))   # ai/ root: shared features.npy / targets.npz
OUT_DIR = os.path.join(DATA_DIR, "overnight_out")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "log.txt")
TRIALS_CSV = os.path.join(OUT_DIR, "trial_results.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.json")

N_FOLDS = 5
SEED = 42

ALL_FAMILIES = ["lgb", "xgb", "cat", "hgb", "et"]
ALL_TARGETS = ["gender", "age", "mood"]


# ---------- Config ----------

@dataclass
class Config:
    lgb_trials: int = 120
    xgb_trials: int = 100
    cat_trials: int = 80
    hgb_trials: int = 50
    et_trials: int = 40
    top_lgb: int = 5
    top_xgb: int = 4
    top_cat: int = 4
    top_hgb: int = 3
    top_et: int = 2
    families: list = field(default_factory=lambda: list(ALL_FAMILIES))
    targets: list = field(default_factory=lambda: list(ALL_TARGETS))

    def trials_for(self, family):
        return {
            "lgb": self.lgb_trials, "xgb": self.xgb_trials,
            "cat": self.cat_trials, "hgb": self.hgb_trials,
            "et":  self.et_trials,
        }[family]

    def top_for(self, family):
        return {
            "lgb": self.top_lgb, "xgb": self.top_xgb,
            "cat": self.top_cat, "hgb": self.top_hgb,
            "et":  self.top_et,
        }[family]


# ---------- Logging ----------

def log(msg):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_trial(family, target, trial, score, params, dt):
    new = not os.path.exists(TRIALS_CSV)
    with open(TRIALS_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["family", "target", "trial", "score", "dt", "params"])
        w.writerow([family, target, trial, f"{score:.6f}",
                    f"{dt:.2f}", json.dumps(params)])


# ---------- Data ----------

def load():
    X = np.load(os.path.join(DATA_DIR, "features.npy"))
    t = np.load(os.path.join(DATA_DIR, "targets.npz"))
    return X, t["gender"], t["age"], t["mood"]


# ---------- Sampling helpers ----------

def log_uniform(rng, lo, hi):
    """Log-uniform sample in [lo, hi]."""
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def choose(rng, options):
    """rng.choice that does not coerce ints to floats when the list is mixed."""
    return options[int(rng.integers(0, len(options)))]


# ---------- Hyperparameter sampling per family ----------

def sample_lgb(rng, task):
    p = {
        "n_estimators":      choose(rng, [400, 600, 800, 1200, 1600, 2000]),
        "learning_rate":     log_uniform(rng, 0.015, 0.08),
        "num_leaves":        choose(rng, [15, 23, 31, 47, 63, 95, 127, 191]),
        "max_depth":         choose(rng, [-1, 5, 6, 7, 9, 12]),
        "min_child_samples": choose(rng, [5, 10, 20, 40, 80, 120]),
        "feature_fraction":  choose(rng, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "bagging_fraction":  choose(rng, [0.6, 0.7, 0.8, 0.9, 1.0]),
        "bagging_freq":      choose(rng, [0, 3, 5, 7]),
        "reg_alpha":         log_uniform(rng, 1e-3, 3.0),
        "reg_lambda":        log_uniform(rng, 1e-3, 6.0),
        "min_split_gain":    log_uniform(rng, 1e-3, 0.2),
        "extra_trees":       bool(rng.integers(0, 2)),
        "random_state":      SEED,
        "n_jobs":            -1,
        "verbosity":         -1,
    }
    p["objective"] = "binary"         if task == "gender" else "regression_l1"
    p["metric"]    = "auc"            if task == "gender" else "mae"
    return p


def sample_xgb(rng, task):
    p = {
        "n_estimators":      choose(rng, [400, 600, 800, 1200, 1600, 2000]),
        "learning_rate":     log_uniform(rng, 0.015, 0.08),
        "max_depth":         choose(rng, [4, 5, 6, 7, 8]),
        "min_child_weight":  log_uniform(rng, 1.0, 20.0),
        "subsample":         choose(rng, [0.6, 0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree":  choose(rng, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "colsample_bylevel": choose(rng, [0.7, 0.85, 1.0]),
        "reg_alpha":         log_uniform(rng, 1e-3, 3.0),
        "reg_lambda":        log_uniform(rng, 1e-3, 6.0),
        "gamma":             log_uniform(rng, 1e-3, 0.5),
        "max_bin":           choose(rng, [128, 256]),
        "tree_method":       "hist",
        "random_state":      SEED,
        "n_jobs":            -1,
        "verbosity":         0,
        "early_stopping_rounds": 40,
    }
    if task == "gender":
        p["eval_metric"] = "auc"
        p["objective"]   = "binary:logistic"
    else:
        p["eval_metric"] = "mae"
        p["objective"]   = "reg:absoluteerror"
    return p


def sample_cat(rng, task):
    p = {
        "iterations":        choose(rng, [400, 600, 800, 1200, 1600]),
        "learning_rate":     log_uniform(rng, 0.02, 0.1),
        "depth":             choose(rng, [4, 5, 6, 7, 8]),
        "l2_leaf_reg":       log_uniform(rng, 0.5, 20.0),
        "border_count":      choose(rng, [64, 128, 254]),
        "bagging_temperature": float(rng.uniform(0.0, 1.0)),
        "rsm":               choose(rng, [0.6, 0.7, 0.8, 0.9, 1.0]),
        "random_strength":   log_uniform(rng, 0.2, 8.0),
        "leaf_estimation_iterations": choose(rng, [1, 5]),
        "od_type":           "Iter",
        "od_wait":           30,
        "random_seed":       SEED,
        "thread_count":      -1,
        "verbose":           False,
        "allow_writing_files": False,
    }
    if task == "gender":
        p["loss_function"] = "Logloss"
        p["eval_metric"]   = "AUC"
    else:
        p["loss_function"] = "MAE"
        p["eval_metric"]   = "MAE"
    return p


def sample_hgb(rng, task):
    p = {
        "max_iter":           choose(rng, [400, 600, 800, 1200, 1600, 2400]),
        "learning_rate":      log_uniform(rng, 0.01, 0.15),
        "max_leaf_nodes":     choose(rng, [15, 31, 47, 63, 95, 127]),
        "max_depth":          choose(rng, [None, 5, 7, 9, 12]),
        "min_samples_leaf":   choose(rng, [5, 10, 20, 40, 80]),
        "l2_regularization":  log_uniform(rng, 1e-4, 5.0),
        "max_bins":           choose(rng, [127, 255]),
        "early_stopping":     True,
        "validation_fraction": 0.15,
        "n_iter_no_change":   30,
        "random_state":       SEED,
    }
    if task != "gender":
        p["loss"] = "absolute_error"
    return p


def sample_et(rng, task):
    p = {
        "n_estimators":     choose(rng, [400, 600, 800, 1200, 1600]),
        "max_depth":        choose(rng, [None, 12, 18, 25, 35]),
        "min_samples_split": choose(rng, [2, 4, 8, 16]),
        "min_samples_leaf": choose(rng, [1, 2, 4, 8, 16]),
        "max_features":     choose(rng, ["sqrt", "log2", 0.3, 0.5, 0.8, 1.0]),
        "bootstrap":        bool(rng.integers(0, 2)),
        "n_jobs":           -1,
        "random_state":     SEED,
    }
    return p


SAMPLERS = {
    "lgb": sample_lgb,
    "xgb": sample_xgb,
    "cat": sample_cat,
    "hgb": sample_hgb,
    "et":  sample_et,
}


# ---------- Model builders ----------

def build_model(family, params, task):
    if family == "lgb":
        return (lgb.LGBMClassifier(**params) if task == "gender"
                else lgb.LGBMRegressor(**params))
    if family == "xgb":
        return (xgb.XGBClassifier(**params) if task == "gender"
                else xgb.XGBRegressor(**params))
    if family == "cat":
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
    """Fit model with per-family early stopping wiring."""
    if family == "lgb":
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(0)])
    elif family == "xgb":
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif family == "cat":
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    elif family == "hgb":
        # sklearn HGB has its own internal val split via early_stopping=True
        m.fit(X_tr, y_tr)
    elif family == "et":
        m.fit(X_tr, y_tr)
    else:
        raise ValueError(family)


# ---------- CV scoring ----------

def cv_oof(family, params, X, y, task):
    if task == "gender":
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(cv.split(X, y))
    else:
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(cv.split(X))

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

    if task == "gender":
        score = float(roc_auc_score(y, oof))
    else:
        score = float(mean_absolute_error(y, oof))
    return score, oof, fitted


# ---------- Random search per family/target ----------

def search(family, target, X, y, n_trials):
    rng = np.random.default_rng(SEED + abs(hash(family + target)) % 100000)
    sampler = SAMPLERS[family]

    log(f"--- {family.upper()} search on {target}: {n_trials} trials ---")
    better = (lambda new, best: new > best) if target == "gender" \
             else (lambda new, best: new < best)
    best_score = -np.inf if target == "gender" else np.inf
    best_params = None
    best_oof = None
    best_models = None
    trials = []

    t_start = time.time()
    metric = "AUC" if target == "gender" else "MAE"

    for t in range(n_trials):
        params = sampler(rng, target)
        t0 = time.time()
        try:
            score, oof, fitted = cv_oof(family, params, X, y, target)
        except Exception as exc:
            log(f"  trial {t+1:3d}/{n_trials}  FAILED: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t0
        mark = ""
        if better(score, best_score):
            best_score = score
            best_params = params
            best_oof = oof
            best_models = fitted
            mark = "  *"
        write_trial(family, target, t + 1, score, params, dt)
        trials.append((score, oof, params))
        elapsed = time.time() - t_start
        eta = elapsed / (t + 1) * (n_trials - t - 1)
        log(f"  trial {t+1:3d}/{n_trials}  {metric}={score:.4f}  "
            f"({dt:.1f}s, elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m){mark}")

    # Sort trials best -> worst
    trials.sort(key=lambda r: r[0], reverse=(target == "gender"))
    return best_score, best_params, best_oof, best_models, trials


def top_k_oofs(trials, k):
    return [t[1] for t in trials[:k]]


# ---------- Stacking ----------

def stack_gender(oofs, y):
    """LogisticRegressionCV over OOF probs from base models."""
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


# ---------- Importance dump for the best LGB on a target ----------

def lgb_importance(models, target):
    """Average gain importances across the 5 fold-fitted LGB models."""
    if not models:
        return None
    gains = []
    for m in models:
        booster = m.booster_
        gains.append(booster.feature_importance(importance_type="gain"))
    avg = np.mean(np.stack(gains, axis=0), axis=0)
    return avg.tolist()


# ---------- Main per-target driver ----------

def summary_path_for(target):
    return os.path.join(OUT_DIR, f"summary_{target}.json")


def best_params_path_for(target):
    return os.path.join(OUT_DIR, f"best_params_{target}.json")


def importance_path_for(target):
    return os.path.join(OUT_DIR, f"importance_{target}.json")


def run_target(name, X, y, cfg):
    log(f"\n############# TARGET: {name.upper()} #############")

    per_family = {}     # family -> (score, params, oof, models, trials)
    partial_params = {}
    for family in cfg.families:
        if family == "cat" and not HAVE_CATBOOST:
            log(f"--- skipping cat (catboost not installed) ---")
            continue
        score, params, oof, models, trials = search(
            family, name, X, y, cfg.trials_for(family),
        )
        per_family[family] = (score, params, oof, models, trials)

        # Incremental save: per-family OOF + accumulated best params.
        # If the run is killed mid-target, these stay on disk and can be
        # picked up by rescue_target.py.
        top_oofs = top_k_oofs(trials, cfg.top_for(family))
        per_fam_path = os.path.join(OUT_DIR, f"oof_{name}_{family}.npz")
        np.savez(per_fam_path,
                 best_oof=oof,
                 top_oofs=np.column_stack(top_oofs) if top_oofs else np.zeros((len(y), 0)))
        partial_params[family] = params
        with open(best_params_path_for(name), "w", encoding="utf-8") as fh:
            json.dump(partial_params, fh, indent=2, default=str)
        log(f"--- wrote partial: oof_{name}_{family}.npz "
            f"+ best_params_{name}.json ({len(per_family)} families done) ---")

    # Stack top-K OOFs from each completed family
    base_oofs = []
    base_labels = []
    for family, (_, _, _, _, trials) in per_family.items():
        k = cfg.top_for(family)
        oofs = top_k_oofs(trials, k)
        base_oofs.extend(oofs)
        base_labels.extend([f"{family}_top{i+1}" for i in range(len(oofs))])

    if name == "gender":
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
        for family, (sc, _, _, _, _) in per_family.items():
            summary[f"best_{family}_auc"] = sc

        np_kwargs = {"stacked": stacked, "threshold": np.array(thr, dtype=np.float32)}
        for family, (_, _, oof, _, _) in per_family.items():
            np_kwargs[family] = oof
        np.savez(os.path.join(OUT_DIR, "oof_gender.npz"), **np_kwargs)

    else:
        stacked = stack_reg(base_oofs, y)
        summary = {
            "stacked_mae": float(mean_absolute_error(y, stacked)),
            "stacked_r2":  float(r2_score(y, stacked)),
            "n_base_models_in_stack": len(base_oofs),
        }
        for family, (sc, _, _, _, _) in per_family.items():
            summary[f"best_{family}_mae"] = sc

        if name == "age":
            summary["stacked_bucket_acc"] = float(
                (age_bucket(stacked.round().clip(6, 68).astype(int))
                 == age_bucket(y)).mean()
            )

        np_kwargs = {"stacked": stacked}
        for family, (_, _, oof, _, _) in per_family.items():
            np_kwargs[family] = oof
        np.savez(os.path.join(OUT_DIR, f"oof_{name}.npz"), **np_kwargs)

    # Best params per family
    best_params_out = {
        family: per_family[family][1] for family in per_family
    }
    with open(best_params_path_for(name), "w", encoding="utf-8") as fh:
        json.dump(best_params_out, fh, indent=2, default=str)

    # Feature importances from best LGB if we ran it
    if "lgb" in per_family:
        imp = lgb_importance(per_family["lgb"][3], name)
        if imp is not None:
            with open(importance_path_for(name), "w", encoding="utf-8") as fh:
                json.dump({"gain": imp}, fh, indent=2)

    # Persist per-target summary (lets --skip-existing detect completed targets)
    with open(summary_path_for(name), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    log(f"--- {name.upper()} summary ---")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    return summary


# ---------- CLI entry ----------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=",".join(ALL_TARGETS),
                        help=f"Comma-separated subset of {{{','.join(ALL_TARGETS)}}}")
    parser.add_argument("--families", default=",".join(ALL_FAMILIES),
                        help=f"Comma-separated subset of {{{','.join(ALL_FAMILIES)}}}")
    parser.add_argument("--lgb-trials", type=int, default=120)
    parser.add_argument("--xgb-trials", type=int, default=100)
    parser.add_argument("--cat-trials", type=int, default=80)
    parser.add_argument("--hgb-trials", type=int, default=50)
    parser.add_argument("--et-trials",  type=int, default=40)
    parser.add_argument("--quick", action="store_true",
                        help="5 trials per family, for smoke testing.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip targets whose summary_<target>.json already exists.")
    return parser.parse_args()


def build_config(args):
    cfg = Config(
        lgb_trials=args.lgb_trials,
        xgb_trials=args.xgb_trials,
        cat_trials=args.cat_trials,
        hgb_trials=args.hgb_trials,
        et_trials=args.et_trials,
    )
    cfg.targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    cfg.families = [f.strip() for f in args.families.split(",") if f.strip()]
    if args.quick:
        cfg.lgb_trials = cfg.xgb_trials = cfg.cat_trials = 5
        cfg.hgb_trials = cfg.et_trials = 5
        cfg.top_lgb = cfg.top_xgb = cfg.top_cat = 2
        cfg.top_hgb = cfg.top_et = 2

    bad_t = [t for t in cfg.targets if t not in ALL_TARGETS]
    bad_f = [f for f in cfg.families if f not in ALL_FAMILIES]
    if bad_t:
        sys.exit(f"unknown targets: {bad_t}. valid: {ALL_TARGETS}")
    if bad_f:
        sys.exit(f"unknown families: {bad_f}. valid: {ALL_FAMILIES}")
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)

    # Reset log + trial CSV unless we are resuming
    if not args.skip_existing:
        open(LOG_PATH, "w", encoding="utf-8").close()
        if os.path.exists(TRIALS_CSV):
            os.remove(TRIALS_CSV)

    log(f"Started at {datetime.now().isoformat()}")
    log(f"Python {sys.version.split()[0]}, lightgbm {lgb.__version__}, "
        f"xgboost {xgb.__version__}"
        + (f", catboost installed" if HAVE_CATBOOST else ", catboost NOT installed"))
    log(f"Targets:  {cfg.targets}")
    log(f"Families: {cfg.families}")
    log(f"Trials:   lgb={cfg.lgb_trials} xgb={cfg.xgb_trials} "
        f"cat={cfg.cat_trials} hgb={cfg.hgb_trials} et={cfg.et_trials}")
    log(f"Top-K stack: lgb={cfg.top_lgb} xgb={cfg.top_xgb} cat={cfg.top_cat} "
        f"hgb={cfg.top_hgb} et={cfg.top_et}")
    log(f"Resumable mode: {args.skip_existing}")

    X, g, a, m = load()
    log(f"X: {X.shape}  gender boys/girls={int((g==0).sum())}/{int((g==1).sum())}  "
        f"age mean={a.mean():.1f}  mood mean={m.mean():.1f}")

    label_map = {"gender": g, "age": a, "mood": m}
    all_summaries = {}

    for tname in cfg.targets:
        if args.skip_existing and os.path.exists(summary_path_for(tname)):
            log(f"=== skipping {tname}: {summary_path_for(tname)} exists ===")
            with open(summary_path_for(tname), encoding="utf-8") as fh:
                all_summaries[tname] = json.load(fh)
            continue
        all_summaries[tname] = run_target(tname, X, label_map[tname], cfg)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(all_summaries, fh, indent=2)
    log(f"\nWrote combined summary to {SUMMARY_PATH}")
    log(f"Wrote trial table to {TRIALS_CSV}")
    log(f"Wrote OOF preds to {OUT_DIR}/oof_*.npz")
    log(f"Finished at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
