"""
Test whether perceptually-aware colour features (LAB, gender prototypes,
within-group difficulty) help on top of the existing 441 features.

For each valid session we compute 33 extra features in five blocks:

  A. Final pick in CIE L*a*b*                                       (3)
  B. r1 winners: L/a/b mean and std                                 (6)
  C. r2 winners: L/a/b mean and std                                 (6)
  D. Offered set: L/a/b mean and std                                (6)
  E. Gender-prototype distances (DeltaE in LAB)                     (5)
  F. Per-question offered-group difficulty + relative decisiveness  (4)
  G. Time anomalies (rushed / dwelled / CV)                         (3)

Concatenate these to the existing features.npy (441 cols) and train Single
LightGBM (same config as baselines/train.py) with the same 5-fold CV.
Compare to the published baseline numbers (0.876 / 6.95 / 8.85).
"""

import json
import os
import time

import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, ".."))
SOURCE = os.path.join(DATA_DIR, "raw", "save.ligma")

# Shared validity + troll filter: must match features.py exactly so X_extra
# lines up row-for-row with features.npy.
sys.path.insert(0, DATA_DIR)
from data_cleaning import is_valid_clean  # noqa: E402

N_FOLDS = 5
SEED = 42

DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000
N_QUESTIONS = 21
N_R1 = 16
N_R2 = 4

# Published Single LightGBM baselines from baselines/train.py
BASELINE = {"gender": 0.876, "age": 6.95, "mood": 8.85}


# ---------- sRGB -> CIE L*a*b* (D65) ----------

def _gamma(c):
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92


def srgb_to_lab(rgb):
    r, g, b = (_gamma(x / 255.0) for x in rgb)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


# ---------- Gender prototype colours (mean LAB) ----------

GIRL_PROTOTYPES_RGB = [
    (255, 182, 193),  # light pink
    (255, 105, 180),  # hot pink
    (200, 100, 200),  # light purple
    (220,  40,  40),  # red
]
BOY_PROTOTYPES_RGB = [
    ( 40,  60, 220),  # bright blue
    ( 60, 130, 220),  # sky blue
    ( 50, 170,  60),  # green
    ( 60,  60,  60),  # dark gray
]
GIRL_PROTO = np.mean([srgb_to_lab(c) for c in GIRL_PROTOTYPES_RGB], axis=0)
BOY_PROTO  = np.mean([srgb_to_lab(c) for c in BOY_PROTOTYPES_RGB],  axis=0)


# ---------- Validation (identical to features.py) ----------

def is_valid(row):
    # Shared filter — must match features.py exactly so X_extra lines up
    # row-for-row with features.npy.
    return is_valid_clean(row)


# ---------- Extra feature extraction ----------

def extract_extra(row):
    valg = row[6]
    tider = [int(x) for x in row[7]]
    offered = row[8][0]
    r1 = row[8][1]
    r2 = row[8][2]
    final = row[8][3]

    names = []
    vals = []

    def push(name, value):
        names.append(name)
        vals.append(float(value))

    # Convert once to LAB
    offered_lab = np.array([srgb_to_lab(c) for c in offered], dtype=np.float32)
    r1_lab      = np.array([srgb_to_lab(c) for c in r1],      dtype=np.float32)
    r2_lab      = np.array([srgb_to_lab(c) for c in r2],      dtype=np.float32)
    final_lab   = np.array(srgb_to_lab(final), dtype=np.float32)

    # Block A: final pick in LAB
    push("final_lab_L", final_lab[0])
    push("final_lab_a", final_lab[1])
    push("final_lab_b", final_lab[2])

    # Block B: r1 LAB stats
    push("r1_lab_L_mean", r1_lab[:, 0].mean())
    push("r1_lab_a_mean", r1_lab[:, 1].mean())
    push("r1_lab_b_mean", r1_lab[:, 2].mean())
    push("r1_lab_L_std",  r1_lab[:, 0].std())
    push("r1_lab_a_std",  r1_lab[:, 1].std())
    push("r1_lab_b_std",  r1_lab[:, 2].std())

    # Block C: r2 LAB stats
    push("r2_lab_L_mean", r2_lab[:, 0].mean())
    push("r2_lab_a_mean", r2_lab[:, 1].mean())
    push("r2_lab_b_mean", r2_lab[:, 2].mean())
    push("r2_lab_L_std",  r2_lab[:, 0].std())
    push("r2_lab_a_std",  r2_lab[:, 1].std())
    push("r2_lab_b_std",  r2_lab[:, 2].std())

    # Block D: offered LAB stats
    push("off_lab_L_mean", offered_lab[:, 0].mean())
    push("off_lab_a_mean", offered_lab[:, 1].mean())
    push("off_lab_b_mean", offered_lab[:, 2].mean())
    push("off_lab_L_std",  offered_lab[:, 0].std())
    push("off_lab_a_std",  offered_lab[:, 1].std())
    push("off_lab_b_std",  offered_lab[:, 2].std())

    # Block E: gender-prototype distances in LAB (DeltaE)
    d_final_girl = float(np.linalg.norm(final_lab - GIRL_PROTO))
    d_final_boy  = float(np.linalg.norm(final_lab - BOY_PROTO))
    push("final_to_girl_proto", d_final_girl)
    push("final_to_boy_proto",  d_final_boy)
    push("final_proto_log_ratio",
         float(np.log((d_final_boy + 1.0) / (d_final_girl + 1.0))))
    push("r1_to_girl_proto_mean",
         float(np.mean(np.linalg.norm(r1_lab - GIRL_PROTO, axis=1))))
    push("r1_to_boy_proto_mean",
         float(np.mean(np.linalg.norm(r1_lab - BOY_PROTO,  axis=1))))

    # Block F: per-question difficulty and relative decisiveness
    diversities = []
    relative_decisives = []
    for q in range(N_R1):
        group = offered_lab[q * 4:(q + 1) * 4]  # (4, 3)
        # mean pairwise LAB distance within the offered group
        diff = group[:, None, :] - group[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        triu = dists[np.triu_indices(4, 1)]
        diversity = float(triu.mean())
        diversities.append(diversity)

        try:
            idx = int(valg[q])
            if not (0 <= idx <= 3):
                idx = 0
        except (ValueError, IndexError):
            idx = 0
        chosen   = group[idx]
        rejected = group[np.arange(4) != idx]
        delta    = float(np.linalg.norm(chosen - rejected.mean(axis=0)))
        relative_decisives.append(delta / (diversity + 1.0))

    push("r1_offered_diversity_mean",  float(np.mean(diversities)))
    push("r1_offered_diversity_std",   float(np.std(diversities)))
    push("r1_relative_decisive_mean",  float(np.mean(relative_decisives)))
    push("r1_relative_decisive_std",   float(np.std(relative_decisives)))

    # Block G: time anomalies
    deltas = [max(0, tider[0])] + [
        max(0, tider[i] - tider[i - 1]) for i in range(1, len(tider))
    ]
    deltas_sec = [d / 1000.0 for d in deltas]
    push("time_rushed_frac",  sum(1 for t in deltas_sec if t < 1.0) / len(deltas_sec))
    push("time_dwelled_frac", sum(1 for t in deltas_sec if t > 7.0) / len(deltas_sec))
    push("time_cv",           float(np.std(deltas) / (np.mean(deltas) + 1.0)))

    return names, vals


# ---------- LightGBM builders matching baselines/train.py ----------

def lgb_clf():
    return lgb.LGBMClassifier(
        n_estimators=800, num_leaves=63, learning_rate=0.03,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=20, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )


def lgb_reg():
    return lgb.LGBMRegressor(
        n_estimators=1000, num_leaves=63, learning_rate=0.03,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=20, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )


# ---------- CV runners ----------

def cv_clf(X, y):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    t0 = time.time()
    for tr, va in cv.split(X, y):
        m = lgb_clf()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof, time.time() - t0


def cv_reg(X, y):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    t0 = time.time()
    for tr, va in cv.split(X):
        m = lgb_reg()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    return oof, time.time() - t0


# ---------- Main ----------

def main():
    # 1. Load existing data
    X_orig = np.load(os.path.join(DATA_DIR, "features.npy"))
    targets = np.load(os.path.join(DATA_DIR, "targets.npz"))
    g, a, m = targets["gender"], targets["age"], targets["mood"]
    with open(os.path.join(DATA_DIR, "feature_names.json"), encoding="utf-8") as fh:
        orig_names = json.load(fh)
    print(f"X_orig: {X_orig.shape}  (existing 441-feature vector)")

    # 2. Compute extra features
    print("Computing extra features...")
    with open(SOURCE, "r", encoding="utf-8") as fh:
        rows = json.load(fh)

    X_extra = []
    extra_names = None
    for row in rows:
        if not is_valid(row):
            continue
        try:
            names, vals = extract_extra(row)
        except Exception as exc:
            print(f"  skip: {exc}")
            continue
        if extra_names is None:
            extra_names = names
        elif names != extra_names:
            print("  feature order drift, skipping row")
            continue
        X_extra.append(vals)

    X_extra = np.array(X_extra, dtype=np.float32)
    print(f"X_extra: {X_extra.shape}  ({X_extra.shape[1]} new features)")
    assert X_orig.shape[0] == X_extra.shape[0], "row count mismatch!"

    X_combined = np.concatenate([X_orig, X_extra], axis=1)
    print(f"X_combined: {X_combined.shape}\n")

    np.save(os.path.join(HERE, "features_extra.npy"), X_extra)
    with open(os.path.join(HERE, "feature_names_extra.json"), "w", encoding="utf-8") as fh:
        json.dump(extra_names, fh, indent=2)

    # 3. Train Single LightGBM on combined feature set, all three targets
    print("=" * 70)
    print("Single LightGBM on combined feature set (5-fold CV, seed 42)")
    print("=" * 70)

    # Gender
    print("\nGENDER")
    oof_g, dt = cv_clf(X_combined, g)
    auc_g = roc_auc_score(g, oof_g)
    pred_g = (oof_g >= 0.5).astype(int)
    print(f"  combined ({X_combined.shape[1]} features): AUC={auc_g:.4f}  "
          f"acc={accuracy_score(g, pred_g):.4f}  "
          f"F1={f1_score(g, pred_g):.4f}  ({dt:.1f}s)")
    print(f"  baseline (441 features):              AUC={BASELINE['gender']:.4f}")
    print(f"  delta:                                 {auc_g - BASELINE['gender']:+.4f}")

    # Age
    print("\nAGE")
    oof_a, dt = cv_reg(X_combined, a)
    mae_a = mean_absolute_error(a, oof_a)
    r2_a  = r2_score(a, oof_a)
    print(f"  combined ({X_combined.shape[1]} features): MAE={mae_a:.3f}  "
          f"R2={r2_a:+.3f}  ({dt:.1f}s)")
    print(f"  baseline (441 features):              MAE={BASELINE['age']:.3f}")
    print(f"  delta:                                 {mae_a - BASELINE['age']:+.3f}  (negative is better)")

    # Mood
    print("\nMOOD")
    oof_m, dt = cv_reg(X_combined, m)
    mae_m = mean_absolute_error(m, oof_m)
    r2_m  = r2_score(m, oof_m)
    print(f"  combined ({X_combined.shape[1]} features): MAE={mae_m:.3f}  "
          f"R2={r2_m:+.3f}  ({dt:.1f}s)")
    print(f"  baseline (441 features):              MAE={BASELINE['mood']:.3f}")
    print(f"  delta:                                 {mae_m - BASELINE['mood']:+.3f}  (negative is better)")

    # 4. Feature importance from a full fit on gender (the headline metric)
    print("\n" + "=" * 70)
    print("Feature importance on gender (full-data LGB, gain-based)")
    print("=" * 70)
    full = lgb_clf()
    full.fit(X_combined, g)
    importance = full.booster_.feature_importance(importance_type="gain")
    all_names = orig_names + extra_names
    order = np.argsort(importance)[::-1]

    print("Top 30 features by gain (* = new feature):")
    print(f"  {'rank':>4}  {'feature':40s}  {'gain':>10s}")
    for rank, i in enumerate(order[:30]):
        marker = "*" if i >= len(orig_names) else " "
        print(f"  {rank+1:4d}  {marker} {all_names[i]:38s}  {importance[i]:10.1f}")

    print(f"\nAll {len(extra_names)} new feature ranks (out of {len(all_names)}):")
    rank = np.empty(len(all_names), dtype=int)
    rank[order] = np.arange(len(all_names))
    extra_with_rank = [
        (rank[i], all_names[i], importance[i])
        for i in range(len(orig_names), len(all_names))
    ]
    extra_with_rank.sort()
    print(f"  {'rank':>5}  {'feature':38s}  {'gain':>10s}")
    for r, n, g_v in extra_with_rank:
        print(f"  {r+1:5d}  {n:38s}  {g_v:10.1f}")

    # 5. Save summary
    summary = {
        "n_features_combined": int(X_combined.shape[1]),
        "n_features_extra":    int(X_extra.shape[1]),
        "gender": {"combined_auc": float(auc_g), "baseline_auc": BASELINE["gender"]},
        "age":    {"combined_mae": float(mae_a), "baseline_mae": BASELINE["age"]},
        "mood":   {"combined_mae": float(mae_m), "baseline_mae": BASELINE["mood"]},
        "top_new_features": [
            {"rank": int(r + 1), "name": n, "gain": float(g_v)}
            for r, n, g_v in extra_with_rank[:10]
        ],
    }
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote summary.json")


if __name__ == "__main__":
    main()
