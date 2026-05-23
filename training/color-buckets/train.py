"""
Target-encoded colour bucket features on top of the 474-feature combined vector.

For each session we build TWO 512-dim bucket-delta vectors over an 8x8x8 RGB
grid (32-wide buckets, centres at 16, 48, ..., 240):

  * discrete_delta[i, b]  - each event assigned to one bucket (the one its RGB
    falls into).
  * smooth_delta[i, b]    - each event's value spread trilinearly across the 8
    bucket centres surrounding its RGB, weights summing to 1.

Event values follow the user's rule, designed so each session's vector sums to
0:
  - each of the 48 round-1 offered colours NOT picked: -0.1
  - each of the 21 picked colours (16 r1 + 4 r2 + 1 final): +0.1 * 16 * 3 / 21

Per-fold target encoding (no leakage):
  1. For each fold, compute per-bucket score grids using only the training
     fold's discrete deltas. For gender (binary):
         girly_grid[b]  = mean discrete_delta[i, b] over training girls
         masc_grid[b]   = mean discrete_delta[i, b] over training boys
     For age and mood (regression):
         age_grid[b]    = mean ((age - mean) * discrete_delta) across training
         mood_grid[b]   = mean ((mood - mean) * discrete_delta) across training
     (covariance-style scoring: positive grid weight = bucket presence
      predicts above-average target.)
  2. For every row (train and test, since the grid was built without them):
         girly_total / masc_total / signed_total  (for the gender head)
         age_total                                (for the age head)
         mood_total                               (for the mood head)
     All computed as smooth_delta @ grid (interpolated lookup).
  3. Concatenate to the 474-feature combined vector and train Single LightGBM.

After CV, save the FULL-data grids (girly, masc, age, mood) to
bucket_scores.npz so they can be reused as a cached lookup for any new session.
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

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, ".."))
EXTRA_DIR = os.path.join(DATA_DIR, "extra-features")
SOURCE = os.path.join(DATA_DIR, "raw", "save.ligma")

N_FOLDS = 5
SEED = 42

DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000
N_QUESTIONS = 21
N_R1 = 16

GRID = 8
N_BUCKETS = GRID ** 3   # 512
BUCKET_WIDTH = 256 / GRID   # 32
BUCKET_CENTER_OFFSET = BUCKET_WIDTH / 2   # 16

PICK_VALUE = 0.1 * 16 * 3 / N_QUESTIONS   # = 4.8 / 21 ~ 0.22857
NOT_PICK_VALUE = -0.1


def is_valid(row):
    try:
        if row[5] not in ("g", "j"):
            return False
        age = int(row[3])
        if not (6 <= age <= 68):
            return False
        if row[8] == "no data":
            return False
        if len(row[8]) < 4:
            return False
        if len(row[8][0]) < 64 or len(row[8][1]) < 16 or len(row[8][2]) < 4:
            return False
        if len(row[7]) < N_QUESTIONS:
            return False
        total = int(row[7][-1])
        if total < DURATION_MIN_MS or total > DURATION_MAX_MS:
            return False
        if not str(row[4]).lstrip("-").isdigit():
            return False
        return True
    except Exception:
        return False


def bucket_id(r_idx, g_idx, b_idx):
    return r_idx * GRID * GRID + g_idx * GRID + b_idx


def discrete_bucket(rgb):
    """Nearest-bucket assignment (RGB triple in 0..255 -> bucket index in 0..511)."""
    r = min(GRID - 1, rgb[0] // int(BUCKET_WIDTH))
    g = min(GRID - 1, rgb[1] // int(BUCKET_WIDTH))
    b = min(GRID - 1, rgb[2] // int(BUCKET_WIDTH))
    return bucket_id(r, g, b)


def trilinear_weights(rgb):
    """Return list of (bucket_index, weight) for trilinear interpolation.

    Maps the RGB triple to fractional bucket coordinates centred on the bucket
    centres (16, 48, ..., 240) so a colour exactly at a bucket centre gets all
    its weight on that single bucket, and a colour between two centres gets
    weight split linearly between them in each channel.
    """
    fr = max(0.0, min(GRID - 1, (rgb[0] - BUCKET_CENTER_OFFSET) / BUCKET_WIDTH))
    fg = max(0.0, min(GRID - 1, (rgb[1] - BUCKET_CENTER_OFFSET) / BUCKET_WIDTH))
    fb = max(0.0, min(GRID - 1, (rgb[2] - BUCKET_CENTER_OFFSET) / BUCKET_WIDTH))

    ir, ig, ib = int(fr), int(fg), int(fb)
    dr, dg, db = fr - ir, fg - ig, fb - ib

    out = []
    for offset_r, wr in ((0, 1.0 - dr), (1, dr)):
        if wr == 0:
            continue
        br = min(GRID - 1, ir + offset_r)
        for offset_g, wg in ((0, 1.0 - dg), (1, dg)):
            if wg == 0:
                continue
            bg = min(GRID - 1, ig + offset_g)
            for offset_b, wb in ((0, 1.0 - db), (1, db)):
                if wb == 0:
                    continue
                bb = min(GRID - 1, ib + offset_b)
                out.append((bucket_id(br, bg, bb), wr * wg * wb))
    return out


def compute_deltas(row):
    """Return (discrete_delta, smooth_delta), each shape (N_BUCKETS,)."""
    offered = row[8][0]
    r1 = row[8][1]
    r2 = row[8][2]
    final = row[8][3]
    valg = row[6]

    # which round-1 offered colours did the user pick?
    picked_offered_idx = set()
    for q in range(N_R1):
        try:
            idx = int(valg[q])
            if 0 <= idx <= 3:
                picked_offered_idx.add(q * 4 + idx)
        except (ValueError, IndexError):
            pass

    discrete = np.zeros(N_BUCKETS, dtype=np.float32)
    smooth   = np.zeros(N_BUCKETS, dtype=np.float32)

    def add_event(color, value):
        discrete[discrete_bucket(color)] += value
        for b, w in trilinear_weights(color):
            smooth[b] += value * w

    # - 0.1 for each round-1 offered colour that was NOT picked
    for i in range(64):
        if i not in picked_offered_idx:
            add_event(offered[i], NOT_PICK_VALUE)

    # + PICK_VALUE for each of the 21 pick events
    for c in r1:
        add_event(c, PICK_VALUE)
    for c in r2:
        add_event(c, PICK_VALUE)
    add_event(final, PICK_VALUE)

    return discrete, smooth


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


# Published baseline (Single LGB + 33 perceptual extras, no bucket features)
BASELINE = {"gender": 0.878, "age": 6.92, "mood": 8.87}


def main():
    t_total = time.time()

    # 1. Load existing static features (441 base + 33 perceptual)
    X_base = np.load(os.path.join(DATA_DIR, "features.npy"))
    X_extra = np.load(os.path.join(EXTRA_DIR, "features_extra.npy"))
    X_static = np.concatenate([X_base, X_extra], axis=1).astype(np.float32)
    targets = np.load(os.path.join(DATA_DIR, "targets.npz"))
    g, a, m = targets["gender"], targets["age"], targets["mood"]
    print(f"X_static: {X_static.shape}  (441 base + 33 perceptual extras)")
    print(f"gender boys/girls: {int((g == 0).sum())}/{int((g == 1).sum())}")

    # 2. Compute discrete + smooth delta vectors for every valid session
    print("\nComputing 512-bucket discrete + smooth delta vectors...")
    t0 = time.time()
    with open(SOURCE, encoding="utf-8") as fh:
        rows = json.load(fh)

    discrete_list = []
    smooth_list   = []
    for row in rows:
        if not is_valid(row):
            continue
        try:
            d, s = compute_deltas(row)
        except Exception as exc:
            print(f"  skip: {exc}")
            continue
        discrete_list.append(d)
        smooth_list.append(s)
    discrete = np.array(discrete_list, dtype=np.float32)
    smooth   = np.array(smooth_list,   dtype=np.float32)
    print(f"  discrete: {discrete.shape}, smooth: {smooth.shape}  ({time.time()-t0:.1f}s)")
    assert discrete.shape[0] == X_static.shape[0], "row count mismatch"

    # Sanity: per-session sums should be ~0 (sum to zero by construction)
    print(f"  per-session sum: discrete max |sum| = {np.abs(discrete.sum(axis=1)).max():.4f}, "
          f"smooth max |sum| = {np.abs(smooth.sum(axis=1)).max():.4f}")

    # 3. GENDER  - 5-fold CV with per-fold target encoding (no leakage)
    cv_clf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_clf = list(cv_clf.split(X_static, g))

    print("\n" + "=" * 70)
    print("GENDER  Single LightGBM + 3 gender-bucket features (477 total)")
    print("=" * 70)

    oof_g = np.zeros(len(g), dtype=np.float32)
    fold_aucs = []
    t0 = time.time()
    for fold_i, (tr, va) in enumerate(splits_clf):
        train_girls = (g[tr] == 1)
        train_boys  = (g[tr] == 0)
        girly_grid = discrete[tr][train_girls].mean(axis=0)
        masc_grid  = discrete[tr][train_boys].mean(axis=0)

        girly_total  = smooth @ girly_grid
        masc_total   = smooth @ masc_grid
        signed_total = girly_total - masc_total

        X_fold = np.concatenate([
            X_static,
            girly_total[:, None], masc_total[:, None], signed_total[:, None],
        ], axis=1)

        model = lgb_clf()
        model.fit(X_fold[tr], g[tr])
        oof_g[va] = model.predict_proba(X_fold[va])[:, 1]
        fold_auc = roc_auc_score(g[va], oof_g[va])
        fold_aucs.append(fold_auc)
        print(f"  fold {fold_i+1}/{N_FOLDS}  AUC={fold_auc:.4f}  (t+{time.time()-t0:.1f}s)")

    auc = roc_auc_score(g, oof_g)
    pred = (oof_g >= 0.5).astype(int)
    print(f"  combined+buckets AUC={auc:.4f}  acc={accuracy_score(g, pred):.4f}  "
          f"F1={f1_score(g, pred):.4f}")
    print(f"  baseline (no buckets)        AUC={BASELINE['gender']:.4f}")
    print(f"  delta:                       {auc - BASELINE['gender']:+.4f}")
    print(f"  fold AUC mean +/- std:       {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")

    # 4. AGE  - 5-fold CV with per-fold age-covariance grid encoding
    cv_reg = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_reg = list(cv_reg.split(X_static))

    print("\n" + "=" * 70)
    print("AGE  Single LightGBM + 1 age-bucket feature (475 total)")
    print("=" * 70)

    oof_a = np.zeros(len(a), dtype=np.float32)
    t0 = time.time()
    for fold_i, (tr, va) in enumerate(splits_reg):
        # age covariance grid: mean ((age - mean_age) * delta[b]) over training rows
        a_centered = (a[tr] - a[tr].mean()).astype(np.float32)
        age_grid = (a_centered[:, None] * discrete[tr]).mean(axis=0)
        age_total = smooth @ age_grid

        X_fold = np.concatenate([X_static, age_total[:, None]], axis=1)
        model = lgb_reg()
        model.fit(X_fold[tr], a[tr])
        oof_a[va] = model.predict(X_fold[va])
        mae_fold = mean_absolute_error(a[va], oof_a[va])
        print(f"  fold {fold_i+1}/{N_FOLDS}  MAE={mae_fold:.3f}  (t+{time.time()-t0:.1f}s)")

    mae_a = mean_absolute_error(a, oof_a)
    print(f"  combined+buckets MAE={mae_a:.3f}  R2={r2_score(a, oof_a):+.3f}")
    print(f"  baseline (no buckets) MAE={BASELINE['age']:.3f}")
    print(f"  delta:                {mae_a - BASELINE['age']:+.3f}  (negative is better)")

    # 5. MOOD  - 5-fold CV with per-fold mood-covariance grid encoding
    print("\n" + "=" * 70)
    print("MOOD  Single LightGBM + 1 mood-bucket feature (475 total)")
    print("=" * 70)

    oof_m = np.zeros(len(m), dtype=np.float32)
    t0 = time.time()
    for fold_i, (tr, va) in enumerate(splits_reg):
        m_centered = (m[tr] - m[tr].mean()).astype(np.float32)
        mood_grid = (m_centered[:, None] * discrete[tr]).mean(axis=0)
        mood_total = smooth @ mood_grid

        X_fold = np.concatenate([X_static, mood_total[:, None]], axis=1)
        model = lgb_reg()
        model.fit(X_fold[tr], m[tr])
        oof_m[va] = model.predict(X_fold[va])
        mae_fold = mean_absolute_error(m[va], oof_m[va])
        print(f"  fold {fold_i+1}/{N_FOLDS}  MAE={mae_fold:.3f}  (t+{time.time()-t0:.1f}s)")

    mae_m = mean_absolute_error(m, oof_m)
    print(f"  combined+buckets MAE={mae_m:.3f}  R2={r2_score(m, oof_m):+.3f}")
    print(f"  baseline (no buckets) MAE={BASELINE['mood']:.3f}")
    print(f"  delta:                {mae_m - BASELINE['mood']:+.3f}  (negative is better)")

    # 6. Final grids from ALL data (for the cached lookup) + feature importance on gender
    print("\nComputing final cached grids from all rows...")
    girly_grid_full = discrete[g == 1].mean(axis=0)
    masc_grid_full  = discrete[g == 0].mean(axis=0)
    age_grid_full   = ((a - a.mean()).astype(np.float32)[:, None] * discrete).mean(axis=0)
    mood_grid_full  = ((m - m.mean()).astype(np.float32)[:, None] * discrete).mean(axis=0)

    girly_total_full  = smooth @ girly_grid_full
    masc_total_full   = smooth @ masc_grid_full
    signed_total_full = girly_total_full - masc_total_full

    X_full = np.concatenate([
        X_static,
        girly_total_full[:, None], masc_total_full[:, None], signed_total_full[:, None],
    ], axis=1)
    full = lgb_clf()
    full.fit(X_full, g)
    importance = full.booster_.feature_importance(importance_type="gain")
    n_static = X_static.shape[1]
    order = np.argsort(importance)[::-1]
    rank = np.empty(len(importance), dtype=int)
    rank[order] = np.arange(len(importance))

    print("\nBucket feature importance (gender LGB on full data):")
    for offset, name in enumerate(["girly_total", "masc_total", "signed_total"]):
        i = n_static + offset
        print(f"  {name:14s}  gain={importance[i]:10.1f}  rank #{rank[i]+1} of {len(importance)}")

    girly_lead = girly_grid_full - masc_grid_full
    top_girl_buckets = np.argsort(girly_lead)[::-1][:5]
    top_boy_buckets  = np.argsort(girly_lead)[:5]

    def describe_bucket(b, lead_arr, label):
        r_idx = b // (GRID * GRID)
        g_idx = (b // GRID) % GRID
        b_idx = b % GRID
        rc = int(BUCKET_CENTER_OFFSET + r_idx * BUCKET_WIDTH)
        gc = int(BUCKET_CENTER_OFFSET + g_idx * BUCKET_WIDTH)
        bc = int(BUCKET_CENTER_OFFSET + b_idx * BUCKET_WIDTH)
        return f"RGB~({rc},{gc},{bc})  {label}={lead_arr[b]:+.4f}"

    print("\nBuckets that lean MOST girly (full-data lead):")
    for b in top_girl_buckets:
        print(f"  bucket {b:3d}: {describe_bucket(b, girly_lead, 'lead')}")
    print("Buckets that lean MOST masculine:")
    for b in top_boy_buckets:
        print(f"  bucket {b:3d}: {describe_bucket(b, girly_lead, 'lead')}")

    top_age_buckets = np.argsort(age_grid_full)[::-1][:5]
    bot_age_buckets = np.argsort(age_grid_full)[:5]
    print("\nBuckets whose presence MOST predicts above-average age:")
    for b in top_age_buckets:
        print(f"  bucket {b:3d}: {describe_bucket(b, age_grid_full, 'cov')}")
    print("Buckets whose presence MOST predicts below-average age:")
    for b in bot_age_buckets:
        print(f"  bucket {b:3d}: {describe_bucket(b, age_grid_full, 'cov')}")

    # 7. Save cached grids + summary
    np.savez(
        os.path.join(HERE, "bucket_scores.npz"),
        girly_grid=girly_grid_full.reshape(GRID, GRID, GRID),
        masc_grid=masc_grid_full.reshape(GRID, GRID, GRID),
        age_grid=age_grid_full.reshape(GRID, GRID, GRID),
        mood_grid=mood_grid_full.reshape(GRID, GRID, GRID),
        pick_value=np.float32(PICK_VALUE),
        not_pick_value=np.float32(NOT_PICK_VALUE),
        bucket_width=np.int32(int(BUCKET_WIDTH)),
        bucket_center_offset=np.int32(int(BUCKET_CENTER_OFFSET)),
    )

    summary = {
        "gender": {"with_buckets": float(auc),  "baseline": BASELINE["gender"],
                   "delta": float(auc - BASELINE["gender"]),  "fold_aucs": [float(x) for x in fold_aucs]},
        "age":    {"with_buckets": float(mae_a), "baseline": BASELINE["age"],
                   "delta": float(mae_a - BASELINE["age"])},
        "mood":   {"with_buckets": float(mae_m), "baseline": BASELINE["mood"],
                   "delta": float(mae_m - BASELINE["mood"])},
        "bucket_features_gender": {
            "girly_total":  {"gain": float(importance[n_static]),     "rank": int(rank[n_static]+1)},
            "masc_total":   {"gain": float(importance[n_static+1]),   "rank": int(rank[n_static+1]+1)},
            "signed_total": {"gain": float(importance[n_static+2]),   "rank": int(rank[n_static+2]+1)},
        },
    }
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nWrote bucket_scores.npz and summary.json")
    print(f"Total wall time: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
