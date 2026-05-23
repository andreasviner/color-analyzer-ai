"""
Fit the final LightGBM heads on ALL valid data and save them so the Cloudflare
inference function can load them at start-up.

Three artifacts are written to this folder:
    gender_model.txt   LightGBM binary classifier on 477 features
    age_model.txt      LightGBM regressor on 475 features
    mood_model.txt     LightGBM regressor on 475 features
    metadata.json      feature counts, target stats, training timestamp

Hyperparameters match color-buckets/train.py (the model the leaderboard names
"LightGBM + bucket scores").

Run:
    python train_final.py
"""

import json
import os
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.normpath(os.path.join(HERE, ".."))
EXTRA_DIR = os.path.join(TRAINING_DIR, "extra-features")
BUCKETS_DIR = os.path.join(TRAINING_DIR, "color-buckets")
RAW_SOURCE = os.path.join(TRAINING_DIR, "raw", "save.ligma")

SEED = 42
N_R1 = 16
N_QUESTIONS = 21
GRID = 8
N_BUCKETS = GRID ** 3
BUCKET_WIDTH = 256 / GRID
BUCKET_CENTER_OFFSET = BUCKET_WIDTH / 2
PICK_VALUE = 0.1 * 16 * 3 / N_QUESTIONS
NOT_PICK_VALUE = -0.1

DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000


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


def _bucket_id(r_idx, g_idx, b_idx):
    return r_idx * GRID * GRID + g_idx * GRID + b_idx


def _discrete_bucket(rgb):
    r = min(GRID - 1, int(rgb[0]) // int(BUCKET_WIDTH))
    g = min(GRID - 1, int(rgb[1]) // int(BUCKET_WIDTH))
    b = min(GRID - 1, int(rgb[2]) // int(BUCKET_WIDTH))
    return _bucket_id(r, g, b)


def _trilinear_weights(rgb):
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
                out.append((_bucket_id(br, bg, bb), wr * wg * wb))
    return out


def compute_deltas_for_row(row):
    offered = row[8][0]
    r1 = row[8][1]
    r2 = row[8][2]
    final = row[8][3]
    valg = row[6]

    picked = set()
    for q in range(N_R1):
        try:
            idx = int(valg[q])
            if 0 <= idx <= 3:
                picked.add(q * 4 + idx)
        except (ValueError, IndexError):
            pass

    discrete = np.zeros(N_BUCKETS, dtype=np.float32)
    smooth = np.zeros(N_BUCKETS, dtype=np.float32)

    def add_event(color, value):
        discrete[_discrete_bucket(color)] += value
        for b, w in _trilinear_weights(color):
            smooth[b] += value * w

    for i in range(64):
        if i not in picked:
            add_event(offered[i], NOT_PICK_VALUE)
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


def main():
    t_total = time.time()

    X_base = np.load(os.path.join(TRAINING_DIR, "features.npy"))
    X_extra = np.load(os.path.join(EXTRA_DIR, "features_extra.npy"))
    X_static = np.concatenate([X_base, X_extra], axis=1).astype(np.float32)
    targets = np.load(os.path.join(TRAINING_DIR, "targets.npz"))
    g = targets["gender"].astype(np.int8)
    a = targets["age"].astype(np.int32)
    m = targets["mood"].astype(np.int32)
    print(f"X_static {X_static.shape} | gender boys/girls {int((g == 0).sum())}/{int((g == 1).sum())}")

    print("Building bucket delta vectors from raw...")
    with open(RAW_SOURCE, encoding="utf-8") as fh:
        rows = json.load(fh)

    discrete_list = []
    smooth_list = []
    for row in rows:
        if not is_valid(row):
            continue
        d, s = compute_deltas_for_row(row)
        discrete_list.append(d)
        smooth_list.append(s)
    discrete = np.array(discrete_list, dtype=np.float32)
    smooth = np.array(smooth_list, dtype=np.float32)
    assert discrete.shape[0] == X_static.shape[0], f"row count mismatch: {discrete.shape[0]} vs {X_static.shape[0]}"

    print("Computing full-data bucket grids...")
    girly_grid_full = discrete[g == 1].mean(axis=0)
    masc_grid_full  = discrete[g == 0].mean(axis=0)
    age_grid_full   = ((a - a.mean()).astype(np.float32)[:, None] * discrete).mean(axis=0)
    mood_grid_full  = ((m - m.mean()).astype(np.float32)[:, None] * discrete).mean(axis=0)

    girly_total_full  = smooth @ girly_grid_full
    masc_total_full   = smooth @ masc_grid_full
    signed_total_full = girly_total_full - masc_total_full
    age_total_full    = smooth @ age_grid_full
    mood_total_full   = smooth @ mood_grid_full

    X_gender = np.concatenate([
        X_static,
        girly_total_full[:, None], masc_total_full[:, None], signed_total_full[:, None],
    ], axis=1)
    X_age = np.concatenate([X_static, age_total_full[:, None]], axis=1)
    X_mood = np.concatenate([X_static, mood_total_full[:, None]], axis=1)
    print(f"X_gender {X_gender.shape} | X_age {X_age.shape} | X_mood {X_mood.shape}")

    print("Fitting GENDER classifier on full data...")
    t0 = time.time()
    clf = lgb_clf()
    clf.fit(X_gender, g)
    print(f"  done in {time.time() - t0:.1f}s")
    clf.booster_.save_model(os.path.join(HERE, "gender_model.txt"))
    with open(os.path.join(HERE, "gender_model.pkl"), "wb") as fh:
        pickle.dump(clf, fh)

    print("Fitting AGE regressor on full data...")
    t0 = time.time()
    reg_a = lgb_reg()
    reg_a.fit(X_age, a)
    print(f"  done in {time.time() - t0:.1f}s")
    reg_a.booster_.save_model(os.path.join(HERE, "age_model.txt"))
    with open(os.path.join(HERE, "age_model.pkl"), "wb") as fh:
        pickle.dump(reg_a, fh)

    print("Fitting MOOD regressor on full data...")
    t0 = time.time()
    reg_m = lgb_reg()
    reg_m.fit(X_mood, m)
    print(f"  done in {time.time() - t0:.1f}s")
    reg_m.booster_.save_model(os.path.join(HERE, "mood_model.txt"))
    with open(os.path.join(HERE, "mood_model.pkl"), "wb") as fh:
        pickle.dump(reg_m, fh)

    # Refresh the bucket-scores cache so inference uses the same grids the
    # final models saw at fit time.
    np.savez(
        os.path.join(BUCKETS_DIR, "bucket_scores.npz"),
        girly_grid=girly_grid_full.reshape(GRID, GRID, GRID),
        masc_grid=masc_grid_full.reshape(GRID, GRID, GRID),
        age_grid=age_grid_full.reshape(GRID, GRID, GRID),
        mood_grid=mood_grid_full.reshape(GRID, GRID, GRID),
        pick_value=np.float32(PICK_VALUE),
        not_pick_value=np.float32(NOT_PICK_VALUE),
        bucket_width=np.int32(int(BUCKET_WIDTH)),
        bucket_center_offset=np.int32(int(BUCKET_CENTER_OFFSET)),
    )

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(X_static.shape[0]),
        "n_features": {
            "gender": int(X_gender.shape[1]),
            "age":    int(X_age.shape[1]),
            "mood":   int(X_mood.shape[1]),
        },
        "target_stats": {
            "gender_girl_frac": float((g == 1).mean()),
            "age_min": int(a.min()), "age_max": int(a.max()), "age_mean": float(a.mean()),
            "mood_min": int(m.min()), "mood_max": int(m.max()), "mood_mean": float(m.mean()),
        },
        "seed": SEED,
    }
    with open(os.path.join(HERE, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nSaved gender_model.txt, age_model.txt, mood_model.txt, metadata.json")
    print(f"Total wall time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
