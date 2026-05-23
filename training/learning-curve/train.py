"""
Learning curve for the gender target on the leaderboard champion model
(LightGBM + bucket scores, currently 0.881 OOF AUC).

Question this answers: are we limited by the model or by the dataset size?
If AUC keeps climbing as we add training rows, more data would push us
past 0.882. If AUC plateaus before 6,000 rows, more data won't help and
the ~0.88 ceiling is intrinsic to the signal at this feature resolution.

Setup:
  * Fixed validation set: a stratified-by-gender random 710-row hold-out
    drawn with sklearn's train_test_split (seed=42) - the EXACT same
    validation set the production trainer uses. Pinning to the same split
    means the curve's rightmost point matches the production leaderboard
    row directly.
  * Training pool: the other 6,000 sessions left after the val draw.
  * Train sizes: 500, 1000, 1500, ..., 6000 (12 points).
  * At each size we draw 5 random stratified-by-gender subsamples from
    the training pool, refit, and report mean / min / max AUC on the
    fixed validation set.
  * Model: LightGBM with the same config as color-buckets/train.py
    (n_estimators=800, num_leaves=63, lr=0.03, ...) plus the same 3 RGB
    bucket-encoded gender scores. Bucket grids are built ONLY from the
    current subsample (no leakage), then looked up on validation.

Outputs:
  learning_curve.json     mean/min/max AUC per train size + per-seed AUCs
  learning_curve.svg      plot of mean AUC with min/max band, dark theme
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, ".."))
EXTRA_DIR = os.path.join(DATA_DIR, "extra-features")
SOURCE = os.path.join(DATA_DIR, "raw", "save.ligma")
INFO_OUT = os.path.normpath(os.path.join(DATA_DIR, "..", "info", "learning_curve.svg"))

DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000
N_QUESTIONS = 21
N_R1 = 16

VAL_SIZE = 710
TRAIN_SIZES = list(range(500, 6001, 500))   # 500, 1000, ..., 6000
SEEDS = [42, 101, 202, 303, 404]
VAL_SPLIT_SEED = 42                          # MUST match lgb-production/train_and_emit.py
LEADERBOARD_REFERENCE = 0.881               # color-buckets OOF AUC


# ---------- Bucket helpers (8x8x8 RGB, same as color-buckets) ----------

GRID = 8
N_BUCKETS = GRID ** 3
BUCKET_WIDTH = 256 / GRID
BUCKET_CENTER_OFFSET = BUCKET_WIDTH / 2

PICK_VALUE = 0.1 * 16 * 3 / N_QUESTIONS
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


def bucket_id(r, g, b):
    return r * GRID * GRID + g * GRID + b


def discrete_bucket(rgb):
    r = min(GRID - 1, rgb[0] // int(BUCKET_WIDTH))
    g = min(GRID - 1, rgb[1] // int(BUCKET_WIDTH))
    b = min(GRID - 1, rgb[2] // int(BUCKET_WIDTH))
    return bucket_id(r, g, b)


def trilinear_weights(rgb):
    fr = max(0.0, min(GRID - 1, (rgb[0] - BUCKET_CENTER_OFFSET) / BUCKET_WIDTH))
    fg = max(0.0, min(GRID - 1, (rgb[1] - BUCKET_CENTER_OFFSET) / BUCKET_WIDTH))
    fb = max(0.0, min(GRID - 1, (rgb[2] - BUCKET_CENTER_OFFSET) / BUCKET_WIDTH))
    ir, ig, ib = int(fr), int(fg), int(fb)
    dr, dg, db = fr - ir, fg - ig, fb - ib
    out = []
    for ox, wx in ((0, 1.0 - dr), (1, dr)):
        if wx == 0: continue
        br = min(GRID - 1, ir + ox)
        for oy, wy in ((0, 1.0 - dg), (1, dg)):
            if wy == 0: continue
            bg_ = min(GRID - 1, ig + oy)
            for oz, wz in ((0, 1.0 - db), (1, db)):
                if wz == 0: continue
                bb = min(GRID - 1, ib + oz)
                out.append((bucket_id(br, bg_, bb), wx * wy * wz))
    return out


def compute_deltas(row):
    offered = row[8][0]
    r1 = row[8][1]
    r2 = row[8][2]
    final = row[8][3]
    valg = row[6]

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

    for i in range(64):
        if i not in picked_offered_idx:
            add_event(offered[i], NOT_PICK_VALUE)
    for c in r1:
        add_event(c, PICK_VALUE)
    for c in r2:
        add_event(c, PICK_VALUE)
    add_event(final, PICK_VALUE)

    return discrete, smooth


def lgb_clf(seed=42):
    return lgb.LGBMClassifier(
        n_estimators=800, num_leaves=63, learning_rate=0.03,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=20, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def main():
    t_total = time.time()

    print("Loading static features...")
    X_base  = np.load(os.path.join(DATA_DIR, "features.npy"))
    X_extra = np.load(os.path.join(EXTRA_DIR, "features_extra.npy"))
    X_static = np.concatenate([X_base, X_extra], axis=1).astype(np.float32)
    t = np.load(os.path.join(DATA_DIR, "targets.npz"))
    g = t["gender"]
    N = X_static.shape[0]
    print(f"  X_static: {X_static.shape}, gender balance: "
          f"{int((g==0).sum())}/{int((g==1).sum())} boys/girls "
          f"({g.mean():.3f} girls)")

    print("Computing per-session bucket vectors (discrete + smooth, RGB 8x8x8)...")
    t0 = time.time()
    with open(SOURCE, encoding="utf-8") as fh:
        rows = json.load(fh)

    discrete_list, smooth_list = [], []
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
    print(f"  shapes {discrete.shape}, {smooth.shape}  ({time.time()-t0:.1f}s)")
    assert discrete.shape[0] == N, f"row mismatch: bucket {discrete.shape[0]} vs static {N}"

    # Fixed stratified random split (matches production exactly: seed=42,
    # stratify on gender). The val set is the SAME 710 rows the production
    # trainer holds out, so the rightmost point on this curve is directly
    # comparable to the deployed model's headline number.
    all_idx = np.arange(N)
    pool_idx, val_idx = train_test_split(
        all_idx, test_size=VAL_SIZE, random_state=VAL_SPLIT_SEED, stratify=g,
    )
    pool_idx = np.sort(pool_idx)
    val_idx  = np.sort(val_idx)
    print(f"Validation set: stratified random, {VAL_SIZE} rows, "
          f"{g[val_idx].mean():.3f} girls")
    print(f"Training pool:  {len(pool_idx)} rows, "
          f"{g[pool_idx].mean():.3f} girls")

    X_val_static = X_static[val_idx]
    smooth_val   = smooth[val_idx]
    y_val        = g[val_idx]

    results = {n: {"per_seed_auc": {}, "mean": None, "min": None, "max": None}
               for n in TRAIN_SIZES}

    for n in TRAIN_SIZES:
        per_seed_aucs = []
        for seed in SEEDS:
            t_iter = time.time()
            if n >= len(pool_idx):
                sub_idx = pool_idx.copy()
            else:
                # stratified subsample of size n from the pool
                sub_idx, _ = train_test_split(
                    pool_idx, train_size=n,
                    random_state=seed, stratify=g[pool_idx],
                )
            sub_idx = np.sort(sub_idx)

            # Build bucket grids from this subsample only
            d_tr  = discrete[sub_idx]
            sm_tr = smooth[sub_idx]
            y_tr  = g[sub_idx]
            girly_grid = d_tr[y_tr == 1].mean(axis=0)
            masc_grid  = d_tr[y_tr == 0].mean(axis=0)

            girly_total_tr  = sm_tr @ girly_grid
            masc_total_tr   = sm_tr @ masc_grid
            signed_total_tr = girly_total_tr - masc_total_tr

            girly_total_val  = smooth_val @ girly_grid
            masc_total_val   = smooth_val @ masc_grid
            signed_total_val = girly_total_val - masc_total_val

            X_tr  = np.concatenate([
                X_static[sub_idx],
                girly_total_tr[:, None], masc_total_tr[:, None], signed_total_tr[:, None],
            ], axis=1)
            X_vl = np.concatenate([
                X_val_static,
                girly_total_val[:, None], masc_total_val[:, None], signed_total_val[:, None],
            ], axis=1)

            m = lgb_clf(seed=seed)
            m.fit(X_tr, y_tr)
            p_val = m.predict_proba(X_vl)[:, 1]
            auc = roc_auc_score(y_val, p_val)
            per_seed_aucs.append(float(auc))
            results[n]["per_seed_auc"][seed] = float(auc)
            print(f"  n={n:5d}  seed={seed}  AUC={auc:.4f}  "
                  f"({time.time()-t_iter:.1f}s)")

        results[n]["mean"] = float(np.mean(per_seed_aucs))
        results[n]["min"]  = float(np.min(per_seed_aucs))
        results[n]["max"]  = float(np.max(per_seed_aucs))
        print(f"  -- n={n:5d} mean={results[n]['mean']:.4f}  "
              f"[{results[n]['min']:.4f}, {results[n]['max']:.4f}]")

    summary = {
        "val_size": VAL_SIZE,
        "n_train_pool": int(len(pool_idx)),
        "train_sizes": TRAIN_SIZES,
        "seeds": SEEDS,
        "leaderboard_reference_oof_auc": LEADERBOARD_REFERENCE,
        "results": {str(k): v for k, v in results.items()},
        "wall_time_sec": float(time.time() - t_total),
    }
    with open(os.path.join(HERE, "learning_curve.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # ---- Plot (dark theme matching the portfolio site) ----
    bg     = "#0a0d18"
    text   = "#ebe6da"
    mute   = "#8a8fa3"
    hobby  = "#f5b95c"
    hair   = "#2a2f44"

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=144)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    xs    = np.array(TRAIN_SIZES, dtype=float)
    means = np.array([results[n]["mean"] for n in TRAIN_SIZES])
    mins  = np.array([results[n]["min"]  for n in TRAIN_SIZES])
    maxs  = np.array([results[n]["max"]  for n in TRAIN_SIZES])

    ax.fill_between(xs, mins, maxs, color=hobby, alpha=0.18, linewidth=0,
                    label=f"min / max across {len(SEEDS)} seeds")
    ax.plot(xs, means, color=hobby, linewidth=2.0, marker="o",
            markersize=5, markerfacecolor=hobby, markeredgecolor=bg,
            markeredgewidth=1.2, label="mean AUC")

    ax.axhline(LEADERBOARD_REFERENCE, color=text, linestyle=(0, (4, 4)),
               linewidth=1.0, alpha=0.55,
               label=f"leaderboard OOF AUC ({LEADERBOARD_REFERENCE:.3f})")
    ax.axhline(0.9, color=mute, linestyle=(0, (2, 4)),
               linewidth=0.9, alpha=0.7, label="0.900 target")

    ax.set_xlabel("training-pool size  (sessions)", color=text, fontsize=10)
    ax.set_ylabel("ROC-AUC on stratified 710-row random hold-out",
                  color=text, fontsize=10)
    ax.set_title("Gender AUC vs. training-set size",
                 color=text, fontsize=12, pad=12, loc="left",
                 fontstyle="italic")

    ax.tick_params(colors=mute, labelsize=9)
    for spine_name, spine in ax.spines.items():
        spine.set_color(hair)
        if spine_name in ("top", "right"):
            spine.set_visible(False)
    ax.grid(True, color=hair, linewidth=0.6, alpha=0.6)

    leg = ax.legend(loc="lower right", facecolor=bg, edgecolor=hair,
                    labelcolor=text, fontsize=9, framealpha=0.7)
    for t in leg.get_texts():
        t.set_color(text)

    plt.tight_layout()
    plt.savefig(INFO_OUT, format="svg",
                facecolor=bg, edgecolor="none", bbox_inches="tight")
    plt.close(fig)

    print(f"\nwrote learning_curve.json  ({os.path.join(HERE, 'learning_curve.json')})")
    print(f"wrote learning_curve.svg   ({INFO_OUT})")
    print(f"total wall time: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
