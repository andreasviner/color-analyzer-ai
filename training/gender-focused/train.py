"""
Gender-only model — focused pass.

Previous attempts:
  v1: 8-seed LGB bag on the 477 color-buckets feature set -> 0.8824 AUC
  v2: + 25 new features incl. a direct logistic-on-bucket prob -> 0.8660
      (regressed because the LR feature was leaky on training rows)
  v3: dropped leaky feature + noisy multi-stage stages -> 0.8779
      (still below v1; extra features were diluting LGB's attention)

This v4 keeps only the additions that add genuinely orthogonal signal and
that pass a "would-color-buckets-also-do-this" sanity check:

    A. rgb_all signed totals (3, replicates color-buckets champion).

    B. LAB 8x8x8 signed totals (3, NEW).
       Same construction as A but in CIE L*a*b* space, so the pink region
       (high a*, moderate b*) gets dedicated buckets independent of how the
       RGB grid happened to slice it. Adds perceptually-uniform resolution
       in the part of color space that carries gender signal.

    C. Per-question gender-choice features (4, NEW).
       For each of the 16 round-1 questions, look up the per-fold signed
       (girly - masc) grid value at all four offered colours. Aggregate
       across the 16 questions as:
         pq_chosen_lead_mean   mean signed-lead of the picked colour
         pq_chosen_lead_std    std (decisiveness across questions)
         pq_picked_most_girly  fraction where chosen was the girliest of 4
         pq_picked_least_girly fraction where chosen was the least girly of 4
       This is the most genuinely orthogonal block — the totals capture
       "where on the gender colour map did the session's picks land", these
       capture "given a four-way choice, did the picker actually reach for
       the girlier option each time", which is a different signal.

  Net feature count: 474 static + 3 + 3 + 4 = 484 features.

Models:
    1. LightGBM bag (8 seeds)  — same as v1's headline source
    2. XGBoost  bag (5 seeds)  — different decision surface
    3. CatBoost bag (3 seeds)  — third diverse tree family
    4. Logistic meta-stacker over the three bag OOFs (per fold),
       L2 with C swept across {0.1, 1.0, 10.0}, picked by inner-train AUC.

CV: 5-fold stratified on gender, seed 42, same scheme as every other model.

Outputs:
  gender_oof.npz        OOF probs for each model, each bag, and the stack
  summary.json          AUC numbers, per-fold AUC, hyperparams used
  feature_names.json    names of all new features added on top of 474 static
"""

import json
import os
import time

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from catboost import CatBoostClassifier
    HAVE_CATBOOST = True
except ImportError:
    HAVE_CATBOOST = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, ".."))
EXTRA_DIR = os.path.join(DATA_DIR, "extra-features")
SOURCE = os.path.join(DATA_DIR, "raw", "save.ligma")

N_FOLDS = 5
SEED = 42
LGB_BAG_SEEDS = [42, 101, 202, 303, 404, 505, 606, 707]
XGB_BAG_SEEDS = [42, 101, 202, 303, 404]
CAT_BAG_SEEDS = [42, 101, 202]

DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000
N_QUESTIONS = 21
N_R1 = 16
N_R2 = 4

# Existing leaderboard champions (for reference printing)
BASELINE_COLOR_BUCKETS = 0.8809
PREV_LGB_BAG_AUC       = 0.8824


# ---------- Validation ----------

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


# ---------- sRGB -> LAB ----------

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


# ---------- Bucket helpers ----------

GRID = 8
N_BUCKETS = GRID ** 3
RGB_WIDTH = 256 / GRID
RGB_CENTER = RGB_WIDTH / 2

LAB_LO = np.array([0.0, -100.0, -100.0], dtype=np.float32)
LAB_HI = np.array([100.0, 100.0, 100.0], dtype=np.float32)
LAB_W  = (LAB_HI - LAB_LO) / GRID


def bid(r, g, b):
    return r * GRID * GRID + g * GRID + b


def rgb_discrete(rgb):
    r = min(GRID - 1, rgb[0] // int(RGB_WIDTH))
    g = min(GRID - 1, rgb[1] // int(RGB_WIDTH))
    b = min(GRID - 1, rgb[2] // int(RGB_WIDTH))
    return bid(r, g, b)


def rgb_trilinear(rgb):
    fr = max(0.0, min(GRID - 1, (rgb[0] - RGB_CENTER) / RGB_WIDTH))
    fg = max(0.0, min(GRID - 1, (rgb[1] - RGB_CENTER) / RGB_WIDTH))
    fb = max(0.0, min(GRID - 1, (rgb[2] - RGB_CENTER) / RGB_WIDTH))
    ir, ig, ib = int(fr), int(fg), int(fb)
    dr, dg, db = fr - ir, fg - ig, fb - ib
    out = []
    for ox, wx in ((0, 1.0 - dr), (1, dr)):
        if wx == 0: continue
        br = min(GRID - 1, ir + ox)
        for oy, wy in ((0, 1.0 - dg), (1, dg)):
            if wy == 0: continue
            bg = min(GRID - 1, ig + oy)
            for oz, wz in ((0, 1.0 - db), (1, db)):
                if wz == 0: continue
                bb = min(GRID - 1, ib + oz)
                out.append((bid(br, bg, bb), wx * wy * wz))
    return out


def lab_discrete(lab):
    idx = np.clip(((np.asarray(lab, dtype=np.float32) - LAB_LO) / LAB_W).astype(int),
                  0, GRID - 1)
    return int(idx[0]) * GRID * GRID + int(idx[1]) * GRID + int(idx[2])


def lab_trilinear(lab):
    frac = (np.asarray(lab, dtype=np.float32) - LAB_LO) / LAB_W - 0.5
    frac = np.clip(frac, 0.0, GRID - 1)
    i = frac.astype(int)
    d = frac - i
    out = []
    for ox, wx in ((0, 1.0 - d[0]), (1, d[0])):
        if wx == 0: continue
        bx = min(GRID - 1, i[0] + ox)
        for oy, wy in ((0, 1.0 - d[1]), (1, d[1])):
            if wy == 0: continue
            by = min(GRID - 1, i[1] + oy)
            for oz, wz in ((0, 1.0 - d[2]), (1, d[2])):
                if wz == 0: continue
                bz = min(GRID - 1, i[2] + oz)
                out.append((bid(bx, by, bz), wx * wy * wz))
    return out


PICK_VALUE = 0.1 * 16 * 3 / N_QUESTIONS
NOT_PICK_VALUE = -0.1


def session_bucket_data(row):
    """
    For one session, return a dict of (discrete, smooth) bucket-vector pairs
    for each stage and space. Discrete is used to BUILD per-fold grids;
    smooth is used to LOOK UP scores against grids (trilinear weighting).

    Stages we split out:
        rgb_all     — full RGB grid (combines picks + non-picks, like color-buckets)
        rgb_final   — final pick only
        rgb_r2      — round-2 winners (4 picks)
        rgb_r1      — round-1 winners (16 picks)
        rgb_nopick  — non-picked offered colours

    Plus the LAB equivalent of the "all" combination:
        lab_all
    """
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

    def zeros2():
        return (np.zeros(N_BUCKETS, dtype=np.float32),
                np.zeros(N_BUCKETS, dtype=np.float32))

    rgb_all_d, rgb_all_s = zeros2()
    rgb_final_d, rgb_final_s = zeros2()
    rgb_r2_d, rgb_r2_s = zeros2()
    rgb_r1_d, rgb_r1_s = zeros2()
    rgb_nopick_d, rgb_nopick_s = zeros2()
    lab_all_d, lab_all_s = zeros2()

    def add_rgb(d, s, c, value):
        d[rgb_discrete(c)] += value
        for b, w in rgb_trilinear(c):
            s[b] += value * w

    def add_lab(d, s, c, value):
        lab = srgb_to_lab(c)
        d[lab_discrete(lab)] += value
        for b, w in lab_trilinear(lab):
            s[b] += value * w

    # r1 picks
    for c in r1:
        add_rgb(rgb_all_d, rgb_all_s, c, PICK_VALUE)
        add_rgb(rgb_r1_d,  rgb_r1_s,  c, 1.0)            # weight=1 for stage-only grids
        add_lab(lab_all_d, lab_all_s, c, PICK_VALUE)

    # r2 picks
    for c in r2:
        add_rgb(rgb_all_d, rgb_all_s, c, PICK_VALUE)
        add_rgb(rgb_r2_d,  rgb_r2_s,  c, 1.0)
        add_lab(lab_all_d, lab_all_s, c, PICK_VALUE)

    # final pick
    add_rgb(rgb_all_d, rgb_all_s, final, PICK_VALUE)
    add_rgb(rgb_final_d, rgb_final_s, final, 1.0)
    add_lab(lab_all_d, lab_all_s, final, PICK_VALUE)

    # non-picks
    for i in range(64):
        if i in picked_offered_idx:
            continue
        c = offered[i]
        add_rgb(rgb_all_d, rgb_all_s, c, NOT_PICK_VALUE)
        add_rgb(rgb_nopick_d, rgb_nopick_s, c, 1.0)
        add_lab(lab_all_d, lab_all_s, c, NOT_PICK_VALUE)

    return {
        "rgb_all":    (rgb_all_d,    rgb_all_s),
        "rgb_final":  (rgb_final_d,  rgb_final_s),
        "rgb_r2":     (rgb_r2_d,     rgb_r2_s),
        "rgb_r1":     (rgb_r1_d,     rgb_r1_s),
        "rgb_nopick": (rgb_nopick_d, rgb_nopick_s),
        "lab_all":    (lab_all_d,    lab_all_s),
    }


# ---------- Per-question gender choice features (vectorised) ----------
#
# We precompute, per session, the trilinear (bucket, weight) decomposition of
# each of the 64 offered round-1 colours so the per-fold computation reduces
# to a numpy gather + weighted sum. Plus the chosen-index (0..3) per question.

def precompute_offered_lookup(rows):
    """
    Returns
      offered_idx:  (N, 64, 8) int16  — bucket indices for trilinear weights
      offered_w:    (N, 64, 8) float32 — weights (zero-padded if fewer than 8)
      chosen_idx:   (N, 16) int8 — which of 4 in each group was chosen (0..3)
    """
    N = len(rows)
    offered_idx = np.zeros((N, 64, 8), dtype=np.int16)
    offered_w   = np.zeros((N, 64, 8), dtype=np.float32)
    chosen_idx  = np.zeros((N, N_R1), dtype=np.int8)
    for i, row in enumerate(rows):
        offered = row[8][0]
        valg = row[6]
        for c_i in range(64):
            tri = rgb_trilinear(offered[c_i])
            for k, (b, w) in enumerate(tri[:8]):
                offered_idx[i, c_i, k] = b
                offered_w[i, c_i, k] = w
        for q in range(N_R1):
            try:
                idx = int(valg[q])
                if not (0 <= idx <= 3):
                    idx = 0
            except (ValueError, IndexError):
                idx = 0
            chosen_idx[i, q] = idx
    return offered_idx, offered_w, chosen_idx


def per_question_features_all(signed_grid, offered_idx, offered_w, chosen_idx):
    """
    Compute the 4 per-question gender-choice features for ALL sessions at once,
    given the per-fold signed_grid (N_BUCKETS,) and the precomputed lookup
    tensors from precompute_offered_lookup.

    Returns (N, 4) array.
    """
    N = offered_idx.shape[0]
    # gather: signed_grid[offered_idx] -> (N, 64, 8); * weights, sum on axis 2
    gathered = signed_grid[offered_idx]                # (N, 64, 8)
    lookups = (gathered * offered_w).sum(axis=2)       # (N, 64)
    groups = lookups.reshape(N, N_R1, 4)               # (N, 16, 4)

    chosen_leads = groups[np.arange(N)[:, None],
                          np.arange(N_R1)[None, :],
                          chosen_idx]                  # (N, 16)
    mean_lead = chosen_leads.mean(axis=1)
    std_lead  = chosen_leads.std(axis=1)
    most_g = (groups.argmax(axis=2) == chosen_idx).mean(axis=1)
    least_g = (groups.argmin(axis=2) == chosen_idx).mean(axis=1)
    return np.column_stack([mean_lead, std_lead, most_g, least_g]).astype(np.float32)


# ---------- Hue x time interactions ----------

def hue_time_features(row):
    """
    2 features built once per session (target-independent, so they can live
    outside the per-fold loop):

      warm_minus_cool_dwell:  mean dwell ms on picks with R > B minus same for
                              picks with R < B (positive = lingers on warm)
      light_minus_dark_dwell: same split, on (R+G+B) above vs below median
    """
    r1 = row[8][1]
    r2 = row[8][2]
    final = row[8][3]
    tider = [int(x) for x in row[7]]
    deltas = [max(0, tider[0])] + [
        max(0, tider[i] - tider[i - 1]) for i in range(1, len(tider))
    ]

    picks = list(r1) + list(r2) + [final]
    times = deltas[:len(picks)]

    warm = [t for c, t in zip(picks, times) if c[0] > c[2]]
    cool = [t for c, t in zip(picks, times) if c[0] < c[2]]
    warm_minus_cool = (np.mean(warm) - np.mean(cool)) if warm and cool else 0.0

    lights = [sum(c) for c in picks]
    med = float(np.median(lights))
    bright = [t for c, t in zip(picks, times) if sum(c) >= med]
    dark   = [t for c, t in zip(picks, times) if sum(c) <  med]
    light_minus_dark = (np.mean(bright) - np.mean(dark)) if bright and dark else 0.0

    return [float(warm_minus_cool), float(light_minus_dark)]


# ---------- Models ----------

def lgb_clf(seed):
    return lgb.LGBMClassifier(
        n_estimators=800, num_leaves=63, learning_rate=0.03,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=20, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def xgb_clf(seed):
    return xgb.XGBClassifier(
        n_estimators=900, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, eval_metric="logloss",
        tree_method="hist", verbosity=0,
    )


def cat_clf(seed):
    return CatBoostClassifier(
        iterations=900, depth=6, learning_rate=0.03,
        l2_leaf_reg=3.0, random_seed=seed,
        verbose=False, allow_writing_files=False,
        loss_function="Logloss", eval_metric="AUC",
    )


# ---------- Logistic stacker ----------

def logistic_stack(oof_lgb, oof_xgb, oof_cat, y, splits):
    """
    Per-fold logistic stack: for each test fold, fit LR on the OOF probs of the
    OTHER folds (the training folds' rows have OOF probs filled in by the
    per-fold base models, but those WERE used to build the meta-features for
    the test fold so we re-train the meta-learner per fold using only the
    rows whose OOF probs come from the OTHER inner folds — same logic the
    GBM stack uses).

    For simplicity here we just fit LR on the training rows of each fold,
    using their OOF probs as features (these probs come from base models
    trained on the OTHER 4 folds, so they are out-of-fold for those rows and
    not leaky).
    """
    N = len(y)
    oof_stack = np.zeros(N, dtype=np.float32)
    chosen_C = []
    for tr, va in splits:
        F_tr = np.column_stack([oof_lgb[tr], oof_xgb[tr], oof_cat[tr]])
        F_va = np.column_stack([oof_lgb[va], oof_xgb[va], oof_cat[va]])

        best_auc = -1.0
        best_p = None
        best_C = None
        for C in (0.1, 1.0, 10.0):
            m = LogisticRegression(C=C, max_iter=400, solver="lbfgs")
            m.fit(F_tr, y[tr])
            p = m.predict_proba(F_va)[:, 1]
            # pick C by training-fold AUC (NOT test-fold AUC — that would leak)
            tr_auc = roc_auc_score(y[tr], m.predict_proba(F_tr)[:, 1])
            if tr_auc > best_auc:
                best_auc = tr_auc
                best_p = p
                best_C = C
        oof_stack[va] = best_p
        chosen_C.append(best_C)
    return oof_stack, chosen_C


def main():
    t_total = time.time()
    print(f"CatBoost available: {HAVE_CATBOOST}")

    # 1. Static features (441 base + 33 perceptual)
    X_base  = np.load(os.path.join(DATA_DIR, "features.npy"))
    X_extra = np.load(os.path.join(EXTRA_DIR, "features_extra.npy"))
    X_static = np.concatenate([X_base, X_extra], axis=1).astype(np.float32)
    t = np.load(os.path.join(DATA_DIR, "targets.npz"))
    g = t["gender"]
    print(f"X_static: {X_static.shape}  (441 base + 33 perceptual)")
    print(f"gender boys/girls: {int((g==0).sum())}/{int((g==1).sum())}  "
          f"({g.mean():.3f} girls)")

    # 2. Per-session bucket vectors for every stage we want to encode
    print("\nComputing per-session bucket vectors (multi-stage + LAB)...")
    t0 = time.time()
    with open(SOURCE, encoding="utf-8") as fh:
        rows = json.load(fh)

    stage_keys = ["rgb_all", "rgb_final", "rgb_r2", "rgb_r1", "rgb_nopick", "lab_all"]
    discrete = {k: [] for k in stage_keys}
    smooth   = {k: [] for k in stage_keys}

    valid_rows = []
    hue_time_per_session = []
    for row in rows:
        if not is_valid(row):
            continue
        try:
            data = session_bucket_data(row)
        except Exception as exc:
            print(f"  skip: {exc}")
            continue
        for k in stage_keys:
            d, s = data[k]
            discrete[k].append(d)
            smooth[k].append(s)
        hue_time_per_session.append(hue_time_features(row))
        valid_rows.append(row)

    for k in stage_keys:
        discrete[k] = np.array(discrete[k], dtype=np.float32)
        smooth[k]   = np.array(smooth[k],   dtype=np.float32)
    hue_time_per_session = np.array(hue_time_per_session, dtype=np.float32)
    print(f"  done in {time.time()-t0:.1f}s, shape per stage {discrete['rgb_all'].shape}")
    assert discrete["rgb_all"].shape[0] == X_static.shape[0], \
        f"row count mismatch: bucket rows {discrete['rgb_all'].shape[0]} vs static {X_static.shape[0]}"

    print("Precomputing per-session offered-colour trilinear lookups...")
    t0 = time.time()
    offered_idx, offered_w, chosen_idx = precompute_offered_lookup(valid_rows)
    print(f"  done in {time.time()-t0:.1f}s, shape {offered_idx.shape}")

    # 3. CV setup
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_static, g))
    N = len(g)

    extra_names = []
    # Block A: rgb_all signed totals (3 features, replicates color-buckets)
    for suffix in ["girly_total", "masc_total", "signed_total"]:
        extra_names.append(f"rgb_all_{suffix}")
    # Block B: LAB lab_all signed totals (3 features, NEW)
    for suffix in ["girly_total", "masc_total", "signed_total"]:
        extra_names.append(f"lab_all_{suffix}")
    # Block C: per-question gender choice features (4 features, NEW)
    extra_names.extend([
        "pq_chosen_lead_mean", "pq_chosen_lead_std",
        "pq_picked_most_girly_frac", "pq_picked_least_girly_frac",
    ])

    n_extra = len(extra_names)
    print(f"\nExtra features built per fold: {n_extra}")

    # OOFs
    oof_lgb_per = {s: np.zeros(N, dtype=np.float32) for s in LGB_BAG_SEEDS}
    oof_xgb_per = {s: np.zeros(N, dtype=np.float32) for s in XGB_BAG_SEEDS}
    oof_cat_per = {s: np.zeros(N, dtype=np.float32) for s in CAT_BAG_SEEDS} if HAVE_CATBOOST else {}
    oof_lgb_bag = np.zeros(N, dtype=np.float32)
    oof_xgb_bag = np.zeros(N, dtype=np.float32)
    oof_cat_bag = np.zeros(N, dtype=np.float32)

    fold_records = []
    for fold_i, (tr, va) in enumerate(splits):
        t_fold = time.time()

        # ---- Block A: rgb_all target encoding (matches color-buckets) ----
        def stage_grids(stage_key):
            d = discrete[stage_key]
            girly = d[tr][g[tr] == 1].mean(axis=0)
            masc  = d[tr][g[tr] == 0].mean(axis=0)
            return girly, masc

        block_features = []
        girly_grid, masc_grid = stage_grids("rgb_all")
        sm = smooth["rgb_all"]
        girly_total = sm @ girly_grid
        masc_total  = sm @ masc_grid
        signed_total = girly_total - masc_total
        block_features.extend([girly_total, masc_total, signed_total])

        # ---- Block B: LAB target encoding ----
        girly_grid_lab, masc_grid_lab = stage_grids("lab_all")
        sm_lab = smooth["lab_all"]
        block_features.extend([
            sm_lab @ girly_grid_lab,
            sm_lab @ masc_grid_lab,
            sm_lab @ girly_grid_lab - sm_lab @ masc_grid_lab,
        ])

        # ---- Block C: per-question gender-choice features ----
        # uses the SIGNED girly-grid built from rgb_all_discrete (training only).
        signed_grid_all = girly_grid - masc_grid
        pq = per_question_features_all(signed_grid_all, offered_idx, offered_w, chosen_idx)
        for col in range(pq.shape[1]):
            block_features.append(pq[:, col])

        extras = np.column_stack(block_features).astype(np.float32)
        assert extras.shape[1] == n_extra, f"got {extras.shape[1]} extras, expected {n_extra}"

        X_fold = np.concatenate([X_static, extras], axis=1)

        # ---- LightGBM bag ----
        bag_va = np.zeros(len(va), dtype=np.float32)
        lgb_seed_aucs = {}
        for s in LGB_BAG_SEEDS:
            m = lgb_clf(s)
            m.fit(X_fold[tr], g[tr])
            p = m.predict_proba(X_fold[va])[:, 1]
            oof_lgb_per[s][va] = p
            bag_va += p
            lgb_seed_aucs[s] = float(roc_auc_score(g[va], p))
        bag_va /= len(LGB_BAG_SEEDS)
        oof_lgb_bag[va] = bag_va
        lgb_bag_auc = roc_auc_score(g[va], bag_va)

        # ---- XGBoost bag ----
        bag_va = np.zeros(len(va), dtype=np.float32)
        xgb_seed_aucs = {}
        for s in XGB_BAG_SEEDS:
            m = xgb_clf(s)
            m.fit(X_fold[tr], g[tr])
            p = m.predict_proba(X_fold[va])[:, 1]
            oof_xgb_per[s][va] = p
            bag_va += p
            xgb_seed_aucs[s] = float(roc_auc_score(g[va], p))
        bag_va /= len(XGB_BAG_SEEDS)
        oof_xgb_bag[va] = bag_va
        xgb_bag_auc = roc_auc_score(g[va], bag_va)

        # ---- CatBoost bag ----
        cat_seed_aucs = {}
        if HAVE_CATBOOST:
            bag_va = np.zeros(len(va), dtype=np.float32)
            for s in CAT_BAG_SEEDS:
                m = cat_clf(s)
                m.fit(X_fold[tr], g[tr])
                p = m.predict_proba(X_fold[va])[:, 1]
                oof_cat_per[s][va] = p
                bag_va += p
                cat_seed_aucs[s] = float(roc_auc_score(g[va], p))
            bag_va /= len(CAT_BAG_SEEDS)
            oof_cat_bag[va] = bag_va
            cat_bag_auc = roc_auc_score(g[va], bag_va)
        else:
            cat_bag_auc = float("nan")

        # 50/50 of LGB+XGB and equal blend of all three
        blend_lx = 0.5 * oof_lgb_bag[va] + 0.5 * oof_xgb_bag[va]
        if HAVE_CATBOOST:
            blend_all = (oof_lgb_bag[va] + oof_xgb_bag[va] + oof_cat_bag[va]) / 3.0
            blend_all_auc = roc_auc_score(g[va], blend_all)
        else:
            blend_all_auc = float("nan")

        print(f"  fold {fold_i+1}/{N_FOLDS}  lgb={lgb_bag_auc:.4f}  "
              f"xgb={xgb_bag_auc:.4f}  cat={cat_bag_auc:.4f}  "
              f"blend_all={blend_all_auc:.4f}  ({time.time()-t_fold:.1f}s)")
        fold_records.append({
            "fold": fold_i + 1,
            "lgb_bag_auc": float(lgb_bag_auc),
            "xgb_bag_auc": float(xgb_bag_auc),
            "cat_bag_auc": float(cat_bag_auc) if HAVE_CATBOOST else None,
            "blend_all_auc": float(blend_all_auc) if HAVE_CATBOOST else None,
            "lgb_seed_aucs": lgb_seed_aucs,
            "xgb_seed_aucs": xgb_seed_aucs,
            "cat_seed_aucs": cat_seed_aucs if HAVE_CATBOOST else {},
        })

    # 4. Combined AUCs and meta-stack
    auc_lgb_bag = roc_auc_score(g, oof_lgb_bag)
    auc_xgb_bag = roc_auc_score(g, oof_xgb_bag)
    auc_cat_bag = roc_auc_score(g, oof_cat_bag) if HAVE_CATBOOST else float("nan")

    oof_blend_lx = 0.5 * oof_lgb_bag + 0.5 * oof_xgb_bag
    auc_blend_lx = roc_auc_score(g, oof_blend_lx)

    if HAVE_CATBOOST:
        oof_blend_all = (oof_lgb_bag + oof_xgb_bag + oof_cat_bag) / 3.0
        auc_blend_all = roc_auc_score(g, oof_blend_all)
        oof_stack, chosen_C = logistic_stack(
            oof_lgb_bag, oof_xgb_bag, oof_cat_bag, g, splits
        )
        auc_stack = roc_auc_score(g, oof_stack)
    else:
        oof_blend_all = None
        auc_blend_all = float("nan")
        oof_stack = None
        auc_stack = float("nan")
        chosen_C = []

    candidates = {
        "lgb_bag":   (auc_lgb_bag,  oof_lgb_bag),
        "xgb_bag":   (auc_xgb_bag,  oof_xgb_bag),
        "blend_lx":  (auc_blend_lx, oof_blend_lx),
    }
    if HAVE_CATBOOST:
        candidates["cat_bag"]   = (auc_cat_bag,   oof_cat_bag)
        candidates["blend_all"] = (auc_blend_all, oof_blend_all)
        candidates["stack_lr"]  = (auc_stack,     oof_stack)

    best_name, (best_auc, best_oof) = max(
        candidates.items(), key=lambda kv: kv[1][0]
    )
    pred = (best_oof >= 0.5).astype(int)

    lgb_per_seed_auc = {s: float(roc_auc_score(g, oof_lgb_per[s])) for s in LGB_BAG_SEEDS}
    xgb_per_seed_auc = {s: float(roc_auc_score(g, oof_xgb_per[s])) for s in XGB_BAG_SEEDS}
    cat_per_seed_auc = ({s: float(roc_auc_score(g, oof_cat_per[s])) for s in CAT_BAG_SEEDS}
                        if HAVE_CATBOOST else {})

    print("\n" + "=" * 70)
    print("FINAL GENDER AUC (5-fold OOF, seed 42 for CV)")
    print("=" * 70)
    print(f"  LightGBM bag ({len(LGB_BAG_SEEDS)} seeds)         {auc_lgb_bag:.4f}")
    print(f"  XGBoost  bag ({len(XGB_BAG_SEEDS)} seeds)         {auc_xgb_bag:.4f}")
    if HAVE_CATBOOST:
        print(f"  CatBoost bag ({len(CAT_BAG_SEEDS)} seeds)         {auc_cat_bag:.4f}")
    print(f"  50/50 LGB+XGB                          {auc_blend_lx:.4f}")
    if HAVE_CATBOOST:
        print(f"  3-way equal blend                      {auc_blend_all:.4f}")
        print(f"  Logistic stack (per-fold LR, C swept)  {auc_stack:.4f}")
    print(f"  ----")
    print(f"  HEADLINE                               {best_auc:.4f}  ({best_name})")
    print(f"  baseline LGB+buckets (color-buckets)   {BASELINE_COLOR_BUCKETS:.4f}")
    print(f"  previous gender-focused (LGB bag)      {PREV_LGB_BAG_AUC:.4f}")
    print(f"  delta vs previous                      {best_auc - PREV_LGB_BAG_AUC:+.4f}")
    print(f"  delta vs 0.900 target                  {best_auc - 0.900:+.4f}")
    print(f"  headline acc@0.5                       {accuracy_score(g, pred):.4f}")
    print(f"  headline F1                            {f1_score(g, pred):.4f}")

    # 5. Save
    save_kwargs = dict(
        lgb_bag=oof_lgb_bag, xgb_bag=oof_xgb_bag, gender=g,
        best=best_oof, blend_lx=oof_blend_lx,
        **{f"lgb_seed_{s}": oof_lgb_per[s] for s in LGB_BAG_SEEDS},
        **{f"xgb_seed_{s}": oof_xgb_per[s] for s in XGB_BAG_SEEDS},
    )
    if HAVE_CATBOOST:
        save_kwargs["cat_bag"]   = oof_cat_bag
        save_kwargs["blend_all"] = oof_blend_all
        save_kwargs["stack_lr"]  = oof_stack
        for s in CAT_BAG_SEEDS:
            save_kwargs[f"cat_seed_{s}"] = oof_cat_per[s]
    np.savez(os.path.join(HERE, "gender_oof.npz"), **save_kwargs)

    summary = {
        "best_auc": float(best_auc),
        "best_name": best_name,
        "auc_lgb_bag": float(auc_lgb_bag),
        "auc_xgb_bag": float(auc_xgb_bag),
        "auc_cat_bag": float(auc_cat_bag) if HAVE_CATBOOST else None,
        "auc_blend_lx": float(auc_blend_lx),
        "auc_blend_all": float(auc_blend_all) if HAVE_CATBOOST else None,
        "auc_stack_lr": float(auc_stack) if HAVE_CATBOOST else None,
        "lgb_per_seed_auc": lgb_per_seed_auc,
        "xgb_per_seed_auc": xgb_per_seed_auc,
        "cat_per_seed_auc": cat_per_seed_auc,
        "baseline_color_buckets": BASELINE_COLOR_BUCKETS,
        "prev_lgb_bag_auc": PREV_LGB_BAG_AUC,
        "delta_vs_previous": float(best_auc - PREV_LGB_BAG_AUC),
        "delta_vs_0_900": float(best_auc - 0.9),
        "n_static_features": int(X_static.shape[1]),
        "n_extra_features": int(n_extra),
        "extra_feature_names": extra_names,
        "lgb_bag_seeds": LGB_BAG_SEEDS,
        "xgb_bag_seeds": XGB_BAG_SEEDS,
        "cat_bag_seeds": CAT_BAG_SEEDS if HAVE_CATBOOST else [],
        "stack_chosen_C_per_fold": chosen_C,
        "fold_records": fold_records,
        "wall_time_sec": float(time.time() - t_total),
        "have_catboost": HAVE_CATBOOST,
    }
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(HERE, "feature_names.json"), "w", encoding="utf-8") as fh:
        json.dump(extra_names, fh, indent=2)

    print(f"\nwrote gender_oof.npz, summary.json, feature_names.json")
    print(f"total wall time: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
