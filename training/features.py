"""
Turn save.ligma rows into a flat per-session feature matrix.

Each row in save.ligma is:
    [id, time, ip, age, mood, gender, valg, tider, farger]
    [id, time (same as time.time()), ip (sensored), age (in years),
     mood(60-0) (0=sad, 60=happy), "gender", "choices",
     "timings for the answers", "colors"]

valg   = 21 ASCII digits, picks for the 16 round-1 groups (16 chars)
         + 4 knockout picks + 1 final.
tider  = 21 cumulative ms timestamps, last entry = total session ms.
farger = [[64 offered colors], [16 round1 winners], [4 round2 winners], final].

Validation mirrors new_code/treat_data.py so the model trains on the same
clean ~6.7k sessions the portfolio visualisation uses.

This version keeps the original summary features and adds a much richer
per-question representation:
    - chosen color per round-1 and round-2 question (RGB + HSL)
    - chosen-minus-rejected-mean delta in RGB per question
    - per-question position (which of 4 corners) and time delta
    - 4x4x4 voxel histogram of the 16 round-1 picks
    - YUV of the final color
    - PCA-friendly raw moments

Outputs three files in this folder:
    features.npy   (float32, shape [N, F])
    targets.npz    (gender, age, mood arrays)
    feature_names.json
"""

import colorsys
import datetime
import json
import os
from collections import Counter

import numpy as np

try:
    from zoneinfo import ZoneInfo
    OSLO = ZoneInfo("Europe/Oslo")
except ImportError:
    OSLO = None

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "raw", "save.ligma")
OUT_DIR = HERE

# Shared validity + troll filter (single source of truth across all trainers).
sys.path.insert(0, HERE)
from data_cleaning import is_valid_clean  # noqa: E402

DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000
N_QUESTIONS = 21
N_R1 = 16
N_R2 = 4

VOXEL_GRID = 4  # coarse 4x4x4 = 64 buckets for the round-1 pick histogram

# Reference anchor colors for "distance from final pick to X". The idea
# is to give the trees a head start at carving the color space along the
# axes that actually mean something to humans, instead of hoping the model
# discovers "near pink" from raw RGB.
REFERENCE_COLORS = {
    "pink":   (255, 182, 193),
    "red":    (220,  40,  40),
    "orange": (255, 140,   0),
    "yellow": (250, 220,  20),
    "green":  ( 50, 170,  60),
    "cyan":   ( 20, 200, 220),
    "blue":   ( 40,  60, 220),
    "purple": (140,  60, 200),
    "brown":  (140,  90,  50),
    "gray":   (128, 128, 128),
    "black":  ( 20,  20,  20),
    "white":  (240, 240, 240),
}
REFERENCE_RGB = np.array(list(REFERENCE_COLORS.values()), dtype=np.float32) / 255.0
REFERENCE_NAMES = list(REFERENCE_COLORS.keys())

TIME_BUCKETS_SEC = [(0, 1), (1, 3), (3, 7), (7, 15), (15, 1e9)]


def reference_distances(rgb):
    """Euclidean distances from one [r,g,b] color to each REFERENCE_COLORS entry."""
    c = np.array(rgb, dtype=np.float32) / 255.0
    return np.linalg.norm(REFERENCE_RGB - c, axis=1).tolist()


def is_valid(row):
    # Delegates to the shared validity + troll filter so every trainer selects
    # the exact same row set (and order) as features.npy.
    return is_valid_clean(row)


def rgb_to_hsl(rgb):
    r, g, b = (c / 255.0 for c in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, s, l


def rgb_to_yuv(rgb):
    r, g, b = (c / 255.0 for c in rgb)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.169 * r - 0.331 * g + 0.500 * b
    v = 0.500 * r - 0.419 * g - 0.081 * b
    return y, u, v


def color_stats(colors):
    """Mean and std across R, G, B, H, S, L for a list of [r,g,b] colors."""
    if not colors:
        return [0.0] * 12
    arr = np.array(colors, dtype=np.float32) / 255.0
    hsl = np.array([colorsys.rgb_to_hls(*c) for c in arr], dtype=np.float32)
    h, l, s = hsl[:, 0], hsl[:, 1], hsl[:, 2]
    rgb_mean = arr.mean(axis=0)
    rgb_std  = arr.std(axis=0)
    hsl_mean = np.array([h.mean(), s.mean(), l.mean()])
    hsl_std  = np.array([h.std(),  s.std(),  l.std()])
    return np.concatenate([rgb_mean, rgb_std, hsl_mean, hsl_std]).tolist()


def color_stat_names(prefix):
    return [
        f"{prefix}_r_mean", f"{prefix}_g_mean", f"{prefix}_b_mean",
        f"{prefix}_r_std",  f"{prefix}_g_std",  f"{prefix}_b_std",
        f"{prefix}_h_mean", f"{prefix}_s_mean", f"{prefix}_l_mean",
        f"{prefix}_h_std",  f"{prefix}_s_std",  f"{prefix}_l_std",
    ]


def voxel_id(rgb, grid=8):
    shift = 8 - int(np.log2(grid))
    return (rgb[0] >> shift, rgb[1] >> shift, rgb[2] >> shift)


def hour_minute(timestamp):
    try:
        t = int(timestamp)
    except (TypeError, ValueError):
        return None, None
    if t <= 0:
        return None, None
    try:
        dt = (datetime.datetime.fromtimestamp(t, tz=OSLO)
              if OSLO is not None
              else datetime.datetime.fromtimestamp(t))
        return dt.hour, dt.minute
    except (OSError, OverflowError, ValueError):
        return None, None


def per_question_color_block(prefix, color, names_out, vals_out):
    """Add R, G, B, H, S, L plus warmth and chroma for one chosen color."""
    r, g, b = color
    h, s, l = rgb_to_hsl(color)
    names_out.extend([
        f"{prefix}_r", f"{prefix}_g", f"{prefix}_b",
        f"{prefix}_h", f"{prefix}_s", f"{prefix}_l",
        f"{prefix}_warmth", f"{prefix}_chroma",
    ])
    vals_out.extend([
        r / 255.0, g / 255.0, b / 255.0,
        h, s, l,
        (r - b) / 255.0,
        (max(color) - min(color)) / 255.0,
    ])


def extract_features(row):
    age = int(row[3])
    mood = int(row[4])
    gender = 1 if row[5] == "j" else 0

    valg = row[6]
    tider = [int(x) for x in row[7]]
    offered = row[8][0]
    r1 = row[8][1]
    r2 = row[8][2]
    final = row[8][3]

    names = []
    vals  = []

    def push(name, value):
        names.append(name)
        vals.append(float(value))

    # ---- Per-stage color summary stats ----
    for prefix, colors in (("off", offered), ("r1", r1), ("r2", r2)):
        for name, value in zip(color_stat_names(prefix), color_stats(colors)):
            push(name, value)

    # ---- Final color: RGB, HSL, YUV, warmth, chroma ----
    fr, fg, fb = final
    fh, fs, fl = rgb_to_hsl(final)
    fy, fu, fv = rgb_to_yuv(final)
    push("final_r", fr / 255.0)
    push("final_g", fg / 255.0)
    push("final_b", fb / 255.0)
    push("final_h", fh)
    push("final_s", fs)
    push("final_l", fl)
    push("final_y", fy)
    push("final_u", fu)
    push("final_v", fv)
    push("final_warmth", (fr - fb) / 255.0)
    push("final_chroma", (max(final) - min(final)) / 255.0)

    # ---- Selectivity deltas (round-mean vs offered-mean per channel) ----
    off_mean = np.array(offered, dtype=np.float32).mean(axis=0) / 255.0
    r1_mean  = np.array(r1,      dtype=np.float32).mean(axis=0) / 255.0
    r2_mean  = np.array(r2,      dtype=np.float32).mean(axis=0) / 255.0
    push("sel_r1_dr", r1_mean[0] - off_mean[0])
    push("sel_r1_dg", r1_mean[1] - off_mean[1])
    push("sel_r1_db", r1_mean[2] - off_mean[2])
    push("sel_r2_dr", r2_mean[0] - r1_mean[0])
    push("sel_r2_dg", r2_mean[1] - r1_mean[1])
    push("sel_r2_db", r2_mean[2] - r1_mean[2])

    # ---- Voxel diversity ----
    push("voxel_div_r1", len({voxel_id(c, 8) for c in r1}) / N_R1)
    push("voxel_div_r2", len({voxel_id(c, 8) for c in r2}) / N_R2)

    # ---- Internal spread of round-1 picks ----
    arr = np.array(r1, dtype=np.float32) / 255.0
    diffs = arr[:, None, :] - arr[None, :, :]
    push("r1_internal_spread", float(np.sqrt((diffs ** 2).sum(-1)).mean()))

    # ---- Per-round-1 question features: chosen color + delta + position ----
    decisive = []
    for q in range(N_R1):
        group = offered[q * 4:(q + 1) * 4]
        idx = int(valg[q]) if 0 <= int(valg[q]) <= 3 else 0
        chosen = group[idx]
        rejected = np.array(
            [group[i] for i in range(4) if i != idx], dtype=np.float32
        ) / 255.0
        chosen_arr = np.array(chosen, dtype=np.float32) / 255.0

        per_question_color_block(f"q{q:02d}_chosen", chosen, names, vals)

        # Channel-wise delta from rejected mean
        delta = chosen_arr - rejected.mean(axis=0)
        push(f"q{q:02d}_dr", float(delta[0]))
        push(f"q{q:02d}_dg", float(delta[1]))
        push(f"q{q:02d}_db", float(delta[2]))

        # Decisiveness magnitude
        decisive.append(float(np.linalg.norm(delta)))
        push(f"q{q:02d}_decisive", decisive[-1])

        # Position chosen (0..3)
        push(f"q{q:02d}_pos", idx)

    push("mean_decisiveness", float(np.mean(decisive)))
    push("std_decisiveness",  float(np.std(decisive)))

    # ---- Per-round-2 question features: chosen color + position ----
    for q in range(N_R2):
        chosen = r2[q]
        per_question_color_block(f"r2q{q}_chosen", chosen, names, vals)
        try:
            pos = int(valg[N_R1 + q])
            push(f"r2q{q}_pos", pos if 0 <= pos <= 3 else 0)
        except (ValueError, IndexError):
            push(f"r2q{q}_pos", 0)

    # ---- Position picks aggregate (already captured per-question, but easy summaries help linear models) ----
    pos_counts = Counter(valg[:N_R1])
    for p in "0123":
        push(f"pos_{p}_frac", pos_counts.get(p, 0) / N_R1)
    probs = np.array([pos_counts.get(p, 0) for p in "0123"], dtype=np.float32) / N_R1
    probs = probs[probs > 0]
    push("pos_entropy", float(-(probs * np.log(probs)).sum()) if len(probs) else 0.0)

    # ---- Voxel histogram of round-1 picks (4x4x4 = 64 buckets, integer counts 0..16) ----
    hist = np.zeros(VOXEL_GRID ** 3, dtype=np.float32)
    for c in r1:
        vx = voxel_id(c, VOXEL_GRID)
        hist[vx[0] * VOXEL_GRID * VOXEL_GRID + vx[1] * VOXEL_GRID + vx[2]] += 1
    for i, n in enumerate(hist):
        push(f"hist_v{i:02d}", n / N_R1)

    # ---- Reference color distances for the final pick ----
    for name, dist in zip(REFERENCE_NAMES, reference_distances(final)):
        push(f"final_to_{name}", dist)
    push("final_closest_ref", int(np.argmin(reference_distances(final))))

    # ---- Distance from final pick to round-1 mean and round-2 mean ----
    final_arr = np.array(final, dtype=np.float32) / 255.0
    push("final_to_r1_mean", float(np.linalg.norm(final_arr - r1_mean)))
    push("final_to_r2_mean", float(np.linalg.norm(final_arr - r2_mean)))

    # ---- Trajectory across the 16 round-1 picks ----
    r1_arr = np.array(r1, dtype=np.float32) / 255.0
    consec = np.linalg.norm(r1_arr[1:] - r1_arr[:-1], axis=1)
    push("r1_consec_mean", float(consec.mean()))
    push("r1_consec_std",  float(consec.std()))
    centroid = r1_arr.mean(axis=0)
    push("r1_centroid_mean_dist", float(np.linalg.norm(r1_arr - centroid, axis=1).mean()))

    # Slopes of warmth, lightness, saturation across r1 picks (drift over the test)
    warmth_r1 = (r1_arr[:, 0] - r1_arr[:, 2])
    light_r1 = np.array([colorsys.rgb_to_hls(*c)[1] for c in r1_arr], dtype=np.float32)
    sat_r1   = np.array([colorsys.rgb_to_hls(*c)[2] for c in r1_arr], dtype=np.float32)
    xs = np.arange(N_R1, dtype=np.float32)
    push("r1_warmth_slope", float(np.polyfit(xs, warmth_r1, 1)[0]))
    push("r1_light_slope",  float(np.polyfit(xs, light_r1,  1)[0]))
    push("r1_sat_slope",    float(np.polyfit(xs, sat_r1,    1)[0]))

    # ---- Extreme-pick counts across the 16 round-1 questions ----
    # How often the chosen color was the warmest / coolest / lightest / darkest /
    # most-saturated / least-saturated of the 4 offered.
    counts = {"warmest": 0, "coolest": 0, "lightest": 0,
              "darkest": 0, "most_sat": 0, "least_sat": 0}
    for q in range(N_R1):
        group = offered[q * 4:(q + 1) * 4]
        try:
            idx = int(valg[q])
        except (ValueError, IndexError):
            continue
        if not (0 <= idx <= 3):
            continue
        warmths = [c[0] - c[2] for c in group]
        lights  = [colorsys.rgb_to_hls(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)[1]
                   for c in group]
        sats    = [colorsys.rgb_to_hls(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)[2]
                   for c in group]
        if idx == int(np.argmax(warmths)): counts["warmest"]   += 1
        if idx == int(np.argmin(warmths)): counts["coolest"]   += 1
        if idx == int(np.argmax(lights)):  counts["lightest"]  += 1
        if idx == int(np.argmin(lights)):  counts["darkest"]   += 1
        if idx == int(np.argmax(sats)):    counts["most_sat"]  += 1
        if idx == int(np.argmin(sats)):    counts["least_sat"] += 1
    for k, v in counts.items():
        push(f"extreme_{k}_frac", v / N_R1)

    # ---- Round-1 time bucket histogram ----
    # Compute per-question time deltas in seconds (inline; the main timing
    # block below recomputes these in ms for its own features).
    _deltas_ms = [max(0, tider[0])] + [
        max(0, tider[i] - tider[i - 1]) for i in range(1, len(tider))
    ]
    r1_times_sec = [_deltas_ms[q] / 1000.0 for q in range(N_R1)]
    bucket_counts = [0] * len(TIME_BUCKETS_SEC)
    for t in r1_times_sec:
        for i, (lo, hi) in enumerate(TIME_BUCKETS_SEC):
            if lo <= t < hi:
                bucket_counts[i] += 1
                break
    for i, n in enumerate(bucket_counts):
        push(f"r1_tbucket_{i}", n / N_R1)

    # ---- Time x behaviour interactions ----
    mean_q_ms = float(np.mean(_deltas_ms))
    probs2 = np.array(
        [Counter(valg[:N_R1]).get(p, 0) for p in "0123"], dtype=np.float32
    ) / N_R1
    probs2 = probs2[probs2 > 0]
    entropy_val = float(-(probs2 * np.log(probs2)).sum()) if len(probs2) else 0.0
    # Slow + concentrated picking is "deliberate corner-banger"
    push("time_x_low_entropy", mean_q_ms * (1.0 - entropy_val / np.log(4)))
    push("time_per_decisive", mean_q_ms / (float(np.mean(decisive)) + 50.0))

    # ---- Timing ----
    deltas = [tider[0]] + [tider[i] - tider[i - 1] for i in range(1, len(tider))]
    deltas = [max(d, 0) for d in deltas]
    total_ms = tider[-1]
    push("total_sec",      total_ms / 1000.0)
    push("mean_q_ms",      float(np.mean(deltas)))
    push("std_q_ms",       float(np.std(deltas)))
    push("min_q_ms",       float(np.min(deltas)))
    push("max_q_ms",       float(np.max(deltas)))
    push("median_q_ms",    float(np.median(deltas)))
    push("first5_mean_ms", float(np.mean(deltas[:5])))
    push("last5_mean_ms",  float(np.mean(deltas[-5:])))
    x = np.arange(len(deltas), dtype=np.float32)
    y = np.array(deltas, dtype=np.float32)
    push("time_slope", float(np.polyfit(x, y, 1)[0]))
    push("r1_mean_ms",  float(np.mean(deltas[:N_R1])))
    push("r2_mean_ms",  float(np.mean(deltas[N_R1:N_R1 + N_R2])))
    push("final_ms",    float(deltas[-1]))

    # Per-question raw times (21 values, ms / 1000)
    for q in range(N_QUESTIONS):
        push(f"t_q{q:02d}", deltas[q] / 1000.0)

    # ---- Hour of day ----
    h, m = hour_minute(row[1])
    if h is None:
        push("hour_sin", 0.0)
        push("hour_cos", 0.0)
        push("hour_known", 0.0)
    else:
        frac = (h + m / 60.0) / 24.0
        push("hour_sin", float(np.sin(2 * np.pi * frac)))
        push("hour_cos", float(np.cos(2 * np.pi * frac)))
        push("hour_known", 1.0)

    return names, vals, gender, age, mood


def main():
    with open(SOURCE, "r", encoding="utf-8") as fh:
        rows = json.load(fh)

    print(f"loaded {len(rows)} raw rows")

    feat_rows = []
    genders, ages, moods = [], [], []
    feature_names = None

    for row in rows:
        if not is_valid(row):
            continue
        try:
            names, vals, gender, age, mood = extract_features(row)
        except Exception as exc:
            print("skip:", exc)
            continue
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            print("feature order drift, skipping row")
            continue
        feat_rows.append(vals)
        genders.append(gender)
        ages.append(age)
        moods.append(mood)

    X = np.array(feat_rows, dtype=np.float32)
    g = np.array(genders, dtype=np.int8)
    a = np.array(ages,    dtype=np.int16)
    m = np.array(moods,   dtype=np.int16)

    print(f"used {X.shape[0]} valid rows, {X.shape[1]} features")
    print(f"gender: boys={int((g == 0).sum())} girls={int((g == 1).sum())}")
    print(f"age:    min={a.min()} max={a.max()} mean={a.mean():.1f}")
    print(f"mood:   min={m.min()} max={m.max()} mean={m.mean():.1f}")

    np.save(os.path.join(OUT_DIR, "features.npy"), X)
    np.savez(os.path.join(OUT_DIR, "targets.npz"), gender=g, age=a, mood=m)
    with open(os.path.join(OUT_DIR, "feature_names.json"), "w", encoding="utf-8") as fh:
        json.dump(feature_names, fh, indent=2)

    print(f"wrote features.npy, targets.npz, feature_names.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
