"""
Pure-Python feature extraction for the Cloudflare Workers Python runtime.

Ports `training/features.py` + `training/extra-features/train.py` + the
bucket-score lookup from `training/color-buckets/train.py` to plain Python
(no numpy / no scipy). The output is three flat feature vectors - 477 for the
gender head, 475 for age, 475 for mood - in the exact column order the
LightGBM boosters expect.

The browser fetches those vectors, then walks the matching JSON tree from
`models-js/*_trees.json` using `tree_walker.js`.
"""

import colorsys
import datetime
import math
from typing import Dict, List, Sequence, Tuple

try:
    # Available on >=3.9 and on Pyodide. We import lazily because zoneinfo
    # on Pyodide pulls in tzdata which may not be present; fall back to UTC.
    from zoneinfo import ZoneInfo
    _OSLO = ZoneInfo("Europe/Oslo")
except Exception:
    _OSLO = None

from bucket_data import (
    GRID, N_BUCKETS, BUCKET_WIDTH, BUCKET_CENTER_OFFSET,
    PICK_VALUE, NOT_PICK_VALUE,
    GIRLY_GRID, MASC_GRID, AGE_GRID, MOOD_GRID,
)

N_QUESTIONS = 21
N_R1 = 16
N_R2 = 4
VOXEL_GRID = 4  # 4x4x4 = 64 buckets for the round-1 pick histogram

REFERENCE_COLORS = (
    ("pink",   (255, 182, 193)),
    ("red",    (220,  40,  40)),
    ("orange", (255, 140,   0)),
    ("yellow", (250, 220,  20)),
    ("green",  ( 50, 170,  60)),
    ("cyan",   ( 20, 200, 220)),
    ("blue",   ( 40,  60, 220)),
    ("purple", (140,  60, 200)),
    ("brown",  (140,  90,  50)),
    ("gray",   (128, 128, 128)),
    ("black",  ( 20,  20,  20)),
    ("white",  (240, 240, 240)),
)
REFERENCE_NORM = tuple((r / 255.0, g / 255.0, b / 255.0) for _, (r, g, b) in REFERENCE_COLORS)
TIME_BUCKETS_SEC = ((0, 1), (1, 3), (3, 7), (7, 15), (15, 1e9))


# ---------- small numeric helpers ----------

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _polyfit_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    denom = sum((xs[i] - mx) ** 2 for i in range(n))
    return num / denom if denom else 0.0


def _argmin(xs: Sequence[float]) -> int:
    best = 0
    bv = xs[0]
    for i in range(1, len(xs)):
        if xs[i] < bv:
            bv = xs[i]
            best = i
    return best


def _argmax(xs: Sequence[float]) -> int:
    best = 0
    bv = xs[0]
    for i in range(1, len(xs)):
        if xs[i] > bv:
            bv = xs[i]
            best = i
    return best


def _reference_distances(rgb: Sequence[int]) -> List[float]:
    rn = rgb[0] / 255.0
    gn = rgb[1] / 255.0
    bn = rgb[2] / 255.0
    out = []
    for (r, g, b) in REFERENCE_NORM:
        dr = r - rn
        dg = g - gn
        db = b - bn
        out.append(math.sqrt(dr * dr + dg * dg + db * db))
    return out


def _rgb_to_hsl(rgb: Sequence[int]) -> Tuple[float, float, float]:
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, s, l


def _rgb_to_yuv(rgb: Sequence[int]) -> Tuple[float, float, float]:
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.169 * r - 0.331 * g + 0.500 * b
    v = 0.500 * r - 0.419 * g - 0.081 * b
    return y, u, v


def _color_stats(colors: Sequence[Sequence[int]]) -> List[float]:
    """mean+std across R, G, B (normalized 0..1) and H, S, L."""
    if not colors:
        return [0.0] * 12
    r_vals, g_vals, b_vals = [], [], []
    h_vals, s_vals, l_vals = [], [], []
    for c in colors:
        r_vals.append(c[0] / 255.0)
        g_vals.append(c[1] / 255.0)
        b_vals.append(c[2] / 255.0)
        h, s, l = _rgb_to_hsl(c)
        h_vals.append(h)
        s_vals.append(s)
        l_vals.append(l)
    return [
        _mean(r_vals), _mean(g_vals), _mean(b_vals),
        _std(r_vals),  _std(g_vals),  _std(b_vals),
        _mean(h_vals), _mean(s_vals), _mean(l_vals),
        _std(h_vals),  _std(s_vals),  _std(l_vals),
    ]


def _voxel_id(rgb: Sequence[int], grid: int) -> Tuple[int, int, int]:
    shift = 8 - int(math.log2(grid))
    return (rgb[0] >> shift, rgb[1] >> shift, rgb[2] >> shift)


def _hour_minute(timestamp) -> Tuple:
    try:
        t = int(timestamp)
    except (TypeError, ValueError):
        return None, None
    if t <= 0:
        return None, None
    try:
        if _OSLO is not None:
            dt = datetime.datetime.fromtimestamp(t, tz=_OSLO)
        else:
            dt = datetime.datetime.fromtimestamp(t)
        return dt.hour, dt.minute
    except (OSError, OverflowError, ValueError):
        return None, None


def _per_question_color_block(prefix: str, color: Sequence[int]) -> List[float]:
    r, g, b = color
    h, s, l = _rgb_to_hsl(color)
    return [
        r / 255.0, g / 255.0, b / 255.0,
        h, s, l,
        (r - b) / 255.0,
        (max(color) - min(color)) / 255.0,
    ]


# ---------- LAB conversion (for the perceptual extras) ----------

def _gamma(c: float) -> float:
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92


def _srgb_to_lab(rgb: Sequence[int]) -> Tuple[float, float, float]:
    r = _gamma(rgb[0] / 255.0)
    g = _gamma(rgb[1] / 255.0)
    b = _gamma(rgb[2] / 255.0)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    def f(t):
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


_GIRL_PROTOTYPES_RGB = (
    (255, 182, 193),
    (255, 105, 180),
    (200, 100, 200),
    (220,  40,  40),
)
_BOY_PROTOTYPES_RGB = (
    ( 40,  60, 220),
    ( 60, 130, 220),
    ( 50, 170,  60),
    ( 60,  60,  60),
)


def _proto_mean(rgbs):
    labs = [_srgb_to_lab(c) for c in rgbs]
    return (
        sum(l[0] for l in labs) / len(labs),
        sum(l[1] for l in labs) / len(labs),
        sum(l[2] for l in labs) / len(labs),
    )


_GIRL_PROTO = _proto_mean(_GIRL_PROTOTYPES_RGB)
_BOY_PROTO  = _proto_mean(_BOY_PROTOTYPES_RGB)


def _lab_distance(lab1, lab2):
    return math.sqrt(
        (lab1[0] - lab2[0]) ** 2
        + (lab1[1] - lab2[1]) ** 2
        + (lab1[2] - lab2[2]) ** 2
    )


# ---------- 441 base features ----------

def _extract_base(payload: Dict, submit_unix: int) -> List[float]:
    """Port of training/features.py:extract_features (drop the gender/age/mood
    return tuple since they're targets, not features)."""
    valg = payload["valg"]
    tider = [int(x) for x in payload["tider"]]
    offered = payload["offered"]
    r1 = payload["r1"]
    r2 = payload["r2"]
    final = payload["final"]

    vals: List[float] = []

    # ---- Per-stage color summary stats ----
    for colors in (offered, r1, r2):
        vals.extend(_color_stats(colors))

    # ---- Final color: RGB, HSL, YUV, warmth, chroma ----
    fr, fg, fb = final
    fh, fs, fl = _rgb_to_hsl(final)
    fy, fu, fv = _rgb_to_yuv(final)
    vals.extend([
        fr / 255.0, fg / 255.0, fb / 255.0,
        fh, fs, fl,
        fy, fu, fv,
        (fr - fb) / 255.0,
        (max(final) - min(final)) / 255.0,
    ])

    # ---- Selectivity deltas ----
    off_r_mean = sum(c[0] for c in offered) / (255.0 * len(offered))
    off_g_mean = sum(c[1] for c in offered) / (255.0 * len(offered))
    off_b_mean = sum(c[2] for c in offered) / (255.0 * len(offered))
    r1_r_mean = sum(c[0] for c in r1) / (255.0 * len(r1))
    r1_g_mean = sum(c[1] for c in r1) / (255.0 * len(r1))
    r1_b_mean = sum(c[2] for c in r1) / (255.0 * len(r1))
    r2_r_mean = sum(c[0] for c in r2) / (255.0 * len(r2))
    r2_g_mean = sum(c[1] for c in r2) / (255.0 * len(r2))
    r2_b_mean = sum(c[2] for c in r2) / (255.0 * len(r2))
    vals.extend([
        r1_r_mean - off_r_mean,
        r1_g_mean - off_g_mean,
        r1_b_mean - off_b_mean,
        r2_r_mean - r1_r_mean,
        r2_g_mean - r1_g_mean,
        r2_b_mean - r1_b_mean,
    ])

    # ---- Voxel diversity ----
    voxel_r1 = {_voxel_id(c, 8) for c in r1}
    voxel_r2 = {_voxel_id(c, 8) for c in r2}
    vals.append(len(voxel_r1) / N_R1)
    vals.append(len(voxel_r2) / N_R2)

    # ---- Internal spread of round-1 picks ----
    r1_norm = [(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0) for c in r1]
    n_r1 = len(r1_norm)
    spread_sum = 0.0
    for i in range(n_r1):
        for j in range(n_r1):
            dr = r1_norm[i][0] - r1_norm[j][0]
            dg = r1_norm[i][1] - r1_norm[j][1]
            db = r1_norm[i][2] - r1_norm[j][2]
            spread_sum += math.sqrt(dr * dr + dg * dg + db * db)
    vals.append(spread_sum / (n_r1 * n_r1))

    # ---- Per-round-1 question features ----
    decisive: List[float] = []
    pos_counts = [0, 0, 0, 0]
    for q in range(N_R1):
        group = offered[q * 4:(q + 1) * 4]
        try:
            idx = int(valg[q])
        except (ValueError, IndexError):
            idx = 0
        if not (0 <= idx <= 3):
            idx = 0
        chosen = group[idx]
        rejected = [group[i] for i in range(4) if i != idx]
        rej_r = sum(c[0] for c in rejected) / (255.0 * 3)
        rej_g = sum(c[1] for c in rejected) / (255.0 * 3)
        rej_b = sum(c[2] for c in rejected) / (255.0 * 3)
        chosen_r = chosen[0] / 255.0
        chosen_g = chosen[1] / 255.0
        chosen_b = chosen[2] / 255.0
        dr = chosen_r - rej_r
        dg = chosen_g - rej_g
        db = chosen_b - rej_b

        vals.extend(_per_question_color_block(f"q{q:02d}_chosen", chosen))
        vals.extend([dr, dg, db])

        dec = math.sqrt(dr * dr + dg * dg + db * db)
        decisive.append(dec)
        vals.append(dec)
        vals.append(idx)
        pos_counts[idx] += 1

    vals.append(_mean(decisive))
    vals.append(_std(decisive))

    # ---- Per-round-2 question features ----
    for q in range(N_R2):
        chosen = r2[q]
        vals.extend(_per_question_color_block(f"r2q{q}_chosen", chosen))
        try:
            pos = int(valg[N_R1 + q])
            if not (0 <= pos <= 3):
                pos = 0
        except (ValueError, IndexError):
            pos = 0
        vals.append(pos)

    # ---- Position-pick aggregate ----
    for p in range(4):
        vals.append(pos_counts[p] / N_R1)
    probs = [c / N_R1 for c in pos_counts if c > 0]
    if probs:
        vals.append(-sum(p * math.log(p) for p in probs))
    else:
        vals.append(0.0)

    # ---- Voxel histogram of round-1 picks (4x4x4 = 64 buckets) ----
    hist = [0.0] * (VOXEL_GRID ** 3)
    for c in r1:
        v = _voxel_id(c, VOXEL_GRID)
        hist[v[0] * VOXEL_GRID * VOXEL_GRID + v[1] * VOXEL_GRID + v[2]] += 1
    for n in hist:
        vals.append(n / N_R1)

    # ---- Reference color distances ----
    dists = _reference_distances(final)
    vals.extend(dists)
    vals.append(_argmin(dists))

    # ---- Distance from final to r1_mean and r2_mean ----
    final_n = (final[0] / 255.0, final[1] / 255.0, final[2] / 255.0)
    r1_mean = (r1_r_mean, r1_g_mean, r1_b_mean)
    r2_mean = (r2_r_mean, r2_g_mean, r2_b_mean)
    vals.append(_norm((final_n[0] - r1_mean[0], final_n[1] - r1_mean[1], final_n[2] - r1_mean[2])))
    vals.append(_norm((final_n[0] - r2_mean[0], final_n[1] - r2_mean[1], final_n[2] - r2_mean[2])))

    # ---- Trajectory across the 16 round-1 picks ----
    consec = []
    for i in range(1, n_r1):
        dr = r1_norm[i][0] - r1_norm[i - 1][0]
        dg = r1_norm[i][1] - r1_norm[i - 1][1]
        db = r1_norm[i][2] - r1_norm[i - 1][2]
        consec.append(math.sqrt(dr * dr + dg * dg + db * db))
    vals.append(_mean(consec))
    vals.append(_std(consec))
    cx = sum(c[0] for c in r1_norm) / n_r1
    cy = sum(c[1] for c in r1_norm) / n_r1
    cz = sum(c[2] for c in r1_norm) / n_r1
    cent_d = []
    for c in r1_norm:
        cent_d.append(math.sqrt((c[0] - cx) ** 2 + (c[1] - cy) ** 2 + (c[2] - cz) ** 2))
    vals.append(_mean(cent_d))

    warmth_r1 = [c[0] - c[2] for c in r1_norm]
    light_r1 = [colorsys.rgb_to_hls(c[0], c[1], c[2])[1] for c in r1_norm]
    sat_r1   = [colorsys.rgb_to_hls(c[0], c[1], c[2])[2] for c in r1_norm]
    xs = list(range(N_R1))
    xs_f = [float(x) for x in xs]
    vals.append(_polyfit_slope(xs_f, warmth_r1))
    vals.append(_polyfit_slope(xs_f, light_r1))
    vals.append(_polyfit_slope(xs_f, sat_r1))

    # ---- Extreme-pick counts ----
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
        lights = []
        sats = []
        for c in group:
            _, l, s = colorsys.rgb_to_hls(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
            lights.append(l)
            sats.append(s)
        if idx == _argmax(warmths): counts["warmest"]   += 1
        if idx == _argmin(warmths): counts["coolest"]   += 1
        if idx == _argmax(lights):  counts["lightest"]  += 1
        if idx == _argmin(lights):  counts["darkest"]   += 1
        if idx == _argmax(sats):    counts["most_sat"]  += 1
        if idx == _argmin(sats):    counts["least_sat"] += 1
    for k in ("warmest", "coolest", "lightest", "darkest", "most_sat", "least_sat"):
        vals.append(counts[k] / N_R1)

    # ---- Round-1 time bucket histogram ----
    deltas_ms = [max(0, tider[0])] + [
        max(0, tider[i] - tider[i - 1]) for i in range(1, len(tider))
    ]
    r1_times_sec = [deltas_ms[q] / 1000.0 for q in range(N_R1)]
    bucket_counts = [0] * len(TIME_BUCKETS_SEC)
    for t in r1_times_sec:
        for i, (lo, hi) in enumerate(TIME_BUCKETS_SEC):
            if lo <= t < hi:
                bucket_counts[i] += 1
                break
    for n in bucket_counts:
        vals.append(n / N_R1)

    # ---- Time x behaviour interactions ----
    mean_q_ms = _mean(deltas_ms)
    valg_r1 = valg[:N_R1]
    pc2 = [0, 0, 0, 0]
    for ch in valg_r1:
        try:
            i = int(ch)
            if 0 <= i <= 3:
                pc2[i] += 1
        except ValueError:
            pass
    probs2 = [c / N_R1 for c in pc2 if c > 0]
    entropy_val = -sum(p * math.log(p) for p in probs2) if probs2 else 0.0
    vals.append(mean_q_ms * (1.0 - entropy_val / math.log(4)))
    vals.append(mean_q_ms / (_mean(decisive) + 50.0))

    # ---- Timing ----
    total_ms = tider[-1]
    vals.append(total_ms / 1000.0)
    vals.append(mean_q_ms)
    vals.append(_std(deltas_ms))
    vals.append(float(min(deltas_ms)))
    vals.append(float(max(deltas_ms)))
    sorted_deltas = sorted(deltas_ms)
    n_d = len(sorted_deltas)
    median = (sorted_deltas[n_d // 2] + sorted_deltas[(n_d - 1) // 2]) / 2.0
    vals.append(median)
    vals.append(_mean(deltas_ms[:5]))
    vals.append(_mean(deltas_ms[-5:]))
    xs_t = [float(i) for i in range(len(deltas_ms))]
    ys_t = [float(d) for d in deltas_ms]
    vals.append(_polyfit_slope(xs_t, ys_t))
    vals.append(_mean(deltas_ms[:N_R1]))
    vals.append(_mean(deltas_ms[N_R1:N_R1 + N_R2]))
    vals.append(float(deltas_ms[-1]))

    for q in range(N_QUESTIONS):
        vals.append(deltas_ms[q] / 1000.0)

    # ---- Hour of day ----
    h, m_ = _hour_minute(submit_unix)
    if h is None:
        vals.extend([0.0, 0.0, 0.0])
    else:
        frac = (h + m_ / 60.0) / 24.0
        vals.append(math.sin(2 * math.pi * frac))
        vals.append(math.cos(2 * math.pi * frac))
        vals.append(1.0)

    return vals


# ---------- 33 perceptual extras ----------

def _extract_extras(payload: Dict) -> List[float]:
    valg = payload["valg"]
    tider = [int(x) for x in payload["tider"]]
    offered = payload["offered"]
    r1 = payload["r1"]
    r2 = payload["r2"]
    final = payload["final"]

    offered_lab = [_srgb_to_lab(c) for c in offered]
    r1_lab = [_srgb_to_lab(c) for c in r1]
    r2_lab = [_srgb_to_lab(c) for c in r2]
    final_lab = _srgb_to_lab(final)

    vals: List[float] = []

    # Block A: final LAB
    vals.extend([final_lab[0], final_lab[1], final_lab[2]])

    # Block B: r1 LAB stats
    r1_L = [c[0] for c in r1_lab]
    r1_a = [c[1] for c in r1_lab]
    r1_b = [c[2] for c in r1_lab]
    vals.extend([_mean(r1_L), _mean(r1_a), _mean(r1_b)])
    vals.extend([_std(r1_L),  _std(r1_a),  _std(r1_b)])

    # Block C: r2 LAB stats
    r2_L = [c[0] for c in r2_lab]
    r2_a = [c[1] for c in r2_lab]
    r2_b = [c[2] for c in r2_lab]
    vals.extend([_mean(r2_L), _mean(r2_a), _mean(r2_b)])
    vals.extend([_std(r2_L),  _std(r2_a),  _std(r2_b)])

    # Block D: offered LAB stats
    off_L = [c[0] for c in offered_lab]
    off_a = [c[1] for c in offered_lab]
    off_b = [c[2] for c in offered_lab]
    vals.extend([_mean(off_L), _mean(off_a), _mean(off_b)])
    vals.extend([_std(off_L),  _std(off_a),  _std(off_b)])

    # Block E: gender-prototype distances
    d_final_girl = _lab_distance(final_lab, _GIRL_PROTO)
    d_final_boy  = _lab_distance(final_lab, _BOY_PROTO)
    vals.append(d_final_girl)
    vals.append(d_final_boy)
    vals.append(math.log((d_final_boy + 1.0) / (d_final_girl + 1.0)))
    vals.append(_mean([_lab_distance(c, _GIRL_PROTO) for c in r1_lab]))
    vals.append(_mean([_lab_distance(c, _BOY_PROTO)  for c in r1_lab]))

    # Block F: difficulty + relative decisiveness
    diversities: List[float] = []
    rel_dec: List[float] = []
    for q in range(N_R1):
        group = offered_lab[q * 4:(q + 1) * 4]
        # mean pairwise distance within the offered group (upper triangle)
        total = 0.0
        n = 0
        for i in range(4):
            for j in range(i + 1, 4):
                total += _lab_distance(group[i], group[j])
                n += 1
        diversity = total / n if n else 0.0
        diversities.append(diversity)
        try:
            idx = int(valg[q])
            if not (0 <= idx <= 3):
                idx = 0
        except (ValueError, IndexError):
            idx = 0
        chosen = group[idx]
        rejected = [group[i] for i in range(4) if i != idx]
        ra = sum(r[0] for r in rejected) / 3
        rb = sum(r[1] for r in rejected) / 3
        rc = sum(r[2] for r in rejected) / 3
        delta = math.sqrt(
            (chosen[0] - ra) ** 2 + (chosen[1] - rb) ** 2 + (chosen[2] - rc) ** 2
        )
        rel_dec.append(delta / (diversity + 1.0))

    vals.extend([_mean(diversities), _std(diversities), _mean(rel_dec), _std(rel_dec)])

    # Block G: time anomalies
    deltas_ms = [max(0, tider[0])] + [
        max(0, tider[i] - tider[i - 1]) for i in range(1, len(tider))
    ]
    deltas_sec = [d / 1000.0 for d in deltas_ms]
    rushed = sum(1 for t in deltas_sec if t < 1.0) / len(deltas_sec)
    dwelled = sum(1 for t in deltas_sec if t > 7.0) / len(deltas_sec)
    cv = _std(deltas_ms) / (_mean(deltas_ms) + 1.0)
    vals.extend([rushed, dwelled, cv])

    return vals


# ---------- bucket-score totals ----------

def _bucket_id(r: int, g: int, b: int) -> int:
    return r * GRID * GRID + g * GRID + b


def _discrete_bucket(rgb: Sequence[int]) -> int:
    r = min(GRID - 1, int(rgb[0]) // int(BUCKET_WIDTH))
    g = min(GRID - 1, int(rgb[1]) // int(BUCKET_WIDTH))
    b = min(GRID - 1, int(rgb[2]) // int(BUCKET_WIDTH))
    return _bucket_id(r, g, b)


def _trilinear_weights(rgb: Sequence[int]):
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


def _compute_bucket_totals(payload: Dict):
    """Return (girly_total, masc_total, signed_total, age_total, mood_total) by
    walking the smooth_delta of this session against the precomputed bucket
    grids."""
    offered = payload["offered"]
    r1 = payload["r1"]
    r2 = payload["r2"]
    final = payload["final"]
    valg = payload["valg"]

    picked = set()
    for q in range(N_R1):
        try:
            idx = int(valg[q])
            if 0 <= idx <= 3:
                picked.add(q * 4 + idx)
        except (ValueError, IndexError):
            pass

    girly = 0.0
    masc = 0.0
    age = 0.0
    mood = 0.0

    def add_event(color, v):
        nonlocal girly, masc, age, mood
        for b, w in _trilinear_weights(color):
            ww = v * w
            girly += ww * GIRLY_GRID[b]
            masc  += ww * MASC_GRID[b]
            age   += ww * AGE_GRID[b]
            mood  += ww * MOOD_GRID[b]

    for i in range(64):
        if i not in picked:
            add_event(offered[i], NOT_PICK_VALUE)
    for c in r1:
        add_event(c, PICK_VALUE)
    for c in r2:
        add_event(c, PICK_VALUE)
    add_event(final, PICK_VALUE)

    return girly, masc, girly - masc, age, mood


# ---------- public entry point ----------

def compute_features(payload: Dict, submit_unix: int) -> Dict[str, List[float]]:
    """Build the three feature vectors the LightGBM heads expect.

    payload schema:
        offered: list of 64 [r,g,b] triples
        r1:      list of 16 [r,g,b] triples
        r2:      list of 4 [r,g,b] triples
        final:   [r,g,b]
        valg:    21 ASCII digits
        tider:   21 cumulative ms timestamps

    Returns:
        {"gender": [..477..], "age": [..475..], "mood": [..475..]}
    """
    base = _extract_base(payload, submit_unix)
    extras = _extract_extras(payload)
    girly, masc, signed, age_t, mood_t = _compute_bucket_totals(payload)

    combined = base + extras  # 441 + 33 = 474 floats

    return {
        "gender": combined + [girly, masc, signed],
        "age":    combined + [age_t],
        "mood":   combined + [mood_t],
    }


def validate_payload(payload: Dict) -> None:
    required = ("offered", "r1", "r2", "final", "valg", "tider")
    for k in required:
        if k not in payload:
            raise ValueError(f"missing key: {k}")
    if len(payload["offered"]) != 64:
        raise ValueError(f"offered must have 64 entries, got {len(payload['offered'])}")
    if len(payload["r1"]) != 16:
        raise ValueError(f"r1 must have 16 entries, got {len(payload['r1'])}")
    if len(payload["r2"]) != 4:
        raise ValueError(f"r2 must have 4 entries, got {len(payload['r2'])}")
    if len(payload["final"]) != 3:
        raise ValueError(f"final must be [r,g,b]")
    if len(payload["valg"]) < N_QUESTIONS:
        raise ValueError(f"valg must be {N_QUESTIONS} digits")
    if len(payload["tider"]) < N_QUESTIONS:
        raise ValueError(f"tider must have {N_QUESTIONS} entries")
