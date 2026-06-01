"""
Pure-Python feature extraction for the *long* (256-color) Color Polygraph
survey, mirroring `features.py` for the short survey.

The long bracket is a power-of-four funnel one tier deeper than the short one:

    offered(256) -> r1(64) -> r2(16) -> r3(4) -> final(1),  85 questions.

Equivalently, a long session is four short sessions stacked plus one final
question that picks among the four short winners. The long models are trained
on synthetic long sessions built exactly that way (see
`training/long-models/train_long.py`), so the feature definitions here must
match between training and serving — which is why this single module is used
on BOTH sides.

The design follows `features.py` closely (per-stage colour stats, selectivity
deltas, decisiveness/extreme-pick aggregates, voxel histogram, reference-colour
distances, trajectory slopes, timing, LAB extras, RGB bucket-score totals) but:
  * stages are offered/r1/r2/r3 instead of offered/r1/r2,
  * the per-question round-0 features are kept as AGGREGATES only (there are 64
    round-0 questions; emitting a colour block per question would blow up the
    feature/sample ratio on the smaller synthetic set), and
  * the per-round timing block has four rounds.

Output: three flat vectors {"gender": [...], "age": [...], "mood": [...]} in the
exact column order the long LightGBM heads expect.
"""

import math
from typing import Dict, List, Sequence

# Reuse the short module's numeric + colour helpers so the two extractors can
# never drift on shared primitives. Importing it also pulls in the short
# bucket_data (8x8x8 geometry), which is identical to the long grid geometry.
from features import (
    _mean, _std, _norm, _polyfit_slope, _argmin, _argmax,
    _reference_distances, _rgb_to_hsl, _rgb_to_yuv, _color_stats,
    _voxel_id, _hour_minute, _srgb_to_lab, _lab_distance,
    _GIRL_PROTO, _BOY_PROTO, _trilinear_weights, _discrete_bucket,
    TIME_BUCKETS_SEC,
)

# ---------- long bracket constants ----------

N_LONG_OFFERED = 256
N_LONG_R0Q = 64        # round-0 questions (offered -> r1)
N_LONG_R1 = 64         # round-0 winners
N_LONG_R2 = 16         # round-1 winners
N_LONG_R3 = 4          # round-2 winners
N_LONG_QUESTIONS = 85
VOXEL_GRID = 4         # 4x4x4 = 64 buckets for the r1 pick histogram
N_BUCKETS = 512        # 8x8x8 RGB grid (same geometry as the short model)

# Pick / not-pick event weights for the bucket-score deltas. Mirrors the short
# balance (sum of pick weight == magnitude of not-pick weight): 192 unpicked
# offered colours at -0.1 vs 85 picks. The exact scale is irrelevant to the
# trees as long as training and serving agree.
LONG_NOT_PICK_VALUE = -0.1
LONG_PICK_VALUE = 0.1 * (N_LONG_OFFERED - N_LONG_R1) / N_LONG_QUESTIONS


# ---------- static feature vector ----------

def _chan_mean(colors: Sequence[Sequence[int]]):
    n = len(colors)
    return (
        sum(c[0] for c in colors) / (255.0 * n),
        sum(c[1] for c in colors) / (255.0 * n),
        sum(c[2] for c in colors) / (255.0 * n),
    )


def _extract_static_long(payload: Dict, submit_unix: int) -> List[float]:
    valg = payload["valg"]
    tider = [int(x) for x in payload["tider"]]
    offered = payload["offered"]   # 256
    r1 = payload["r1"]             # 64
    r2 = payload["r2"]             # 16
    r3 = payload["r3"]             # 4
    final = payload["final"]

    vals: List[float] = []

    # ---- Per-stage colour summary stats (4 x 12) ----
    for colors in (offered, r1, r2, r3):
        vals.extend(_color_stats(colors))

    # ---- Final colour: RGB, HSL, YUV, warmth, chroma ----
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

    # ---- Selectivity deltas between consecutive stages (per channel) ----
    om = _chan_mean(offered)
    m1 = _chan_mean(r1)
    m2 = _chan_mean(r2)
    m3 = _chan_mean(r3)
    vals.extend([
        m1[0] - om[0], m1[1] - om[1], m1[2] - om[2],
        m2[0] - m1[0], m2[1] - m1[1], m2[2] - m1[2],
        m3[0] - m2[0], m3[1] - m2[1], m3[2] - m2[2],
    ])

    # ---- Voxel diversity per stage ----
    vals.append(len({_voxel_id(c, 8) for c in r1}) / N_LONG_R1)
    vals.append(len({_voxel_id(c, 8) for c in r2}) / N_LONG_R2)
    vals.append(len({_voxel_id(c, 8) for c in r3}) / N_LONG_R3)
    vals.append(len({_voxel_id(c, 8) for c in offered}) / N_LONG_OFFERED)

    # ---- Internal spread of the 64 round-0 winners ----
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

    # ---- Round-0 decisiveness + position + extreme-pick aggregates (64 q) ----
    decisive: List[float] = []
    pos_counts = [0, 0, 0, 0]
    extreme = {"warmest": 0, "coolest": 0, "lightest": 0,
               "darkest": 0, "most_sat": 0, "least_sat": 0}
    for q in range(N_LONG_R0Q):
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
        dr = chosen[0] / 255.0 - rej_r
        dg = chosen[1] / 255.0 - rej_g
        db = chosen[2] / 255.0 - rej_b
        decisive.append(math.sqrt(dr * dr + dg * dg + db * db))
        pos_counts[idx] += 1

        warmths = [c[0] - c[2] for c in group]
        lights, sats = [], []
        for c in group:
            _, s, l = _rgb_to_hsl(c)
            lights.append(l)
            sats.append(s)
        if idx == _argmax(warmths): extreme["warmest"]   += 1
        if idx == _argmin(warmths): extreme["coolest"]   += 1
        if idx == _argmax(lights):  extreme["lightest"]  += 1
        if idx == _argmin(lights):  extreme["darkest"]   += 1
        if idx == _argmax(sats):    extreme["most_sat"]  += 1
        if idx == _argmin(sats):    extreme["least_sat"] += 1

    vals.append(_mean(decisive))
    vals.append(_std(decisive))
    for p in range(4):
        vals.append(pos_counts[p] / N_LONG_R0Q)
    probs = [c / N_LONG_R0Q for c in pos_counts if c > 0]
    pos_entropy = -sum(p * math.log(p) for p in probs) if probs else 0.0
    vals.append(pos_entropy)
    for k in ("warmest", "coolest", "lightest", "darkest", "most_sat", "least_sat"):
        vals.append(extreme[k] / N_LONG_R0Q)

    # ---- Voxel histogram of the 64 round-0 winners (4x4x4 = 64 buckets) ----
    hist = [0.0] * (VOXEL_GRID ** 3)
    for c in r1:
        v = _voxel_id(c, VOXEL_GRID)
        hist[v[0] * VOXEL_GRID * VOXEL_GRID + v[1] * VOXEL_GRID + v[2]] += 1
    for n in hist:
        vals.append(n / N_LONG_R1)

    # ---- Reference colour distances of the final pick ----
    dists = _reference_distances(final)
    vals.extend(dists)
    vals.append(_argmin(dists))

    # ---- Distance from final pick to each stage mean ----
    fn = (final[0] / 255.0, final[1] / 255.0, final[2] / 255.0)
    vals.append(_norm((fn[0] - m1[0], fn[1] - m1[1], fn[2] - m1[2])))
    vals.append(_norm((fn[0] - m2[0], fn[1] - m2[1], fn[2] - m2[2])))
    vals.append(_norm((fn[0] - m3[0], fn[1] - m3[1], fn[2] - m3[2])))

    # ---- Trajectory across the 64 round-0 winners ----
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
    cent_d = [math.sqrt((c[0] - cx) ** 2 + (c[1] - cy) ** 2 + (c[2] - cz) ** 2) for c in r1_norm]
    vals.append(_mean(cent_d))
    warmth_r1 = [c[0] - c[2] for c in r1_norm]
    light_r1 = [_rgb_to_hsl([c[0] * 255, c[1] * 255, c[2] * 255])[2] for c in r1_norm]
    sat_r1   = [_rgb_to_hsl([c[0] * 255, c[1] * 255, c[2] * 255])[1] for c in r1_norm]
    xs = [float(i) for i in range(n_r1)]
    vals.append(_polyfit_slope(xs, warmth_r1))
    vals.append(_polyfit_slope(xs, light_r1))
    vals.append(_polyfit_slope(xs, sat_r1))

    # ---- Timing ----
    deltas_ms = [max(0, tider[0])] + [
        max(0, tider[i] - tider[i - 1]) for i in range(1, len(tider))
    ]
    total_ms = tider[-1]
    vals.append(total_ms / 1000.0)
    mean_q_ms = _mean(deltas_ms)
    vals.append(mean_q_ms)
    vals.append(_std(deltas_ms))
    vals.append(float(min(deltas_ms)))
    vals.append(float(max(deltas_ms)))
    sorted_d = sorted(deltas_ms)
    n_d = len(sorted_d)
    vals.append((sorted_d[n_d // 2] + sorted_d[(n_d - 1) // 2]) / 2.0)
    vals.append(_mean(deltas_ms[:5]))
    vals.append(_mean(deltas_ms[-5:]))
    xs_t = [float(i) for i in range(len(deltas_ms))]
    vals.append(_polyfit_slope(xs_t, [float(d) for d in deltas_ms]))
    # per-round mean times
    vals.append(_mean(deltas_ms[:N_LONG_R0Q]))
    vals.append(_mean(deltas_ms[N_LONG_R0Q:N_LONG_R0Q + N_LONG_R2]))
    vals.append(_mean(deltas_ms[N_LONG_R0Q + N_LONG_R2:N_LONG_R0Q + N_LONG_R2 + N_LONG_R3]))
    vals.append(float(deltas_ms[-1]))

    # ---- Round-0 time-bucket histogram ----
    r0_sec = [deltas_ms[q] / 1000.0 for q in range(N_LONG_R0Q)]
    bucket_counts = [0] * len(TIME_BUCKETS_SEC)
    for t in r0_sec:
        for i, (lo, hi) in enumerate(TIME_BUCKETS_SEC):
            if lo <= t < hi:
                bucket_counts[i] += 1
                break
    for n in bucket_counts:
        vals.append(n / N_LONG_R0Q)

    # ---- Time x behaviour interactions ----
    vals.append(mean_q_ms * (1.0 - pos_entropy / math.log(4)))
    vals.append(mean_q_ms / (_mean(decisive) + 50.0))

    # ---- Hour of day ----
    h, mn = _hour_minute(submit_unix)
    if h is None:
        vals.extend([0.0, 0.0, 0.0])
    else:
        frac = (h + mn / 60.0) / 24.0
        vals.append(math.sin(2 * math.pi * frac))
        vals.append(math.cos(2 * math.pi * frac))
        vals.append(1.0)

    # ---- LAB extras ----
    final_lab = _srgb_to_lab(final)
    vals.extend([final_lab[0], final_lab[1], final_lab[2]])
    for stage in (r1, r2, r3, offered):
        labs = [_srgb_to_lab(c) for c in stage]
        Ls = [c[0] for c in labs]
        As = [c[1] for c in labs]
        Bs = [c[2] for c in labs]
        vals.extend([_mean(Ls), _mean(As), _mean(Bs), _std(Ls), _std(As), _std(Bs)])

    # gender-prototype distances
    d_girl = _lab_distance(final_lab, _GIRL_PROTO)
    d_boy = _lab_distance(final_lab, _BOY_PROTO)
    vals.append(d_girl)
    vals.append(d_boy)
    vals.append(math.log((d_boy + 1.0) / (d_girl + 1.0)))
    r1_lab = [_srgb_to_lab(c) for c in r1]
    vals.append(_mean([_lab_distance(c, _GIRL_PROTO) for c in r1_lab]))
    vals.append(_mean([_lab_distance(c, _BOY_PROTO) for c in r1_lab]))

    # difficulty + relative decisiveness over the 64 round-0 questions (LAB)
    offered_lab = [_srgb_to_lab(c) for c in offered]
    diversities, rel_dec = [], []
    for q in range(N_LONG_R0Q):
        group = offered_lab[q * 4:(q + 1) * 4]
        total = 0.0
        cnt = 0
        for i in range(4):
            for j in range(i + 1, 4):
                total += _lab_distance(group[i], group[j])
                cnt += 1
        diversity = total / cnt if cnt else 0.0
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
        delta = math.sqrt((chosen[0] - ra) ** 2 + (chosen[1] - rb) ** 2 + (chosen[2] - rc) ** 2)
        rel_dec.append(delta / (diversity + 1.0))
    vals.extend([_mean(diversities), _std(diversities), _mean(rel_dec), _std(rel_dec)])

    # time anomalies
    deltas_sec = [d / 1000.0 for d in deltas_ms]
    rushed = sum(1 for t in deltas_sec if t < 1.0) / len(deltas_sec)
    dwelled = sum(1 for t in deltas_sec if t > 7.0) / len(deltas_sec)
    cv = _std(deltas_ms) / (_mean(deltas_ms) + 1.0)
    vals.extend([rushed, dwelled, cv])

    return vals


# ---------- bucket-score deltas / totals ----------

def _long_picked_set(valg) -> set:
    picked = set()
    for q in range(N_LONG_R0Q):
        try:
            idx = int(valg[q])
            if 0 <= idx <= 3:
                picked.add(q * 4 + idx)
        except (ValueError, IndexError):
            pass
    return picked


def compute_bucket_delta_long(payload: Dict):
    """Return (discrete[512], smooth[512]) event vectors for one long session.
    `discrete` (nearest-bucket) builds the per-trait grids at training time;
    `smooth` (trilinear) is dotted with a grid to produce a session total."""
    offered = payload["offered"]
    r1 = payload["r1"]
    r2 = payload["r2"]
    r3 = payload["r3"]
    final = payload["final"]
    picked = _long_picked_set(payload["valg"])

    discrete = [0.0] * N_BUCKETS
    smooth = [0.0] * N_BUCKETS

    def add_event(color, v):
        discrete[_discrete_bucket(color)] += v
        for b, w in _trilinear_weights(color):
            smooth[b] += v * w

    for i in range(N_LONG_OFFERED):
        if i not in picked:
            add_event(offered[i], LONG_NOT_PICK_VALUE)
    for c in r1:
        add_event(c, LONG_PICK_VALUE)
    for c in r2:
        add_event(c, LONG_PICK_VALUE)
    for c in r3:
        add_event(c, LONG_PICK_VALUE)
    add_event(final, LONG_PICK_VALUE)
    return discrete, smooth


def compute_bucket_totals_long(payload: Dict, grids):
    """(girly_total, masc_total, signed_total, age_total, mood_total) for one
    session against the four precomputed long grids."""
    girly_grid, masc_grid, age_grid, mood_grid = grids
    _, smooth = compute_bucket_delta_long(payload)
    girly = masc = age = mood = 0.0
    for b in range(N_BUCKETS):
        w = smooth[b]
        if w == 0.0:
            continue
        girly += w * girly_grid[b]
        masc  += w * masc_grid[b]
        age   += w * age_grid[b]
        mood  += w * mood_grid[b]
    return girly, masc, girly - masc, age, mood


# ---------- public serving entry point ----------

# The long grids are emitted by the training run. Import lazily so this module
# can be imported (by the trainer) before the grids exist.
try:
    import bucket_data_long as _bdl
    _LONG_GRIDS = (_bdl.GIRLY_GRID_LONG, _bdl.MASC_GRID_LONG,
                   _bdl.AGE_GRID_LONG, _bdl.MOOD_GRID_LONG)
except Exception:
    _LONG_GRIDS = None


def compute_features_long(payload: Dict, submit_unix: int) -> Dict[str, List[float]]:
    """Build the three long feature vectors. Mirrors `compute_features` in
    features.py: a shared static block plus per-head bucket-score totals."""
    if _LONG_GRIDS is None:
        raise RuntimeError("bucket_data_long not available; run train_long.py first")
    static = _extract_static_long(payload, submit_unix)
    girly, masc, signed, age_t, mood_t = compute_bucket_totals_long(payload, _LONG_GRIDS)
    return {
        "gender": static + [girly, masc, signed],
        "age":    static + [age_t],
        "mood":   static + [mood_t],
    }
