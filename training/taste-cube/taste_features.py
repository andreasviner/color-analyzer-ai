"""
Feature extraction for the personal "taste cube" model (short survey).

This is the single source of truth for the model's inputs at TRAIN time. A JS
mirror (models-js/taste_features.js) reproduces the same math bit-for-bit at
serve time; train_taste.py emits taste_parity.json so the JS port can be
checked against this file within 1e-5.

The model is a binary classifier: given a summary of how a person picked colors
(their "fingerprint", built only from the colours they kept choosing) plus a
single candidate colour, it predicts the probability that this person would
pick that candidate. The raw logit doubles as a per-colour desirability score,
so at serve time we score the 512 voxel centres once and Monte-Carlo synthetic
quads to fill the same [offered, r1, r2, final] arrays the population cube reads.

Design constraints that shape the feature set:
  * Fingerprint uses WINNERS ONLY (r1 / r2 / final colours). The shared-link
    result path has no round-0 offered options, but it always has the winners,
    so a winner-only fingerprint works on both the fresh and ?id= paths.
  * HSL and LAB are implemented explicitly here (not via colorsys) so the JS
    mirror can match the exact formula.
"""

import math
from typing import Dict, List, Sequence, Tuple

# Same 12 reference colours features.py uses, so the cube speaks the same
# vocabulary of named hues as the rest of the project.
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

N_R0 = 16  # round-0 questions in a short survey

# How many probe questions (= training samples) to draw from each person. A
# short survey has up to ~12 eligible "loser" questions; we take an evenly
# spread subset so no single person dominates the training set. Easy to edit.
# Sweep (tune_taste.py) showed holdout pick-accuracy climbs then plateaus:
# 2->0.469, 3->0.484, 5->0.491, 8->0.492, 12->0.494. 10 sits on the plateau.
PROBES_PER_SESSION = 10


# ---------- small numeric helpers ----------

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _polyfit_slope(ys: Sequence[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2.0
    my = _mean(ys)
    num = sum((i - mx) * (ys[i] - my) for i in range(n))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else 0.0


# ---------- colour space conversions (explicit, JS-mirrored) ----------

def _rgb_to_hsl(rgb: Sequence[int]) -> Tuple[float, float, float]:
    """Standard HSL, every component in [0, 1]."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    l = (mx + mn) / 2.0
    if mx == mn:
        return 0.0, 0.0, l
    d = mx - mn
    s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == r:
        h = ((g - b) / d) % 6.0
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return h / 6.0, s, l


def _gamma(c: float) -> float:
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92


def _srgb_to_lab(rgb: Sequence[int]) -> Tuple[float, float, float]:
    """sRGB -> CIE L*a*b* (D65). Same coefficients as cloudflare/features.py."""
    r = _gamma(rgb[0] / 255.0)
    g = _gamma(rgb[1] / 255.0)
    b = _gamma(rgb[2] / 255.0)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / 1.00000
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t):
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def _lab_dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _rgb_dist_norm(a, b) -> float:
    """Euclidean distance in 0..1 RGB."""
    return math.sqrt(
        ((a[0] - b[0]) / 255.0) ** 2
        + ((a[1] - b[1]) / 255.0) ** 2
        + ((a[2] - b[2]) / 255.0) ** 2
    )


def _hue_circ_diff(h1: float, h2: float) -> float:
    """Shortest distance between two hues (both in [0,1]); result in [0, 0.5]."""
    d = abs(h1 - h2) % 1.0
    return min(d, 1.0 - d)


def _ref_distances(rgb: Sequence[int]) -> List[float]:
    rn, gn, bn = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    out = []
    for (r, g, b) in REFERENCE_NORM:
        out.append(math.sqrt((r - rn) ** 2 + (g - gn) ** 2 + (b - bn) ** 2))
    return out


def _argmin(xs: Sequence[float]) -> int:
    best, bv = 0, xs[0]
    for i in range(1, len(xs)):
        if xs[i] < bv:
            bv, best = xs[i], i
    return best


# ---------- person fingerprint (winners only) ----------

def session_context(r1_winners: Sequence[Sequence[int]],
                    r2_winners: Sequence[Sequence[int]],
                    final: Sequence[int]) -> Dict:
    """Pre-compute everything the fingerprint and interaction features need from
    a person's winning colours. `r1_winners` is the (possibly probe-adjusted)
    list of round-0 winners; `r2_winners` the round-1 winners; `final` the
    overall pick."""
    r1 = [list(c) for c in r1_winners]
    r2 = [list(c) for c in r2_winners]
    fin = list(final)

    r1_hsl = [_rgb_to_hsl(c) for c in r1]
    r1_lab = [_srgb_to_lab(c) for c in r1]

    def comp(colors, idx, conv=None):
        if conv is None:
            return [c[idx] / 255.0 for c in colors]
        return [conv(c)[idx] for c in colors]

    r1_r = comp(r1, 0)
    r1_g = comp(r1, 1)
    r1_b = comp(r1, 2)
    r1_h = [c[0] for c in r1_hsl]
    r1_s = [c[1] for c in r1_hsl]
    r1_l = [c[2] for c in r1_hsl]

    # circular mean hue (preferred hue)
    sin_sum = sum(math.sin(2 * math.pi * h) for h in r1_h)
    cos_sum = sum(math.cos(2 * math.pi * h) for h in r1_h)
    pref_hue = (math.atan2(sin_sum, cos_sum) / (2 * math.pi)) % 1.0

    r1_mean_rgb = [_mean(r1_r) * 255.0, _mean(r1_g) * 255.0, _mean(r1_b) * 255.0]
    r1_mean_lab = [_mean([c[0] for c in r1_lab]),
                   _mean([c[1] for c in r1_lab]),
                   _mean([c[2] for c in r1_lab])]
    r2_mean_rgb = [_mean([c[0] for c in r2]), _mean([c[1] for c in r2]),
                   _mean([c[2] for c in r2])] if r2 else list(r1_mean_rgb)

    warmth = [(c[0] - c[2]) / 255.0 for c in r1]
    fin_lab = _srgb_to_lab(fin)
    fin_hsl = _rgb_to_hsl(fin)

    r2_lab = [_srgb_to_lab(c) for c in r2] if r2 else [r1_mean_lab]
    r2_mean_lab = [_mean([c[0] for c in r2_lab]),
                   _mean([c[1] for c in r2_lab]),
                   _mean([c[2] for c in r2_lab])]

    # voxel diversity: unique 8x8x8 cells among the round-0 winners
    vox = {((c[0] >> 5), (c[1] >> 5), (c[2] >> 5)) for c in r1}

    # internal spread (mean pairwise RGB distance) of round-0 winners
    spread, npair = 0.0, 0
    for i in range(len(r1)):
        for j in range(i + 1, len(r1)):
            spread += _rgb_dist_norm(r1[i], r1[j])
            npair += 1
    spread = spread / npair if npair else 0.0

    return {
        "r1": r1, "r1_lab": r1_lab,
        "r1_r": r1_r, "r1_g": r1_g, "r1_b": r1_b,
        "r1_h": r1_h, "r1_s": r1_s, "r1_l": r1_l,
        "r2": r2,
        "final": fin, "fin_lab": fin_lab, "fin_hsl": fin_hsl,
        "warmth": warmth,
        "pref_hue": pref_hue,
        "mean_sat": _mean(r1_s),
        "mean_light": _mean(r1_l),
        "mean_warmth": _mean(warmth),
        "r1_mean_rgb": r1_mean_rgb,
        "r1_mean_lab": r1_mean_lab,
        "r2_mean_rgb": r2_mean_rgb,
        "r2_mean_lab": r2_mean_lab,
        "vox_div": len(vox) / max(1, len(r1)),
        "spread": spread,
        "fin_ref_argmin": _argmin(_ref_distances(fin)),
    }


def fingerprint_vector(ctx: Dict) -> List[float]:
    """~34 floats summarising the person's taste. Winners only."""
    r1_r, r1_g, r1_b = ctx["r1_r"], ctx["r1_g"], ctx["r1_b"]
    r1_h, r1_s, r1_l = ctx["r1_h"], ctx["r1_s"], ctx["r1_l"]
    r2 = ctx["r2"]
    fin = ctx["final"]
    fh, fs, fl = ctx["fin_hsl"]
    flab = ctx["fin_lab"]

    out: List[float] = []
    # round-0 winner stats: mean + std of R,G,B,H,S,L
    out += [_mean(r1_r), _mean(r1_g), _mean(r1_b),
            _std(r1_r),  _std(r1_g),  _std(r1_b),
            _mean(r1_h), _mean(r1_s), _mean(r1_l),
            _std(r1_h),  _std(r1_s),  _std(r1_l)]
    # round-1 winner means (4 colours)
    if r2:
        out += [_mean([c[0] for c in r2]) / 255.0,
                _mean([c[1] for c in r2]) / 255.0,
                _mean([c[2] for c in r2]) / 255.0]
        r2_hsl = [_rgb_to_hsl(c) for c in r2]
        out += [_mean([c[0] for c in r2_hsl]),
                _mean([c[1] for c in r2_hsl]),
                _mean([c[2] for c in r2_hsl])]
    else:
        out += [0.0] * 6
    # final colour
    out += [fin[0] / 255.0, fin[1] / 255.0, fin[2] / 255.0,
            fh, fs, fl,
            flab[0], flab[1], flab[2],
            (fin[0] - fin[2]) / 255.0,
            (max(fin) - min(fin)) / 255.0]
    # taste trajectory + shape
    out += [_polyfit_slope(ctx["warmth"]),
            _polyfit_slope(r1_l),
            _polyfit_slope(r1_s),
            ctx["spread"],
            ctx["vox_div"]]
    return out


def candidate_vector(rgb: Sequence[int]) -> List[float]:
    """~23 floats describing one colour on its own."""
    h, s, l = _rgb_to_hsl(rgb)
    lab = _srgb_to_lab(rgb)
    out = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0,
           h, s, l,
           (rgb[0] - rgb[2]) / 255.0,
           (max(rgb) - min(rgb)) / 255.0,
           lab[0], lab[1], lab[2]]
    out += _ref_distances(rgb)
    return out


def interaction_vector(rgb: Sequence[int], ctx: Dict) -> List[float]:
    """~11 floats pairing the candidate with this person's taste."""
    h, s, l = _rgb_to_hsl(rgb)
    lab = _srgb_to_lab(rgb)
    refs = _ref_distances(rgb)
    min_lab_to_winner = min((_lab_dist(lab, wl) for wl in ctx["r1_lab"]), default=0.0)
    min_rgb_to_winner = min((_rgb_dist_norm(rgb, w) for w in ctx["r1"]), default=0.0)
    warmth = (rgb[0] - rgb[2]) / 255.0
    signed_hue = ((h - ctx["pref_hue"] + 0.5) % 1.0) - 0.5
    return [
        _rgb_dist_norm(rgb, ctx["r1_mean_rgb"]),
        _rgb_dist_norm(rgb, ctx["r2_mean_rgb"]),
        _rgb_dist_norm(rgb, ctx["final"]),
        _lab_dist(lab, ctx["r1_mean_lab"]),
        _lab_dist(lab, ctx["fin_lab"]),
        min_lab_to_winner,
        _hue_circ_diff(h, ctx["pref_hue"]),
        s - ctx["mean_sat"],
        l - ctx["mean_light"],
        refs[ctx["fin_ref_argmin"]],
        1.0 if _argmin(refs) == ctx["fin_ref_argmin"] else 0.0,
        # --- added interactions (the ablation showed interactions carry the model) ---
        signed_hue,
        _hue_circ_diff(h, ctx["fin_hsl"][0]),
        min_rgb_to_winner,
        _lab_dist(lab, ctx["r2_mean_lab"]),
        warmth - ctx["mean_warmth"],
        min_lab_to_winner / (ctx["spread"] + 0.05),
    ]


def feature_row(ctx: Dict, rgb: Sequence[int]) -> List[float]:
    return fingerprint_vector(ctx) + candidate_vector(rgb) + interaction_vector(rgb, ctx)


# ---------- probe-row construction for training ----------

def _color_in(color, pool) -> bool:
    c = list(color)
    return any(list(p) == c for p in pool)


def build_probe_rows(session: Dict, n_probes: int = PROBES_PER_SESSION):
    """Yield (row, label, group_id) tuples for one session.

    From the round-0 questions whose winner did NOT advance to round 1 (the
    "loser" questions -- their winner is absent from r2 and final), we take an
    evenly spread subset of up to `n_probes` and make each a probe:
      * overwrite the probe's winner slot with a duplicate of the next question
        so the fingerprint stays 16-winners-wide but no longer contains the
        probe's own pick (leakage-safe),
      * build the fingerprint from those adjusted winners + r2 + final
        (r2/final are safe precisely because this is a loser question),
      * emit one row per candidate colour in the probe's offered quad, labelled
        1 for the colour the person actually picked.

    `session` keys: offered (64x rgb), r1 (16x rgb winners), r2 (4x rgb),
    final (rgb), valg (>=16 pick digits).
    """
    offered = [list(c) for c in session["offered"][:64]]
    r1 = [list(c) for c in session["r1"][:N_R0]]
    r2 = [list(c) for c in session["r2"][:4]]
    final = list(session["final"])
    valg = session["valg"]

    advanced = list(r2) + [final]  # colours that made it past round 0

    # Collect eligible "loser" questions, then keep an evenly spread subset so a
    # single person contributes at most `n_probes` samples.
    eligible = []
    for q in range(N_R0):
        if _color_in(r1[q], advanced):
            continue  # not a loser question; skip to keep r2/final leak-free
        try:
            pick = int(valg[q])
        except (ValueError, IndexError):
            continue
        if not (0 <= pick <= 3):
            continue
        if len(offered[q * 4:(q + 1) * 4]) != 4:
            continue
        eligible.append((q, pick))

    if n_probes and len(eligible) > n_probes:
        # even spread across the eligible list (deterministic, no RNG)
        step = len(eligible) / n_probes
        eligible = [eligible[int(i * step)] for i in range(n_probes)]

    rows = []
    sid = session.get("id", id(session))
    for q, pick in eligible:
        quad = offered[q * 4:(q + 1) * 4]

        # Overwrite the probe's winner with a duplicate of another question's
        # winner, so the fingerprint is 16-wide but drops the probe's own pick.
        donor = (q + 1) % N_R0
        adj_r1 = list(r1)
        adj_r1[q] = r1[donor]
        ctx = session_context(adj_r1, r2, final)
        fp = fingerprint_vector(ctx)

        group_id = (sid, q)
        for idx in range(4):
            row = fp + candidate_vector(quad[idx]) + interaction_vector(quad[idx], ctx)
            rows.append((row, 1 if idx == pick else 0, group_id))
    return rows


# ---------- feature layout (for debugging + JS parity) ----------

def feature_layout() -> Dict[str, int]:
    """Sizes of each block, so the JS mirror can assert the same total."""
    # Build a dummy context to measure block sizes exactly.
    dummy_r1 = [[i, i, i] for i in range(16)]
    ctx = session_context(dummy_r1, [[10, 20, 30]] * 4, [40, 50, 60])
    return {
        "fingerprint": len(fingerprint_vector(ctx)),
        "candidate": len(candidate_vector([10, 20, 30])),
        "interaction": len(interaction_vector([10, 20, 30], ctx)),
        "total": len(feature_row(ctx, [10, 20, 30])),
    }
