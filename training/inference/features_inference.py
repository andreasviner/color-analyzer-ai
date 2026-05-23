"""
Feature extraction for inference: turn a single raw survey payload into the
three feature vectors used by the saved LightGBM heads.

Payload schema (what survey.html POSTs):
    {
      "offered": [[r,g,b], ... 64 entries],          # the 64 round-1 swatches
      "r1":      [[r,g,b], ... 16 entries],          # round-1 winners
      "r2":      [[r,g,b], ... 4 entries],           # round-2 winners
      "final":   [r,g,b],                            # the survivor
      "valg":    "012301230123012301230",            # 21 ASCII digits (positions chosen)
      "tider":   [t1, t2, ..., t21],                 # cumulative ms per question
      "submit_unix": 1716480000                      # optional; defaults to now
    }

Returns a dict with keys "gender", "age", "mood", each a (1, N) float32 array
ready to feed into the matching LightGBM booster.

The base 441 features are produced by importing the canonical features.py
script (so any change to engineering stays in one place). The 33 perceptual
extras and the bucket-score features are ported in this file so the inference
module is self-contained and does not depend on the hyphenated package dirs.
"""

import importlib.util
import os
import time
from typing import Dict, List, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.normpath(os.path.join(HERE, ".."))
BUCKET_SCORES_PATH = os.path.normpath(
    os.path.join(TRAINING_DIR, "color-buckets", "bucket_scores.npz")
)

N_QUESTIONS = 21
N_R1 = 16
N_R2 = 4


def _load_canonical_features_module():
    spec = importlib.util.spec_from_file_location(
        "cp_features_canonical", os.path.join(TRAINING_DIR, "features.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FEATURES = _load_canonical_features_module()


def _gamma(c: float) -> float:
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92


def srgb_to_lab(rgb: Sequence[int]):
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


GIRL_PROTOTYPES_RGB = [
    (255, 182, 193),
    (255, 105, 180),
    (200, 100, 200),
    (220,  40,  40),
]
BOY_PROTOTYPES_RGB = [
    ( 40,  60, 220),
    ( 60, 130, 220),
    ( 50, 170,  60),
    ( 60,  60,  60),
]
GIRL_PROTO = np.mean([srgb_to_lab(c) for c in GIRL_PROTOTYPES_RGB], axis=0)
BOY_PROTO  = np.mean([srgb_to_lab(c) for c in BOY_PROTOTYPES_RGB],  axis=0)


def extract_extra(payload: Dict) -> List[float]:
    """Port of extra-features/train.py:extract_extra, working on the payload dict."""
    valg = payload["valg"]
    tider = [int(x) for x in payload["tider"]]
    offered = payload["offered"]
    r1 = payload["r1"]
    r2 = payload["r2"]
    final = payload["final"]

    offered_lab = np.array([srgb_to_lab(c) for c in offered], dtype=np.float32)
    r1_lab      = np.array([srgb_to_lab(c) for c in r1],      dtype=np.float32)
    r2_lab      = np.array([srgb_to_lab(c) for c in r2],      dtype=np.float32)
    final_lab   = np.array(srgb_to_lab(final), dtype=np.float32)

    vals: List[float] = []

    vals.extend([float(final_lab[0]), float(final_lab[1]), float(final_lab[2])])
    vals.extend([
        float(r1_lab[:, 0].mean()), float(r1_lab[:, 1].mean()), float(r1_lab[:, 2].mean()),
        float(r1_lab[:, 0].std()),  float(r1_lab[:, 1].std()),  float(r1_lab[:, 2].std()),
    ])
    vals.extend([
        float(r2_lab[:, 0].mean()), float(r2_lab[:, 1].mean()), float(r2_lab[:, 2].mean()),
        float(r2_lab[:, 0].std()),  float(r2_lab[:, 1].std()),  float(r2_lab[:, 2].std()),
    ])
    vals.extend([
        float(offered_lab[:, 0].mean()), float(offered_lab[:, 1].mean()), float(offered_lab[:, 2].mean()),
        float(offered_lab[:, 0].std()),  float(offered_lab[:, 1].std()),  float(offered_lab[:, 2].std()),
    ])

    d_final_girl = float(np.linalg.norm(final_lab - GIRL_PROTO))
    d_final_boy  = float(np.linalg.norm(final_lab - BOY_PROTO))
    vals.extend([
        d_final_girl, d_final_boy,
        float(np.log((d_final_boy + 1.0) / (d_final_girl + 1.0))),
        float(np.mean(np.linalg.norm(r1_lab - GIRL_PROTO, axis=1))),
        float(np.mean(np.linalg.norm(r1_lab - BOY_PROTO,  axis=1))),
    ])

    diversities: List[float] = []
    relative_decisives: List[float] = []
    for q in range(N_R1):
        group = offered_lab[q * 4:(q + 1) * 4]
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

    vals.extend([
        float(np.mean(diversities)), float(np.std(diversities)),
        float(np.mean(relative_decisives)), float(np.std(relative_decisives)),
    ])

    deltas = [max(0, tider[0])] + [
        max(0, tider[i] - tider[i - 1]) for i in range(1, len(tider))
    ]
    deltas_sec = [d / 1000.0 for d in deltas]
    vals.extend([
        sum(1 for t in deltas_sec if t < 1.0) / len(deltas_sec),
        sum(1 for t in deltas_sec if t > 7.0) / len(deltas_sec),
        float(np.std(deltas) / (np.mean(deltas) + 1.0)),
    ])

    return vals


GRID = 8
N_BUCKETS = GRID ** 3
BUCKET_WIDTH = 256 / GRID
BUCKET_CENTER_OFFSET = BUCKET_WIDTH / 2
PICK_VALUE = 0.1 * 16 * 3 / N_QUESTIONS
NOT_PICK_VALUE = -0.1


def _bucket_id(r_idx: int, g_idx: int, b_idx: int) -> int:
    return r_idx * GRID * GRID + g_idx * GRID + b_idx


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


def compute_bucket_deltas(payload: Dict):
    offered = payload["offered"]
    r1 = payload["r1"]
    r2 = payload["r2"]
    final = payload["final"]
    valg = payload["valg"]

    picked_offered_idx = set()
    for q in range(N_R1):
        try:
            idx = int(valg[q])
            if 0 <= idx <= 3:
                picked_offered_idx.add(q * 4 + idx)
        except (ValueError, IndexError):
            pass

    discrete = np.zeros(N_BUCKETS, dtype=np.float32)
    smooth = np.zeros(N_BUCKETS, dtype=np.float32)

    def add_event(color, value):
        discrete[_discrete_bucket(color)] += value
        for b, w in _trilinear_weights(color):
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


def _fake_row_for_canonical(payload: Dict):
    """Shape the payload into the row tuple features.py:extract_features expects."""
    submit_unix = int(payload.get("submit_unix") or time.time())
    return [
        "session",
        str(submit_unix),
        "ignored",
        "20",
        "30",
        "g",
        payload["valg"],
        [int(x) for x in payload["tider"]],
        [payload["offered"], payload["r1"], payload["r2"], payload["final"]],
    ]


def compute_features(payload: Dict, bucket_scores: Dict = None) -> Dict[str, np.ndarray]:
    """Build (1, N) feature matrices for gender, age and mood heads.

    `bucket_scores` is the dict-like returned by np.load(bucket_scores.npz); if
    omitted it is loaded from the canonical color-buckets path.
    """
    if bucket_scores is None:
        bucket_scores = dict(np.load(BUCKET_SCORES_PATH))

    girly_grid = bucket_scores["girly_grid"].reshape(N_BUCKETS).astype(np.float32)
    masc_grid  = bucket_scores["masc_grid"].reshape(N_BUCKETS).astype(np.float32)
    age_grid   = bucket_scores["age_grid"].reshape(N_BUCKETS).astype(np.float32)
    mood_grid  = bucket_scores["mood_grid"].reshape(N_BUCKETS).astype(np.float32)

    row = _fake_row_for_canonical(payload)
    _names, base_vals, _g, _a, _m = _FEATURES.extract_features(row)
    base = np.asarray(base_vals, dtype=np.float32)

    extra = np.asarray(extract_extra(payload), dtype=np.float32)
    combined = np.concatenate([base, extra])  # (474,)

    _discrete, smooth = compute_bucket_deltas(payload)
    girly_total  = float(smooth @ girly_grid)
    masc_total   = float(smooth @ masc_grid)
    signed_total = girly_total - masc_total
    age_total    = float(smooth @ age_grid)
    mood_total   = float(smooth @ mood_grid)

    x_gender = np.concatenate([
        combined,
        np.array([girly_total, masc_total, signed_total], dtype=np.float32),
    ])[None, :]
    x_age = np.concatenate([combined, np.array([age_total], dtype=np.float32)])[None, :]
    x_mood = np.concatenate([combined, np.array([mood_total], dtype=np.float32)])[None, :]

    return {"gender": x_gender, "age": x_age, "mood": x_mood}
