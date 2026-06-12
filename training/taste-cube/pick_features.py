"""
Row construction for the colour-pick model.

Goal of the model: a person took the survey; show them 4 NEW colours; predict
which one they would pick.

Person representation = the PROD feature vector (cloudflare/features.py), the
exact same 479 floats the result page already gets back from the worker for the
gender/age/mood predictions:

    person = features["gender"] (477) + [features["age"][-1], features["mood"][-1]]

Leakage-free probe construction (the user's overwrite scheme): pick a round-0
question whose winner did NOT advance to round 1 (a "loser" question -- safe,
because removing it has no effect further down the bracket). Overwrite it with
a duplicate of ANOTHER loser question: its offered quad, its r1 winner and its
valg digit are all replaced. The probe question is then effectively absent from
the dataset, so the prod features computed from the modified session cannot
contain its answer. The probe's own quad becomes the "4 new colours" and the
label is the colour the person actually picked.

Candidate representation = colour descriptors we can compute CLIENT-side (no
server bucket grids): RGB, HSL, CMYK, LAB, YUV, hue sin/cos, warmth, chroma,
12 reference-colour distances.

Optionally an interaction block (candidate vs the person's winning colours,
all client-computable from the history) -- train_pick.py trains with and
without it to measure whether it still helps on top of prod features.
"""

import math
import os
import sys
from typing import Dict, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
CF_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "cloudflare"))
sys.path.insert(0, CF_DIR)
import features as prod  # cloudflare/features.py, the real prod extractor  # noqa: E402

# Reuse the explicit (JS-mirrorable) colour math + interaction block from the
# fingerprint experiment; the interaction features proved their worth there.
import taste_features as tfeat  # noqa: E402

N_R0 = 16

# Probe questions (= training samples) per survey-taker. Sweeps on the
# fingerprint model showed accuracy plateaus around 8-12; prod-feature
# extraction is the cost driver here, so default to 8. Easy to edit.
PROBES_PER_SESSION = 8

PERSON_LEN = 479  # 474 combined + girly/masc/signed + age_total + mood_total


# ---------- candidate colour descriptors (client-side computable) ----------

def _rgb_to_cmyk(rgb: Sequence[int]) -> List[float]:
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    k = 1.0 - max(r, g, b)
    if k >= 1.0:
        return [0.0, 0.0, 0.0, 1.0]
    c = (1.0 - r - k) / (1.0 - k)
    m = (1.0 - g - k) / (1.0 - k)
    y = (1.0 - b - k) / (1.0 - k)
    return [c, m, y, k]


def _rgb_to_yuv(rgb: Sequence[int]) -> List[float]:
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    return [
        0.299 * r + 0.587 * g + 0.114 * b,
        -0.169 * r - 0.331 * g + 0.500 * b,
        0.500 * r - 0.419 * g - 0.081 * b,
    ]


def candidate_vector(rgb: Sequence[int]) -> List[float]:
    """~35 floats describing one new colour, all computable in the browser."""
    h, s, l = tfeat._rgb_to_hsl(rgb)
    lab = tfeat._srgb_to_lab(rgb)
    out = [
        rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0,
        h, s, l,
        math.sin(2 * math.pi * h), math.cos(2 * math.pi * h),
        (rgb[0] - rgb[2]) / 255.0,                       # warmth
        (max(rgb) - min(rgb)) / 255.0,                   # chroma / range
        lab[0], lab[1], lab[2],
    ]
    out += _rgb_to_cmyk(rgb)
    out += _rgb_to_yuv(rgb)
    out += tfeat._ref_distances(rgb)
    out.append(float(tfeat._argmin(tfeat._ref_distances(rgb))))
    return out


def interaction_vector(rgb: Sequence[int], ctx: Dict) -> List[float]:
    """Candidate vs this person's winning colours (client-computable)."""
    return tfeat.interaction_vector(rgb, ctx)


# ---------- probe construction ----------

def eligible_probes(session: Dict) -> List[int]:
    """Round-0 questions whose winner did not advance (absent from r2)."""
    r1 = session["r1"]
    r2 = [list(c) for c in session["r2"]]
    out = []
    for q in range(N_R0):
        if tfeat._color_in(r1[q], r2):
            continue
        try:
            pick = int(session["valg"][q])
        except (ValueError, IndexError):
            continue
        if 0 <= pick <= 3:
            out.append(q)
    return out


def spread_subset(items: List[int], n: int) -> List[int]:
    if not n or len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def modified_payload(session: Dict, probe_q: int, donor_q: int) -> Dict:
    """The session as if the probe question never happened: its offered quad,
    r1 winner and pick digit are all overwritten with the donor question's."""
    offered = [list(c) for c in session["offered"]]
    r1 = [list(c) for c in session["r1"]]
    valg = list(session["valg"])

    offered[probe_q * 4:(probe_q + 1) * 4] = [list(c) for c in
                                              session["offered"][donor_q * 4:(donor_q + 1) * 4]]
    r1[probe_q] = list(session["r1"][donor_q])
    valg[probe_q] = session["valg"][donor_q]

    return {
        "offered": offered,
        "r1": r1,
        "r2": [list(c) for c in session["r2"]],
        "final": list(session["final"]),
        "valg": "".join(valg),
        "tider": list(session["tider"]),
    }


def person_vector(payload: Dict, submit_unix: int) -> List[float]:
    """The exact prod vector the result page gets from the worker."""
    f = prod.compute_features(payload, submit_unix)
    return f["gender"] + [f["age"][-1], f["mood"][-1]]


def build_probe_rows(session: Dict, n_probes: int = PROBES_PER_SESSION,
                     with_interactions: bool = True):
    """Yield (row, label, group_id) for one session.

    row = person(479, prod features of the probe-overwritten session)
        + candidate(~35)
        + interactions(17, optional)
    """
    probes = spread_subset(eligible_probes(session), n_probes)
    if not probes:
        return []

    rows = []
    sid = session.get("id", id(session))
    for i, q in enumerate(probes):
        # donor = another loser question (cycle through the eligible list)
        donor = probes[(i + 1) % len(probes)]
        if donor == q:  # only one eligible probe -> no safe donor
            continue
        pay = modified_payload(session, q, donor)
        person = person_vector(pay, session.get("time", 0))

        ctx = None
        if with_interactions:
            ctx = tfeat.session_context(pay["r1"], pay["r2"], pay["final"])

        pick = int(session["valg"][q])
        quad = session["offered"][q * 4:(q + 1) * 4]
        for idx in range(4):
            row = list(person) + candidate_vector(quad[idx])
            if with_interactions:
                row += interaction_vector(quad[idx], ctx)
            rows.append((row, 1 if idx == pick else 0, (sid, q)))
    return rows


def layout(with_interactions: bool = True) -> Dict[str, int]:
    cand = len(candidate_vector([10, 20, 30]))
    inter = 0
    if with_interactions:
        ctx = tfeat.session_context([[i, i, i] for i in range(16)],
                                    [[10, 20, 30]] * 4, [40, 50, 60])
        inter = len(tfeat.interaction_vector([10, 20, 30], ctx))
    return {"person": PERSON_LEN, "candidate": cand, "interaction": inter,
            "total": PERSON_LEN + cand + inter}
