"""
Row construction for the LONG-survey colour-pick model.

Long bracket: offered(256) -> 64 round-0 questions -> r1(64) -> r2(16) ->
r3(4) -> final. Mirrors pick_features.py exactly, with the long prod features
as the person vector:

    person = features_long["gender"] (237) + [age[-1], mood[-1]]  = 239 floats

Probe construction is the same overwrite scheme on the 64 round-0 questions:
a probe is eligible when its winner did NOT advance past round 0 (absent from
r2 -- removing it cannot affect r2/r3/final), and its offered quad, r1 winner
and valg digit are all overwritten with a donor loser question before the prod
features are computed.

Candidate descriptors and interaction features are SHARED with the short model
(same colour math, same JS mirror); the interaction context is built from the
long winners (r1 64-wide, r2 16-wide, final), which the browser reproduces
from the history rounds verbatim.
"""

import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
CF_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "cloudflare"))
sys.path.insert(0, CF_DIR)
sys.path.insert(0, HERE)

import features_long as fl  # noqa: E402  (the real prod long extractor)
import pick_features as pf  # noqa: E402  (shared candidate descriptors)
import taste_features as tfeat  # noqa: E402  (shared interactions + colour math)

sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "long-models")))
import train_long as tl  # noqa: E402  (duplicate-short long assembly)

N_R0_LONG = 64

# Probe questions per (synthetic) long survey. There are only ~1.7k long rows
# (vs 6.7k shorts), but each has ~48 eligible loser questions, so we take more
# probes per row to keep the training set a healthy size. Easy to edit.
PROBES_PER_SESSION_LONG = 16

PERSON_LEN_LONG = 239  # 234 static + girly/masc/signed + age_total + mood_total


def person_vector(payload: Dict, submit_unix: int) -> List[float]:
    """The exact long prod vector the result page gets from the worker."""
    f = fl.compute_features_long(payload, submit_unix)
    return f["gender"] + [f["age"][-1], f["mood"][-1]]


def _modified_short(short: Dict, probe_q: int, donor_q: int) -> Dict:
    """The SHORT as if the probe question never happened (its offered quad, r1
    winner and valg digit overwritten by a donor loser). Returned in the shape
    tl._dup_short_to_long expects."""
    offered = [list(c) for c in short["offered"]]
    r1 = [list(c) for c in short["r1"]]
    valg = list(short["valg"])
    offered[probe_q * 4:(probe_q + 1) * 4] = [list(c) for c in
                                              short["offered"][donor_q * 4:(donor_q + 1) * 4]]
    r1[probe_q] = list(short["r1"][donor_q])
    valg[probe_q] = short["valg"][donor_q]
    return {
        "offered": offered, "r1": r1,
        "r2": [list(c) for c in short["r2"]],
        "final": list(short["final"]),
        "valg": "".join(valg), "deltas": short["deltas"],
        "gender": short["gender"], "age": short["age"],
        "mood": short["mood"], "time": short.get("time", 0),
    }


def build_probe_rows(short: Dict, n_probes: int = PROBES_PER_SESSION_LONG,
                     with_interactions: bool = True):
    """Yield (row, label, group_id) for one short session, served as a
    duplicate-short long. CRITICAL: the probe question is removed from the SHORT
    *before* it is duplicated 4x into the long, so the probed colour is absent
    from all four blocks. Removing it only once (after duplication) would leave
    three copies in the person features and leak the answer.

    `short` = tl._parse_short output (+ "id"): offered 64, r1 16, r2 4, final,
    valg 21, deltas 21, gender/age/mood/time.
    """
    probes = pf.spread_subset(pf.eligible_probes(short), n_probes)
    if not probes:
        return []

    rows = []
    sid = short.get("id", id(short))
    for i, q in enumerate(probes):
        donor = probes[(i + 1) % len(probes)]
        if donor == q:
            continue
        # remove-then-duplicate: erase q in the short, THEN make the long.
        mod_short = _modified_short(short, q, donor)
        long_pay, _ = tl._dup_short_to_long(mod_short)
        person = person_vector(long_pay, short.get("time", 0))

        ctx = None
        if with_interactions:
            ctx = tfeat.session_context(long_pay["r1"], long_pay["r2"], long_pay["final"])

        pick = int(short["valg"][q])
        quad = short["offered"][q * 4:(q + 1) * 4]   # the ORIGINAL (unremoved) quad
        for idx in range(4):
            row = list(person) + pf.candidate_vector(quad[idx])
            if with_interactions:
                row += tfeat.interaction_vector(quad[idx], ctx)
            rows.append((row, 1 if idx == pick else 0, (sid, q)))
    return rows


def layout(with_interactions: bool = True) -> Dict[str, int]:
    base = pf.layout(with_interactions)
    return {"person": PERSON_LEN_LONG, "candidate": base["candidate"],
            "interaction": base["interaction"],
            "total": PERSON_LEN_LONG + base["candidate"] + base["interaction"]}
