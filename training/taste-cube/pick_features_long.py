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

N_R0_LONG = 64

# Probe questions per (synthetic) long survey. There are only ~1.7k long rows
# (vs 6.7k shorts), but each has ~48 eligible loser questions, so we take more
# probes per row to keep the training set a healthy size. Easy to edit.
PROBES_PER_SESSION_LONG = 16

PERSON_LEN_LONG = 239  # 234 static + girly/masc/signed + age_total + mood_total


def eligible_probes(session: Dict) -> List[int]:
    """Round-0 questions whose winner did not advance (absent from r2)."""
    r1 = session["r1"]
    r2 = [list(c) for c in session["r2"]]
    out = []
    for q in range(N_R0_LONG):
        if tfeat._color_in(r1[q], r2):
            continue
        try:
            pick = int(session["valg"][q])
        except (ValueError, IndexError):
            continue
        if 0 <= pick <= 3:
            out.append(q)
    return out


def modified_payload(session: Dict, probe_q: int, donor_q: int) -> Dict:
    """The long session as if the probe question never happened."""
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
        "r3": [list(c) for c in session["r3"]],
        "final": list(session["final"]),
        "valg": "".join(valg),
        "tider": list(session["tider"]),
    }


def person_vector(payload: Dict, submit_unix: int) -> List[float]:
    """The exact long prod vector the result page gets from the worker."""
    f = fl.compute_features_long(payload, submit_unix)
    return f["gender"] + [f["age"][-1], f["mood"][-1]]


def build_probe_rows(session: Dict, n_probes: int = PROBES_PER_SESSION_LONG,
                     with_interactions: bool = True):
    """Yield (row, label, group_id) for one long session. Same construction as
    the short model's pick_features.build_probe_rows."""
    probes = pf.spread_subset(eligible_probes(session), n_probes)
    if not probes:
        return []

    rows = []
    sid = session.get("id", id(session))
    for i, q in enumerate(probes):
        donor = probes[(i + 1) % len(probes)]
        if donor == q:
            continue
        pay = modified_payload(session, q, donor)
        person = person_vector(pay, session.get("time", 0))

        ctx = None
        if with_interactions:
            ctx = tfeat.session_context(pay["r1"], pay["r2"], pay["final"])

        pick = int(session["valg"][q])
        quad = session["offered"][q * 4:(q + 1) * 4]
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
