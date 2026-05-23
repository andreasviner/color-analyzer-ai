"""
Single inference entrypoint for the Cloudflare endpoint to wrap.

Usage:
    from predict import predict
    result = predict(payload_dict)
    # result = {
    #   "gender": {"label": "girl", "girl_probability": 0.83},
    #   "age":    {"years": 14.2},
    #   "mood":   {"value": 38.7, "label": "happy"},
    #   "model":  "lightgbm-bucket-scores",
    # }

The three LightGBM boosters and the bucket-score grids are loaded once at
import time and cached at module scope so each request after the first only
pays the feature-extraction + tree-eval cost.
"""

import json
import os
from typing import Dict

import numpy as np
import lightgbm as lgb

from features_inference import (
    BUCKET_SCORES_PATH,
    compute_features,
)

HERE = os.path.dirname(os.path.abspath(__file__))

_BUCKET_SCORES = dict(np.load(BUCKET_SCORES_PATH))
_GENDER_MODEL = lgb.Booster(model_file=os.path.join(HERE, "gender_model.txt"))
_AGE_MODEL    = lgb.Booster(model_file=os.path.join(HERE, "age_model.txt"))
_MOOD_MODEL   = lgb.Booster(model_file=os.path.join(HERE, "mood_model.txt"))

with open(os.path.join(HERE, "metadata.json"), encoding="utf-8") as _fh:
    _METADATA = json.load(_fh)


def _validate_payload(payload: Dict) -> None:
    required = ("offered", "r1", "r2", "final", "valg", "tider")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"payload missing keys: {missing}")
    if len(payload["offered"]) != 64:
        raise ValueError(f"offered must have 64 entries, got {len(payload['offered'])}")
    if len(payload["r1"]) != 16:
        raise ValueError(f"r1 must have 16 entries, got {len(payload['r1'])}")
    if len(payload["r2"]) != 4:
        raise ValueError(f"r2 must have 4 entries, got {len(payload['r2'])}")
    if len(payload["final"]) != 3:
        raise ValueError(f"final must be an [r,g,b] triple, got {payload['final']}")
    if len(payload["valg"]) < 21:
        raise ValueError(f"valg must be 21 digits, got {len(payload['valg'])}")
    if len(payload["tider"]) < 21:
        raise ValueError(f"tider must have 21 entries, got {len(payload['tider'])}")


def _mood_label(value: float) -> str:
    # mood is 0 (sad) -> 60 (happy); split into rough quartiles for a friendly label
    if value >= 45: return "happy"
    if value >= 30: return "okay"
    if value >= 15: return "down"
    return "glum"


def predict(payload: Dict) -> Dict:
    _validate_payload(payload)

    feats = compute_features(payload, bucket_scores=_BUCKET_SCORES)

    girl_prob = float(_GENDER_MODEL.predict(feats["gender"])[0])
    age_pred  = float(_AGE_MODEL.predict(feats["age"])[0])
    mood_pred = float(_MOOD_MODEL.predict(feats["mood"])[0])

    age_min = _METADATA["target_stats"]["age_min"]
    age_max = _METADATA["target_stats"]["age_max"]
    age_pred = max(age_min, min(age_max, age_pred))
    mood_pred = max(0.0, min(60.0, mood_pred))

    return {
        "gender": {
            "label": "girl" if girl_prob >= 0.5 else "boy",
            "girl_probability": round(girl_prob, 4),
        },
        "age":  {"years": round(age_pred, 1)},
        "mood": {"value": round(mood_pred, 1), "label": _mood_label(mood_pred)},
        "model": "lightgbm-bucket-scores",
    }


if __name__ == "__main__":
    # Quick smoke-test from the command line: feed the first valid raw row
    # back through the inference function and print the prediction next to the
    # ground truth.
    import sys
    raw_path = os.path.normpath(os.path.join(HERE, "..", "raw", "save.ligma"))
    with open(raw_path, encoding="utf-8") as fh:
        rows = json.load(fh)

    sample_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    chosen = None
    seen = 0
    for row in rows:
        if row[5] not in ("g", "j"):
            continue
        if seen == sample_idx:
            chosen = row
            break
        seen += 1
    if chosen is None:
        raise SystemExit(f"no valid row at index {sample_idx}")

    payload = {
        "offered": chosen[8][0],
        "r1": chosen[8][1],
        "r2": chosen[8][2],
        "final": chosen[8][3],
        "valg": chosen[6],
        "tider": [int(x) for x in chosen[7]],
        "submit_unix": int(chosen[1]) if str(chosen[1]).isdigit() else None,
    }
    result = predict(payload)
    truth = {
        "gender": "girl" if chosen[5] == "j" else "boy",
        "age": int(chosen[3]),
        "mood": int(chosen[4]),
    }
    print("prediction:", json.dumps(result, indent=2))
    print("truth     :", json.dumps(truth, indent=2))
