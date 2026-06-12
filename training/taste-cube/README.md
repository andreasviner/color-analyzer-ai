# Colour-pick model + personal colour cube

> **Current production model: `pick_*`** (prod person features + client-side
> candidate descriptors), trained by `train_pick.py`, served as
> `models-js/pick_trees.json` and rendered live by `models-js/pick_cube.js`.
> The `taste_*` files below it are the earlier fingerprint-based experiment,
> kept for reference (its `taste_features.js` colour math and interaction
> block are still used by the pick pipeline).

## Production: the colour-pick model

Goal: a person took the survey; show them 4 NEW colours; predict which one
they pick. Person = the 479-float prod feature vector the worker already
returns for the gender/age/mood reveal (`features.gender` + age/mood bucket
totals). Probe construction: a non-advancing round-0 question is overwritten
with a duplicate of another loser question (quad + winner + pick digit), so
the prod features computed from the modified session cannot contain the
answer. Candidates get only client-computable descriptors: RGB, HSL, hue
sin/cos, CMYK, LAB, YUV, warmth, chroma, 12 reference distances (+17
interaction features vs the person's winning colours).

Metrics (session-level 90/10 split, seed 42, chance = 0.25):

| Variant | pick-accuracy | leak gate |
|---|---|---|
| A: person + candidate | 0.508 | 0.264 |
| **B: + interactions (deployed)** | **0.513** | 0.264 |

Retrain: `python train_pick.py` (~8 min) — emits `models-js/pick_trees.json`
(bit-exact verified), `pick_parity.json` (JS mirror check, must stay < 1e-5),
`pick_summary.json`.

Serve side:
- Worker `GET /survey/:id` regenerates the feature vectors from the stored
  payload, so shared links can run the cube too.
- `models-js/pick_features.js` mirrors the candidate + interaction blocks.
- `models-js/pick_cube.js` lets the model retake the survey live in the
  browser: survey-identical palette generation (Oklab farthest-point
  sampling), full 64-16-4-1 brackets, 5,000 answered questions (~4 s,
  yielding between brackets), tallied into the same voxel arrays as the
  population cube and rendered as it fills, with a progress bar.

---

# Earlier experiment: taste cube (fingerprint model)

Trains the model behind the **personal colour cube** on the survey result page
(`ai/english_html/color-polygraph/survey-result.html`). The project page's cube
shows what a *group* likes/dislikes, tallied from the whole dataset. This makes
the same cube for a *single person*, even though one survey is far too short to
fill 512 voxels directly: we learn a model of the person's taste and let it
choose between thousands of synthetic colour sets.

**Scope: short survey only.** The long (256-colour) variant will mirror this later.

## How it works

1. **Model** — a binary `LGBMClassifier`. One row = a person's *taste
   fingerprint* + one *candidate colour*; the label is 1 if the person actually
   picked that colour. The raw logit is a globally-comparable per-colour
   desirability, so at serve time we score the 512 voxel centres once and
   Monte-Carlo synthetic 4-colour questions into the same
   `[offered, r1, r2, final]` voxel arrays the population cube renders.

2. **Leakage-safe training rows** — for each person we take a few "loser"
   round-0 questions (winner did *not* advance past round 0, so `r2`/`final`
   stay safe to use). For each probe we overwrite its winner slot with a
   duplicate of another question (keeps the fingerprint 16-winners-wide but
   drops the probe's own pick) and emit 4 rows, one per candidate colour.
   `PROBES_PER_SESSION` in `taste_features.py` controls how many probes per
   person (default 5; easy to edit).

3. **Winners-only fingerprint** — uses only the colours the person *won* (r1 /
   r2 / final), never the offered options, so it also works on the shared `?id=`
   result link (which has no offered options stored).

## Files

| File | Role |
|------|------|
| `taste_features.py` | Canonical feature extractor (train side). HSL/LAB written explicitly so the JS mirror can match exactly. 74 features: 34 fingerprint + 23 candidate + 17 interaction. |
| `train_taste.py` | Build rows, evaluate (session-level split), refit, emit artifacts. |
| `tune_taste.py` | Offline tuning harness (probes sweep, feature-block ablation, hyperparameter grid) on a fixed session-level val set. Not part of the build. |
| `summary.json` | Emitted metrics. |
| `taste_parity.json` | Emitted fixture; the JS mirror is checked against it. |

Emitted outside this folder:
- `ai/english_html/color-polygraph/models-js/taste_trees.json` — flat-tree model the browser walks.
- JS serve side: `models-js/taste_features.js` (mirror of `taste_features.py`) + `models-js/taste_cube.js` (512-grid scoring, Monte Carlo, 3D render).

## Retrain

```
python train_taste.py
```

(~14 min: ~268k rows over 1500 trees, two passes.) Then re-check the JS mirror:

```
node -e "require('./ai/english_html/color-polygraph/models-js/taste_features.js'); \
  const fx=require('./ai/color-polygraph/training/taste-cube/taste_parity.json'); \
  console.log('parity delta', TasteFeatures.checkParity(fx));"
```

Parity delta must be < 1e-5 (currently ~1e-14).

## Current metrics (session-level 90/10 split, seed 42)

- **Holdout pick-accuracy: 0.50** — given a real held-out question's 4 colours,
  the model's top pick matches the person's actual pick 50% of the time (chance
  = 0.25).
- **Fingerprint-only baseline: 0.25** ≈ chance — the leakage gate: a model that
  sees only the fingerprint (identical across a question's 4 candidates) cannot
  do better than chance, confirming no answer leaks through the fingerprint.
- Row AUC 0.72. Tree JSON (1500 trees, ~5.1 MB, lazy-loaded only when the cube
  opens) ↔ LightGBM emit delta 0.0.

## Tuning notes (from `tune_taste.py`)

- **Probes per person** lift accuracy then plateau: 2→0.469, 3→0.484, 5→0.491,
  8→0.492, 12→0.494. `PROBES_PER_SESSION = 10` sits on the plateau.
- **Interactions carry the model.** Block ablation (pick-accuracy): fingerprint
  only 0.26 (≈chance), candidate only 0.31, fingerprint+candidate 0.33,
  candidate+interactions 0.45, all 0.49. The trees can't synthesise
  fingerprint×candidate crosses on their own, so 6 extra interaction features
  (signed hue offset, hue-vs-final, min RGB/LAB to a winner, LAB-to-r2-mean,
  warmth delta, closeness normalised by taste spread) were added and lifted
  accuracy ~1 point.
- **Hyperparameters:** deeper + slower (1500 trees, lr 0.015, num_leaves 63,
  min_child_samples 100) beat the shallow default by ~0.5 point.
