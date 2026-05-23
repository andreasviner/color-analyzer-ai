# Color Polygraph · AI architectures

A six-year follow-up to the 2020 *color polygraph*: re-implementing the same
prediction task under different ML architectures and tracking the numbers on
a public leaderboard at
[ai.andreaslindeman.com/color-polygraph/](https://ai.andreaslindeman.com/color-polygraph/).

## The task

A user is shown 64 colours in groups of 4, picks one per group (16 round-1
picks), then picks one of 4 from the survivors (4 round-2 picks), then one
final color. From this 21-step session and a few timing signals, predict:

- **gender** - binary (boy / girl)
- **age**    - regression, range 6 to 68
- **mood**   - regression, range 0 (happy) to 60 (glum)

The dataset is 6,710 cleaned sessions, the same export the 2020 deep-dive page
uses.

## Folder layout

```
color-polygraph/
├── README.md                this file
├── index.html               the public leaderboard page
├── models/                  one HTML page per architecture (linked from the leaderboard)
│   ├── stacked-gbm.html
│   ├── hybrid-blend.html
│   ├── lgb-search.html
│   ├── lgb-perceptual.html
│   ├── lgb-buckets.html
│   ├── transformer.html
│   ├── bigru.html
│   ├── lstm.html
│   └── baselines.html
├── info/                    supporting documentation pages
│   └── features.html        every one of the 441 engineered features, grouped and explained
└── training/                code, data, and outputs
    ├── features.py          turns raw sessions into a 441-feature engineered vector
    ├── features.npy         the precomputed feature matrix (shape 6710 × 441)
    ├── feature_names.json   ordered names for the feature columns
    ├── targets.npz          gender / age / mood arrays
    ├── raw/save.ligma       raw exported sessions (JSON)
    ├── overnight_out/       shared outputs from the GBM stack (OOF preds, summaries, trial CSV)
    ├── gbm-stack/           LGB+XGB+CAT+HGB+ET with logistic / L1 stacking (best on age)
    ├── extra-features/      single LGB with +33 LAB / prototype / difficulty features
    ├── color-buckets/       single LGB with +5 target-encoded 8x8x8 bucket scores (best on gender)
    ├── lgb-search/          single-family LightGBM random search
    ├── baselines/           five hand-tuned single-model baselines (best on mood)
    ├── transformer/         transformer encoder on the 21-step sequence
    ├── gru/                 bidirectional GRU + attention pooling + side fusion
    ├── seq/                 the original LSTM, smallest sequence model on the board
    └── hybrid/              meta-blender over every model's OOF predictions
```

## Leaderboard

Latest numbers (5-fold CV, seed 42):

| Architecture                  | Gender AUC | Age MAE  | Mood MAE |
|-------------------------------|------------|----------|----------|
| **LightGBM + bucket scores**  | **0.881**  | 6.89     | 8.91     |
| LightGBM + perceptual         | 0.878      | 6.92     | 8.87     |
| Stacked GBM                   | 0.877      | **6.61** | 8.96     |
| Single LightGBM               | 0.876      | 6.95     | **8.85** |
| LightGBM random search        | 0.875      | 6.63     | 9.17     |
| Single XGBoost                | 0.875      | 7.14     | 9.08     |
| Hybrid blend                  | 0.875      | 6.61     | 8.96     |
| HistGradientBoosting          | 0.867      | 7.16     | 9.27     |
| MLP (256, 128, 64)            | 0.823      | 8.41     | 10.84    |
| BiGRU                         | 0.812      | 8.59     | 11.18    |
| Linear                        | 0.810      | 8.77     | 11.35    |
| Transformer                   | 0.797      | 8.88     | 11.29    |
| LSTM                          | 0.781      | 8.64     | 11.18    |

Gradient boosting on the engineered features wins every column, but the column
winners are now three different architectures. The full stack still owns the
hardest target (age regression), the cheaply-tuned default Single LGB stays on
top of mood, and a single LGB with 33 perceptual extras plus 5 target-encoded
8x8x8 colour-bucket scores edges out everything else on gender. The signed
gender-bucket total is the single highest-gain feature in the entire 477-vector
by a factor of 2.5.

The hybrid blend's job is to test whether the sequence models contribute any
signal on top of the trees - the L1 meta-blender's answer is "no, here": it
lands on the GBM stack with weight ~1.0 and assigns the sequence models
essentially zero.

## How to run the training

All scripts assume the working directory is the script's own folder. The feature
matrix needs to be built once before any tree-based architecture is trained.

```bash
# 1. Build the feature matrix (writes features.npy, targets.npz, feature_names.json)
cd training
python features.py

# 2. Pick an architecture and train. Examples:
cd gbm-stack
python overnight_search.py --targets gender,age,mood --skip-existing  # 8-12h
python rescue_target.py --target age --families lgb,xgb,hgb           # rebuild a stack from existing trial OOFs

cd ../baselines
python train.py                                                       # ~25 min, all five baselines, all three heads

cd ../lgb-search
python train_lgb_search.py                                            # 30-trial single-LGB search per target

cd ../transformer
python train_transformer.py                                           # PyTorch, runs on CPU in ~1h

cd ../gru
python train_gru.py                                                   # ~1h CPU

cd ../seq
python train_seq.py                                                   # ~30m CPU

cd ../hybrid
python blend.py                                                       # seconds, just rereads everyone's OOFs

cd ../extra-features
python train.py                                                       # ~3 min, single LGB on 441+33 perceptual features

cd ../color-buckets
python train.py                                                       # ~3 min, +5 target-encoded bucket scores; writes bucket_scores.npz
```

Each training script writes a `*_oof.npz` file with its out-of-fold predictions,
which is what the hybrid blender consumes. The GBM stack additionally writes
`overnight_out/summary_<target>.json` with the headline numbers.

## Validation protocol

- **Folds**: 5-fold CV, seed 42. Tree models use StratifiedKFold for gender and
  plain KFold for the regression heads. Sequence models stratify on gender for
  the multi-task setup (so the same fold definition applies to all three heads).
- **Metric**: ROC-AUC for gender, mean absolute error for age and mood.
- **Meta-blend on the GBM stack**: logistic regression for gender, non-negative
  L1-loss minimisation for age and mood. The L1 detail is important - the base
  LightGBM and XGBoost models are MAE-trained, so their OOF predictions are
  approximately conditional medians, and a ridge / L2 meta-learner pulls the
  blend toward the conditional mean and inflates MAE by roughly 10%.

## When extending the leaderboard

1. Add a new folder under `training/<architecture>/` with a training script
   that writes a `<name>_oof.npz` file in the same fold layout (5-fold CV,
   StratifiedKFold or KFold matching the head, seed 42).
2. Add a row to the leaderboard table in `index.html`.
3. Add a per-architecture detail page in `models/<name>.html` (copy the closest
   existing page and adjust scores, configuration list, and links).
4. Add the new page to `../sitemap.xml`.
5. Optionally, teach `hybrid/blend.py` to load the new OOF file and re-run it
   to see whether the new architecture adds anything to the meta-blend.
