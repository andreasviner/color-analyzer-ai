"""
Diagnose long < short. On the SAME held-out 30% of real long surveys that
train_long.py evaluates on, compare three gender predictors:

  1. long model              (gender_long_trees.json on the long feature vector)
  2. short model, 1 sub-short (short gender_trees.json on sub-short 0)
  3. short model, mean of 4   (decompose the long into its 4 shorts, average)

If (3) beats (1), the long model is underperforming the information it has,
and the fix is to fold the short-model aggregate into the long prediction.
"""
import json, math, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.normpath(os.path.join(HERE, ".."))
CF = os.path.normpath(os.path.join(TRAINING, "..", "cloudflare"))
MODELS = os.path.normpath(os.path.join(TRAINING, "..", "..", "english_html", "color-polygraph", "models-js"))
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, "long-models"))
sys.path.insert(0, CF)

import data_cleaning as dc
import train_long as tl
import features as sf          # short feature extractor (cloudflare)
import features_long as fl     # long feature extractor


def walk(trees, feats):
    total = 0.0
    for t in trees:
        i = 0
        while t[i * 4] != -1:
            i = t[i * 4 + 2] if feats[t[i * 4]] <= t[i * 4 + 1] else t[i * 4 + 3]
        total += t[i * 4 + 1]
    return total


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def load_trees(name):
    with open(os.path.join(MODELS, name), encoding="utf-8") as fh:
        return json.load(fh)["trees"]


def main():
    payloads, labels = tl._load_real_long()
    g = np.array([l["gender"] for l in labels], dtype=np.int8)
    idx = np.arange(len(payloads))
    _, te = train_test_split(idx, test_size=0.30, random_state=42, stratify=g)
    print(f"{len(payloads)} real longs -> {len(te)} in the honest test split")

    g_short = load_trees("gender_trees.json")
    g_long = load_trees("gender_long_trees.json")

    y, p_long, p_short1, p_short4 = [], [], [], []
    for i in te:
        p, lab = payloads[i], labels[i]
        y.append(int(lab["gender"]))
        # long model
        flv = fl.compute_features_long(p, lab["time"])["gender"]
        p_long.append(sig(walk(g_long, flv)))
        # short model on each of the 4 sub-shorts
        shorts = dc.long_payload_to_shorts(p, lab, "d")
        probs = []
        for row in shorts:
            payload = {"offered": row[8][0], "r1": row[8][1], "r2": row[8][2],
                       "final": row[8][3], "valg": row[6], "tider": row[7]}
            fv = sf.compute_features(payload, lab["time"])["gender"]
            probs.append(sig(walk(g_short, fv)))
        p_short1.append(probs[0])
        p_short4.append(float(np.mean(probs)))

    y = np.array(y)
    print(f"  long model            AUC = {roc_auc_score(y, p_long):.4f}")
    print(f"  short model, 1 sub    AUC = {roc_auc_score(y, p_short1):.4f}")
    print(f"  short model, mean(4)  AUC = {roc_auc_score(y, p_short4):.4f}")
    # simple blend
    for w in (0.3, 0.5, 0.7):
        blend = w * np.array(p_short4) + (1 - w) * np.array(p_long)
        print(f"  blend {w:.1f}*short4 + {1-w:.1f}*long  AUC = {roc_auc_score(y, blend):.4f}")


if __name__ == "__main__":
    main()
