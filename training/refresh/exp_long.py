"""
Three long-model experiments, all evaluated on the SAME honest held-out 30%
of real long surveys (seed 42 stratified, the split train_long uses). None of
the training sets contain a test long (or its sub-surveys), so the numbers are
leakage-free and directly comparable.

  current : synthetic blended longs (4 different people stacked) + real_tr   [today's deployed approach]
  exp1    : short model run on the long's 4 sub-surveys, aggregated          [no long model at all]
  exp2    : long model trained on REAL long data only (no synthetic)
  exp3    : synthetic longs built by DUPLICATING one short 4x (one person,
            not blended) + real_tr
"""
import json, math, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.normpath(os.path.join(HERE, ".."))
CF = os.path.normpath(os.path.join(TRAINING, "..", "cloudflare"))
MODELS = os.path.normpath(os.path.join(TRAINING, "..", "..", "english_html", "color-polygraph", "models-js"))
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, "long-models"))
sys.path.insert(0, CF)

import data_cleaning as dc
import train_long as tl
import features as sf
import features_long as fl

SEED = 42
CLF = tl.CHAMPION_CLF
REG = tl.CHAMPION_REG


# ---------- long feature helpers (same math as train_long) ----------

def feats(payloads, labels):
    Xs = np.array([fl._extract_static_long(p, labels[i]["time"]) for i, p in enumerate(payloads)], dtype=np.float32)
    n = len(payloads)
    disc = np.zeros((n, fl.N_BUCKETS), dtype=np.float32)
    smooth = np.zeros((n, fl.N_BUCKETS), dtype=np.float32)
    for i, p in enumerate(payloads):
        d, s = fl.compute_bucket_delta_long(p)
        disc[i] = d
        smooth[i] = s
    return Xs, disc, smooth


def grids(disc, g, a, m):
    girly = disc[g == 1].mean(0); masc = disc[g == 0].mean(0)
    age_g = ((a - a.mean())[:, None] * disc).mean(0)
    mood_g = ((m - m.mean())[:, None] * disc).mean(0)
    return girly, masc, age_g, mood_g


def stack(Xs, smooth, gr):
    girly, masc, age_g, mood_g = gr
    gt = smooth @ girly; mt = smooth @ masc; at = smooth @ age_g; mo = smooth @ mood_g
    Xg = np.concatenate([Xs, gt[:, None], mt[:, None], (gt - mt)[:, None]], 1)
    Xa = np.concatenate([Xs, at[:, None]], 1)
    Xm = np.concatenate([Xs, mo[:, None]], 1)
    return Xg, Xa, Xm


def arrs(labels):
    return (np.array([l["gender"] for l in labels], np.int8),
            np.array([l["age"] for l in labels], np.float32),
            np.array([l["mood"] for l in labels], np.float32))


# ---------- duplicate one short 4x into a long (exp3) ----------

def dup_short_to_long(s):
    """s = tl._parse_short output. Build a long that is this one person's short
    replicated 4x (no blending of different people)."""
    v0 = s["valg"][0:16]; v1 = s["valg"][16:20]; vf = s["valg"][20]
    d0 = s["deltas"][0:16]; d1 = s["deltas"][16:20]; df = s["deltas"][20]
    payload = {
        "offered": [list(c) for c in s["offered"]] * 4,   # 256
        "r1": [list(c) for c in s["r1"]] * 4,               # 64
        "r2": [list(c) for c in s["r2"]] * 4,               # 16
        "r3": [list(s["final"]) for _ in range(4)],         # 4
        "final": list(s["final"]),
        "valg": v0 * 4 + v1 * 4 + vf * 4 + "0",             # 64+16+4+1
    }
    deltas = d0 * 4 + d1 * 4 + [df] * 4 + [float(df)]
    tider, run = [], 0.0
    for d in deltas:
        run += d; tider.append(int(run))
    payload["tider"] = tider
    label = {"gender": s["gender"], "age": s["age"], "mood": s["mood"], "time": s["time"]}
    return payload, label


# ---------- short-model tree walk (exp1) ----------

def _walk(trees, f):
    t = 0.0
    for tr in trees:
        i = 0
        while tr[i * 4] != -1:
            i = tr[i * 4 + 2] if f[tr[i * 4]] <= tr[i * 4 + 1] else tr[i * 4 + 3]
        t += tr[i * 4 + 1]
    return t


def _trees(name):
    return json.load(open(os.path.join(MODELS, name), encoding="utf-8"))["trees"]


def main():
    rng = np.random.RandomState(SEED)

    print("Loading real longs + building test split...")
    real_p, real_l = tl._load_real_long()
    gr = np.array([l["gender"] for l in real_l], np.int8)
    idx = np.arange(len(real_p))
    tr_i, te_i = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=gr)
    te_p = [real_p[i] for i in te_i]; te_l = [real_l[i] for i in te_i]
    rtr_p = [real_p[i] for i in tr_i]; rtr_l = [real_l[i] for i in tr_i]
    print(f"  {len(real_p)} real longs -> train {len(tr_i)}, test {len(te_i)}")

    yte_g, yte_a, yte_m = arrs(te_l)
    TXs, Tdisc, Tsm = feats(te_p, te_l)

    def evaluate(train_p, train_l, weight, name):
        Xs, disc, sm = feats(train_p, train_l)
        g, a, m = arrs(train_l)
        gr_ = grids(disc, g, a, m)
        Xg, Xa, Xm = stack(Xs, sm, gr_)
        TXg, TXa, TXm = stack(TXs, Tsm, gr_)
        clf = lgb.LGBMClassifier(**CLF).fit(Xg, g, sample_weight=weight)
        p = clf.predict_proba(TXg)[:, 1]
        auc = roc_auc_score(yte_g, p); acc = accuracy_score(yte_g, (p >= 0.5).astype(int))
        rega = lgb.LGBMRegressor(**REG).fit(Xa, a, sample_weight=weight)
        regm = lgb.LGBMRegressor(**REG).fit(Xm, m, sample_weight=weight)
        mae_a = mean_absolute_error(yte_a, rega.predict(TXa))
        mae_m = mean_absolute_error(yte_m, regm.predict(TXm))
        print(f"  {name:34s} gender AUC={auc:.4f} acc={acc:.4f}  age MAE={mae_a:.2f}  mood MAE={mae_m:.2f}  (train n={len(train_l)})")

    # current: synthetic blended + real_tr (weight 3 on real)
    print("Building current (synthetic blended) longs...")
    shorts = [tl._parse_short(r) for r in dc.load_short_rows() if tl._is_valid(r)]
    syn_p, syn_l, _ = tl._build_long_sessions(shorts, rng)
    cur_p = syn_p + rtr_p; cur_l = syn_l + rtr_l
    w = np.ones(len(cur_l), np.float32); w[len(syn_l):] = tl.REAL_LONG_WEIGHT
    evaluate(cur_p, cur_l, w, "current: blended synth + real")

    # exp2: real-only
    evaluate(rtr_p, rtr_l, None, "exp2: real long data only")

    # exp3: duplicate-short longs + real_tr
    print("Building exp3 (duplicate-short) longs...")
    dup_p, dup_l = [], []
    for s in shorts:
        p, l = dup_short_to_long(s)
        dup_p.append(p); dup_l.append(l)
    e3_p = dup_p + rtr_p; e3_l = dup_l + rtr_l
    w3 = np.ones(len(e3_l), np.float32); w3[len(dup_l):] = tl.REAL_LONG_WEIGHT
    evaluate(e3_p, e3_l, w3, "exp3: duplicate-short + real")

    # exp1: short model on the 4 sub-surveys, aggregated
    gtr = _trees("gender_trees.json")
    p1 = []
    for p, l in zip(te_p, te_l):
        probs = []
        for row in dc.long_payload_to_shorts(p, l, "e"):
            pay = {"offered": row[8][0], "r1": row[8][1], "r2": row[8][2],
                   "final": row[8][3], "valg": row[6], "tider": row[7]}
            fv = sf.compute_features(pay, l["time"])["gender"]
            probs.append(1.0 / (1.0 + math.exp(-_walk(gtr, fv))))
        p1.append(float(np.mean(probs)))
    auc1 = roc_auc_score(yte_g, p1); acc1 = accuracy_score(yte_g, (np.array(p1) >= 0.5).astype(int))
    print(f"  {'exp1: short model, mean of 4 subs':34s} gender AUC={auc1:.4f} acc={acc1:.4f}  (no long model)")


if __name__ == "__main__":
    main()
