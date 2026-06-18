"""
Sweep the real-long sample weight for the duplicate-short long approach (exp3),
all on the same leakage-free 326 real-long test set. Features + bucket grids are
built once (grids are unweighted means, independent of the weight), so only the
LightGBM refit changes per weight. Run with CP_INCLUDE_DECOMPOSED=0.
"""
import os, sys
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb

os.environ.setdefault("CP_INCLUDE_DECOMPOSED", "0")  # genuine shorts only

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_long as E   # reuse feats/grids/stack/arrs/dup_short_to_long
import train_long as tl
import data_cleaning as dc

SEED = 42
WEIGHTS = [3, 4]


def main():
    rng = np.random.RandomState(SEED)
    real_p, real_l = tl._load_real_long()
    gr = np.array([l["gender"] for l in real_l], np.int8)
    idx = np.arange(len(real_p))
    tr_i, te_i = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=gr)
    te_p = [real_p[i] for i in te_i]; te_l = [real_l[i] for i in te_i]
    rtr_p = [real_p[i] for i in tr_i]; rtr_l = [real_l[i] for i in tr_i]
    print(f"real longs {len(real_p)} -> train {len(tr_i)} test {len(te_i)}")

    shorts = [tl._parse_short(r) for r in dc.load_short_rows() if tl._is_valid(r)]
    dup_p, dup_l = [], []
    for s in shorts:
        p, l = E.dup_short_to_long(s)
        dup_p.append(p); dup_l.append(l)
    train_p = dup_p + rtr_p
    train_l = dup_l + rtr_l
    n_dup = len(dup_l)
    print(f"duplicate-short longs {n_dup} + real_tr {len(rtr_l)} = {len(train_l)}")

    print("Extracting features (once)...")
    Xs, disc, sm = E.feats(train_p, train_l)
    g, a, m = E.arrs(train_l)
    gridz = E.grids(disc, g, a, m)
    Xg, Xa, Xm = E.stack(Xs, sm, gridz)
    TXs, _, Tsm = E.feats(te_p, te_l)
    TXg, TXa, TXm = E.stack(TXs, Tsm, gridz)
    yte_g, yte_a, yte_m = E.arrs(te_l)

    print(f"\n{'weight':>7}  {'gender AUC':>10}  {'gender acc':>10}  {'age MAE':>8}  {'mood MAE':>9}")
    for w in WEIGHTS:
        weight = np.ones(len(train_l), np.float32)
        weight[n_dup:] = w
        clf = lgb.LGBMClassifier(**tl.CHAMPION_CLF).fit(Xg, g, sample_weight=weight)
        p = clf.predict_proba(TXg)[:, 1]
        auc = roc_auc_score(yte_g, p); acc = accuracy_score(yte_g, (p >= 0.5).astype(int))
        rega = lgb.LGBMRegressor(**tl.CHAMPION_REG).fit(Xa, a, sample_weight=weight)
        regm = lgb.LGBMRegressor(**tl.CHAMPION_REG).fit(Xm, m, sample_weight=weight)
        mae_a = mean_absolute_error(yte_a, rega.predict(TXa))
        mae_m = mean_absolute_error(yte_m, regm.predict(TXm))
        print(f"{w:>7}  {auc:>10.4f}  {acc:>10.4f}  {mae_a:>8.2f}  {mae_m:>9.2f}")


if __name__ == "__main__":
    main()
