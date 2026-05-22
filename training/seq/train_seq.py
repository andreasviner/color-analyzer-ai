"""
Multi-task sequence model on the color-polygraph data.

For each session, build a length-21 sequence (16 r1 + 4 r2 + 1 final).
Each step is a 38-dim vector:
    - the 4 offered colors (RGB + HSL = 24)
    - the chosen color (RGB + HSL = 6)
    - position chosen (one-hot, 4)
    - time delta normalised (1)
    - step type one-hot (r1 / r2 / final, 3)

Architecture:
    GRU(input=38, hidden=96, layers=2, dropout=0.2)
    -> MLP trunk (96 -> 64)
    -> three heads: gender (sigmoid), age (linear), mood (linear)

Trained jointly with a weighted multi-task loss. 5-fold CV.
Runs on CPU in about 10 minutes on a recent laptop.

Run features.py once for the validation rules - this script re-parses
save.ligma itself, so it does not need features.npy.
"""

import colorsys
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "..", "raw", "save.ligma")

N_QUESTIONS = 21
N_R1 = 16
N_R2 = 4
SEED = 42

DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000

EPOCHS =100
BATCH_SIZE = 64
LR = 1e-3
LOSS_W_GENDER = 1.0
LOSS_W_AGE = 5.0
LOSS_W_MOOD = 5.0
HIDDEN = 48
NUM_LAYERS = 1
DROPOUT = 0.4
WEIGHT_DECAY = 1e-4


# ---------- Data loading ----------

def is_valid(row):
    try:
        if row[5] not in ("g", "j"):
            return False
        age = int(row[3])
        if not (6 <= age <= 68):
            return False
        if row[8] == "no data":
            return False
        if len(row[8]) < 4:
            return False
        if len(row[8][0]) < 64 or len(row[8][1]) < 16 or len(row[8][2]) < 4:
            return False
        if len(row[7]) < N_QUESTIONS:
            return False
        total = int(row[7][-1])
        if total < DURATION_MIN_MS or total > DURATION_MAX_MS:
            return False
        if not str(row[4]).lstrip("-").isdigit():
            return False
        return True
    except Exception:
        return False


def rgb_hsl(rgb):
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return r, g, b, h, s, l


def build_step(four_colors, position, time_sec, step_type):
    feats = []
    for c in four_colors:
        feats.extend(rgb_hsl(c))                       # 4 * 6 = 24
    feats.extend(rgb_hsl(four_colors[position]))       # 6
    pos_oh = [0.0] * 4
    pos_oh[position] = 1.0
    feats.extend(pos_oh)                               # 4
    # Clip per-question time to 7s and scale to ~[0, 1]
    feats.append(min(max(time_sec, 0.0), 7.0) / 7.0) # 1
    type_oh = [0.0] * 3
    type_oh[step_type] = 1.0
    feats.extend(type_oh)                              # 3
    return feats  # 38


def session_sequence(row):
    valg = row[6]
    tider = [int(x) for x in row[7]]
    offered = row[8][0]
    r1 = row[8][1]
    r2 = row[8][2]

    deltas = [tider[0]] + [tider[i] - tider[i - 1] for i in range(1, len(tider))]
    deltas = [max(d, 0) / 1000.0 for d in deltas]

    steps = []
    # 16 round-1 steps: each group is offered[q*4..q*4+3]
    for q in range(N_R1):
        group = offered[q * 4:(q + 1) * 4]
        pos = max(0, min(3, int(valg[q])))
        steps.append(build_step(group, pos, deltas[q], 0))
    # 4 round-2 steps: each group is r1[q*4..q*4+3]
    for q in range(N_R2):
        group = r1[q * 4:(q + 1) * 4]
        pos = max(0, min(3, int(valg[N_R1 + q])))
        steps.append(build_step(group, pos, deltas[N_R1 + q], 1))
    # final step: group is r2 (4 colors)
    pos = max(0, min(3, int(valg[N_R1 + N_R2])))
    steps.append(build_step(r2, pos, deltas[N_R1 + N_R2], 2))

    return np.array(steps, dtype=np.float32)  # [21, 38]


def load_sequences():
    with open(SOURCE, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    print(f"loaded {len(rows)} raw rows")

    seqs, genders, ages, moods = [], [], [], []
    for row in rows:
        if not is_valid(row):
            continue
        try:
            seqs.append(session_sequence(row))
        except Exception:
            continue
        genders.append(1 if row[5] == "j" else 0)
        ages.append(int(row[3]))
        moods.append(int(row[4]))

    X = np.stack(seqs, axis=0)  # [N, 21, 38]
    g = np.array(genders, dtype=np.int64)
    a = np.array(ages,    dtype=np.float32)
    m = np.array(moods,   dtype=np.float32)
    print(f"built {X.shape[0]} sequences of shape {X.shape[1:]}")
    return X, g, a, m


# ---------- Model ----------

class SeqModel(nn.Module):
    def __init__(self, in_dim=38, hidden=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        # nn.LSTM(dropout=...) is a no-op when num_layers == 1
        self.gru = nn.LSTM(
            in_dim, hidden, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.in_dropout = nn.Dropout(dropout)
        self.trunk = nn.Sequential(
            nn.Linear(hidden, 48),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_gender = nn.Linear(48, 1)
        self.head_age    = nn.Linear(48, 1)
        self.head_mood   = nn.Linear(48, 1)

    def forward(self, x):
        out, _ = self.gru(self.in_dropout(x))
        h = out[:, -1, :]
        z = self.trunk(h)
        return (
            self.head_gender(z).squeeze(-1),
            self.head_age(z).squeeze(-1),
            self.head_mood(z).squeeze(-1),
        )


# ---------- Training ----------

AGE_MIN = 6.0
AGE_SPAN = 62.0
MOOD_SPAN = 60.0


def train_fold(X_tr, g_tr, a_tr, m_tr, X_te, g_te, a_te, m_te, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SeqModel()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_ep = 0

    # Normalised targets for the regression heads
    a_tr_n = (a_tr - AGE_MIN) / AGE_SPAN
    a_te_n = (a_te - AGE_MIN) / AGE_SPAN
    m_tr_n = m_tr / MOOD_SPAN
    m_te_n = m_te / MOOD_SPAN

    X_tr_t = torch.from_numpy(X_tr)
    g_tr_t = torch.from_numpy(g_tr).float()
    a_tr_t = torch.from_numpy(a_tr_n)
    m_tr_t = torch.from_numpy(m_tr_n)
    X_te_t = torch.from_numpy(X_te)
    g_te_t = torch.from_numpy(g_te).float()
    a_te_t = torch.from_numpy(a_te_n)
    m_te_t = torch.from_numpy(m_te_n)

    N = X_tr_t.size(0)

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(N)
        tot = 0.0
        nb = 0
        for i in range(0, N, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = X_tr_t[idx]
            g_logit, a_pred, m_pred = model(xb)
            l_g = F.binary_cross_entropy_with_logits(g_logit, g_tr_t[idx])
            l_a = F.mse_loss(a_pred, a_tr_t[idx])
            l_m = F.mse_loss(m_pred, m_tr_t[idx])
            loss = LOSS_W_GENDER * l_g + LOSS_W_AGE * l_a + LOSS_W_MOOD * l_m
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()

        # Evaluate val every epoch so we can snapshot the best one
        model.eval()
        with torch.no_grad():
            g_logit, a_pred, m_pred = model(X_te_t)
            l_g = F.binary_cross_entropy_with_logits(g_logit, g_te_t).item()
            l_a = F.mse_loss(a_pred, a_te_t).item()
            l_m = F.mse_loss(m_pred, m_te_t).item()
            val_total = LOSS_W_GENDER * l_g + LOSS_W_AGE * l_a + LOSS_W_MOOD * l_m
            g_prob = torch.sigmoid(g_logit).numpy()
            g_acc = ((g_prob >= 0.5).astype(int) == g_te).mean()

        if val_total < best_val:
            best_val = val_total
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_ep = ep

        if ep % 5 == 0 or ep == EPOCHS - 1:
            marker = "  *" if ep == best_ep else ""
            print(f"    ep {ep:3d}  train_loss {tot / nb:.4f}  "
                  f"val_bce {l_g:.3f}  val_age_mse {l_a:.4f}  "
                  f"val_mood_mse {l_m:.4f}  val_gender_acc {g_acc:.3f}{marker}")

    print(f"    restoring best snapshot from ep {best_ep}  (val_total={best_val:.4f})")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        g_logit, a_pred, m_pred = model(X_te_t)
        g_prob   = torch.sigmoid(g_logit).numpy()
        a_pred_u = a_pred.numpy() * AGE_SPAN + AGE_MIN
        m_pred_u = m_pred.numpy() * MOOD_SPAN
    return g_prob, a_pred_u, m_pred_u


def main():
    X, g, a, m = load_sequences()
    N = X.shape[0]
    print(f"X={X.shape}  boys/girls={int((g == 0).sum())}/{int((g == 1).sum())}  "
          f"age mean={a.mean():.1f}  mood mean={m.mean():.1f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_g = np.zeros(N, dtype=np.float32)
    oof_a = np.zeros(N, dtype=np.float32)
    oof_m = np.zeros(N, dtype=np.float32)

    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, g)):
        print(f"\n--- fold {fold + 1}/5  train={len(tr)}  val={len(va)} ---")
        g_prob, a_pred, m_pred = train_fold(
            X[tr], g[tr], a[tr], m[tr],
            X[va], g[va], a[va], m[va],
            seed=SEED + fold,
        )
        oof_g[va] = g_prob
        oof_a[va] = a_pred
        oof_m[va] = m_pred
        elapsed = time.time() - t0
        print(f"    fold done in {elapsed:.1f}s total")

    print()
    print("=" * 64)
    print("SEQUENCE MODEL (GRU, multi-task, 5-fold CV)")
    print("=" * 64)
    g_pred = (oof_g >= 0.5).astype(int)
    print(f"  gender   acc={accuracy_score(g, g_pred):.3f}  "
          f"AUC={roc_auc_score(g, oof_g):.3f}  F1={f1_score(g, g_pred):.3f}")
    print(f"  age      MAE={mean_absolute_error(a, oof_a):.2f}  "
          f"R2={r2_score(a, oof_a):+.3f}")
    print(f"  mood     MAE={mean_absolute_error(m, oof_m):.2f}  "
          f"R2={r2_score(m, oof_m):+.3f}")

    # Save OOF predictions so they can be stacked with the train.py outputs
    np.savez(
        os.path.join(HERE, "seq_oof.npz"),
        gender=oof_g, age=oof_a, mood=oof_m,
    )
    print("\nwrote seq_oof.npz (gender / age / mood OOF predictions)")


if __name__ == "__main__":
    print("Training color-polygraph sequence model (5-fold CV)...")
    main()
