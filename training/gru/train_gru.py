"""
Multi-task GRU on the color-polygraph sequences, with side info.

Architecture:
    input_proj : Linear(40 -> d_model)
    BiGRU      : 2 layers, hidden=64 each direction (so 128 total per step)
    attn_pool  : learned-query additive attention over the 21 time steps
    side_mlp   : Linear(14 -> 128) -> GELU -> Dropout
    fuse       : Linear(256 -> 128) -> GELU -> Dropout
    heads      : gender (sigmoid), age (linear), mood (linear)

Snapshots on val_bce (gender) so we report the model that's best at the
metric we care about, not the combined multi-task loss.

5-fold CV. AdamW + warmup + cosine decay.
"""

import colorsys
import datetime
import json
import math
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    from zoneinfo import ZoneInfo
    OSLO = ZoneInfo("Europe/Oslo")
except ImportError:
    OSLO = None

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "..", "raw", "save.ligma")

N_QUESTIONS = 21
N_R1 = 16
N_R2 = 4
SEED = 42

DURATION_MIN_MS = 15_000
DURATION_MAX_MS = 600_000

EPOCHS = 50
BATCH_SIZE = 64
LR = 6e-4
WARMUP_EPOCHS = 3
WEIGHT_DECAY = 3e-4

D_MODEL = 64
HIDDEN = 64
N_LAYERS = 2
DROPOUT = 0.3

LOSS_W_GENDER = 1.0
LOSS_W_AGE = 3.0
LOSS_W_MOOD = 3.0

TIME_CLIP_SEC = 8.0
RELATIVE_TIME_CAP = 4.0
TOTAL_TIME_CAP = 300.0
MEAN_Q_CAP = 12.0
SLOPE_CAP_MS = 1500.0

STEP_DIM = 40
SIDE_DIM = 14


# ---------- Data loading (identical to train_transformer.py) ----------

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


def hour_minute(timestamp):
    try:
        t = int(timestamp)
    except (TypeError, ValueError):
        return None, None
    if t <= 0:
        return None, None
    try:
        dt = (datetime.datetime.fromtimestamp(t, tz=OSLO)
              if OSLO is not None
              else datetime.datetime.fromtimestamp(t))
        return dt.hour, dt.minute
    except (OSError, OverflowError, ValueError):
        return None, None


def build_step(four_colors, position, time_sec, session_mean_sec, step_type):
    feats = []
    for c in four_colors:
        feats.extend(rgb_hsl(c))
    feats.extend(rgb_hsl(four_colors[position]))
    pos_oh = [0.0] * 4
    pos_oh[position] = 1.0
    feats.extend(pos_oh)
    feats.append(min(max(time_sec, 0.0), TIME_CLIP_SEC) / TIME_CLIP_SEC)
    rel = (time_sec / session_mean_sec) if session_mean_sec > 0.01 else 1.0
    feats.append(min(rel, RELATIVE_TIME_CAP) / RELATIVE_TIME_CAP)
    feats.append(1.0 if time_sec > TIME_CLIP_SEC else 0.0)
    type_oh = [0.0] * 3
    type_oh[step_type] = 1.0
    feats.extend(type_oh)
    return feats


def session_side_features(row, deltas_sec):
    total = sum(deltas_sec)
    mean_q = float(np.mean(deltas_sec))
    std_q = float(np.std(deltas_sec))
    median_q = float(np.median(deltas_sec))
    first5 = float(np.mean(deltas_sec[:5]))
    last5 = float(np.mean(deltas_sec[-5:]))
    x = np.arange(len(deltas_sec), dtype=np.float32)
    slope_sec_per_q = float(np.polyfit(x, np.array(deltas_sec, dtype=np.float32), 1)[0])

    valg = row[6]
    pos_counts = Counter(valg[:N_R1])
    pos_fracs = [pos_counts.get(p, 0) / N_R1 for p in "0123"]
    probs = np.array(pos_fracs, dtype=np.float32)
    probs = probs[probs > 0]
    pos_entropy = float(-(probs * np.log(probs)).sum()) if len(probs) else 0.0

    h, m = hour_minute(row[1])
    if h is None:
        hour_sin, hour_cos = 0.0, 0.0
    else:
        frac = (h + m / 60.0) / 24.0
        hour_sin = float(np.sin(2 * np.pi * frac))
        hour_cos = float(np.cos(2 * np.pi * frac))

    side = [
        min(total, TOTAL_TIME_CAP) / TOTAL_TIME_CAP,
        min(mean_q, MEAN_Q_CAP) / MEAN_Q_CAP,
        min(std_q, MEAN_Q_CAP) / MEAN_Q_CAP,
        min(median_q, MEAN_Q_CAP) / MEAN_Q_CAP,
        min(first5, MEAN_Q_CAP) / MEAN_Q_CAP,
        min(last5, MEAN_Q_CAP) / MEAN_Q_CAP,
        max(-1.0, min(1.0, slope_sec_per_q * 1000 / SLOPE_CAP_MS)),
        *pos_fracs,
        pos_entropy,
        hour_sin,
        hour_cos,
    ]
    assert len(side) == SIDE_DIM, len(side)
    return side


def session_sequence(row):
    valg = row[6]
    tider = [int(x) for x in row[7]]
    offered = row[8][0]
    r1 = row[8][1]
    r2 = row[8][2]

    deltas_ms = [tider[0]] + [tider[i] - tider[i - 1] for i in range(1, len(tider))]
    deltas_ms = [max(d, 0) for d in deltas_ms]
    deltas_sec = [d / 1000.0 for d in deltas_ms]
    mean_sec = float(np.mean(deltas_sec))

    steps = []
    for q in range(N_R1):
        steps.append(build_step(offered[q * 4:(q + 1) * 4],
                                max(0, min(3, int(valg[q]))),
                                deltas_sec[q], mean_sec, 0))
    for q in range(N_R2):
        steps.append(build_step(r1[q * 4:(q + 1) * 4],
                                max(0, min(3, int(valg[N_R1 + q]))),
                                deltas_sec[N_R1 + q], mean_sec, 1))
    pos = max(0, min(3, int(valg[N_R1 + N_R2])))
    steps.append(build_step(r2, pos, deltas_sec[N_R1 + N_R2], mean_sec, 2))

    seq = np.array(steps, dtype=np.float32)
    side = np.array(session_side_features(row, deltas_sec), dtype=np.float32)
    return seq, side


def load_sequences():
    with open(SOURCE, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    print(f"loaded {len(rows)} raw rows")

    seqs, sides, genders, ages, moods = [], [], [], [], []
    skipped = 0
    for row in rows:
        if not is_valid(row):
            continue
        try:
            s, side = session_sequence(row)
        except Exception as exc:
            if skipped < 5:
                print(f"  skipping row {row[0]}: {exc}")
            skipped += 1
            continue
        seqs.append(s)
        sides.append(side)
        genders.append(1 if row[5] == "j" else 0)
        ages.append(int(row[3]))
        moods.append(int(row[4]))
    if skipped:
        print(f"  total skipped during sequence build: {skipped}")

    X = np.stack(seqs, axis=0)
    S = np.stack(sides, axis=0)
    g = np.array(genders, dtype=np.int64)
    a = np.array(ages, dtype=np.float32)
    m = np.array(moods, dtype=np.float32)
    print(f"built {X.shape[0]} sequences of shape {X.shape[1:]}, side {S.shape[1:]}")
    return X, S, g, a, m


# ---------- Model ----------

class AttnPool(nn.Module):
    """Learned-query additive attention over a sequence."""

    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(d))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.proj = nn.Linear(d, d)

    def forward(self, h):
        # h: [B, T, d]
        s = torch.tanh(self.proj(h))           # [B, T, d]
        scores = s @ self.q                    # [B, T]
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        return (w * h).sum(dim=1)              # [B, d]


class GRUWithSide(nn.Module):
    def __init__(self, step_dim=STEP_DIM, side_dim=SIDE_DIM,
                 d_model=D_MODEL, hidden=HIDDEN, n_layers=N_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(step_dim, d_model)
        self.in_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            d_model, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        rep = hidden * 2  # bidirectional
        self.pool = AttnPool(rep)
        self.post_norm = nn.LayerNorm(rep)

        self.side_mlp = nn.Sequential(
            nn.Linear(side_dim, rep),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(rep * 2, rep),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_gender = nn.Linear(rep, 1)
        self.head_age    = nn.Linear(rep, 1)
        self.head_mood   = nn.Linear(rep, 1)

    def forward(self, x_seq, x_side):
        h = self.in_dropout(self.input_proj(x_seq))
        out, _ = self.gru(h)
        pooled = self.post_norm(self.pool(out))
        side = self.side_mlp(x_side)
        z = self.fuse(torch.cat([pooled, side], dim=-1))
        return (
            self.head_gender(z).squeeze(-1),
            self.head_age(z).squeeze(-1),
            self.head_mood(z).squeeze(-1),
        )


# ---------- Training ----------

AGE_MIN = 6.0
AGE_SPAN = 62.0
MOOD_SPAN = 60.0


def lr_lambda(ep):
    if ep < WARMUP_EPOCHS:
        return (ep + 1) / WARMUP_EPOCHS
    p = (ep - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * p))


def train_fold(X_tr, S_tr, g_tr, a_tr, m_tr,
               X_te, S_te, g_te, a_te, m_te, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GRUWithSide()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val_bce = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_ep = 0

    a_tr_n = (a_tr - AGE_MIN) / AGE_SPAN
    a_te_n = (a_te - AGE_MIN) / AGE_SPAN
    m_tr_n = m_tr / MOOD_SPAN
    m_te_n = m_te / MOOD_SPAN

    X_tr_t = torch.from_numpy(X_tr)
    S_tr_t = torch.from_numpy(S_tr)
    g_tr_t = torch.from_numpy(g_tr).float()
    a_tr_t = torch.from_numpy(a_tr_n)
    m_tr_t = torch.from_numpy(m_tr_n)
    X_te_t = torch.from_numpy(X_te)
    S_te_t = torch.from_numpy(S_te)
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
            g_logit, a_pred, m_pred = model(X_tr_t[idx], S_tr_t[idx])
            l_g = F.binary_cross_entropy_with_logits(g_logit, g_tr_t[idx])
            l_a = F.mse_loss(a_pred, a_tr_t[idx])
            l_m = F.mse_loss(m_pred, m_tr_t[idx])
            loss = LOSS_W_GENDER * l_g + LOSS_W_AGE * l_a + LOSS_W_MOOD * l_m
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            g_logit, a_pred, m_pred = model(X_te_t, S_te_t)
            l_g = F.binary_cross_entropy_with_logits(g_logit, g_te_t).item()
            l_a = F.mse_loss(a_pred, a_te_t).item()
            l_m = F.mse_loss(m_pred, m_te_t).item()
            g_prob = torch.sigmoid(g_logit).numpy()
            g_acc = ((g_prob >= 0.5).astype(int) == g_te).mean()

        if l_g < best_val_bce:
            best_val_bce = l_g
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_ep = ep

        if ep % 5 == 0 or ep == EPOCHS - 1:
            marker = "  *" if ep == best_ep else ""
            cur_lr = opt.param_groups[0]["lr"]
            print(f"    ep {ep:3d}  lr {cur_lr:.5f}  train_loss {tot / nb:.4f}  "
                  f"val_bce {l_g:.3f}  val_age_mse {l_a:.4f}  "
                  f"val_mood_mse {l_m:.4f}  val_gender_acc {g_acc:.3f}{marker}")

    print(f"    restoring best snapshot from ep {best_ep}  (val_bce={best_val_bce:.4f})")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        g_logit, a_pred, m_pred = model(X_te_t, S_te_t)
        g_prob = torch.sigmoid(g_logit).numpy()
        a_pred_u = a_pred.numpy() * AGE_SPAN + AGE_MIN
        m_pred_u = m_pred.numpy() * MOOD_SPAN
    return g_prob, a_pred_u, m_pred_u


def main():
    X, S, g, a, m = load_sequences()
    N = X.shape[0]
    print(f"X={X.shape}  S={S.shape}  "
          f"boys/girls={int((g == 0).sum())}/{int((g == 1).sum())}")

    n_params = sum(p.numel() for p in GRUWithSide().parameters())
    print(f"GRU params: {n_params:,}  "
          f"(d_model={D_MODEL}, hidden={HIDDEN}x2(bi), layers={N_LAYERS})")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_g = np.zeros(N, dtype=np.float32)
    oof_a = np.zeros(N, dtype=np.float32)
    oof_m = np.zeros(N, dtype=np.float32)

    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, g)):
        print(f"\n--- fold {fold + 1}/5  train={len(tr)}  val={len(va)} ---")
        g_prob, a_pred, m_pred = train_fold(
            X[tr], S[tr], g[tr], a[tr], m[tr],
            X[va], S[va], g[va], a[va], m[va],
            seed=SEED + fold,
        )
        oof_g[va] = g_prob
        oof_a[va] = a_pred
        oof_m[va] = m_pred
        print(f"    cumulative time: {time.time() - t0:.1f}s")

    print()
    print("=" * 64)
    print("BiGRU + ATTN-POOL + SIDE-INFO (5-fold CV)")
    print("=" * 64)
    g_pred = (oof_g >= 0.5).astype(int)
    print(f"  gender   acc={accuracy_score(g, g_pred):.3f}  "
          f"AUC={roc_auc_score(g, oof_g):.3f}  F1={f1_score(g, g_pred):.3f}")
    print(f"  age      MAE={mean_absolute_error(a, oof_a):.2f}  "
          f"R2={r2_score(a, oof_a):+.3f}")
    print(f"  mood     MAE={mean_absolute_error(m, oof_m):.2f}  "
          f"R2={r2_score(m, oof_m):+.3f}")

    np.savez(
        os.path.join(HERE, "gru_oof.npz"),
        gender=oof_g, age=oof_a, mood=oof_m,
    )
    print("\nwrote gru_oof.npz")


if __name__ == "__main__":
    print("Training color-polygraph BiGRU + side-info (5-fold CV)...")
    main()
