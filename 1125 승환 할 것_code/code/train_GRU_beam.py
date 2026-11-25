# -*- coding: utf-8 -*-
"""
[GC 1124] 빔 각도를 입력에 넣은 GRU 채널 예측기 학습 (승환 제안).

    python tools/build_beam_dataset.py                        # 데이터 먼저 (data_ready_beam/)
    python train_GRU_beam.py                                  # 각도 넣은 모델
    python train_GRU_beam.py --no-angle                       # 같은 코드·같은 데이터로 각도 없는 기준 모델 (비교용)

모델은 Summit 판 SimpleChannelRNN (6_Summit데모_1110/channel-sounder/train_GRU.py) 그대로이고 입력만 늘었다:
    입력 = [토큰 15 (|h|·cos·sin × 5탭) , age 1 , 빔 각도 특징 2 (sin θ, cos θ)]   ← RU 마다 따로 (RU 하나 = 시퀀스 하나)
예측 롤아웃 중 미래 스텝의 각도:
    hold (기본)  예측 시점 t0 의 각도를 그대로 — 실시간 대시보드가 미래 각도를 모르므로 이것과 맞춘다
    true         실제 미래 각도 — 빔 계획을 미리 안다고 볼 때의 상한

저장: checkpoints_beam/<이름>.pt — hj_common.NumpyPredictor 가 그대로 읽는다 (config 에 beam_feat_dim·future_angle).
대시보드: python live_demo/live_dashboard.py --predictor-model checkpoints_beam/<이름>.pt   (빔 상태는 UDP 5008)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))
from beam import beam_features as bf  # noqa: E402

# Summit train_GRU.py 와 같은 값
TAPS = 5                 # 5 × 5 ms = 25 ms 한 스텝
TOK_DIM = 3 * TAPS
GAIN_SCALE = 20.0
EPS_NORM = 1e-6
PRED_STEPS = 5           # +25 ~ +125 ms
LOSS_GAMMA = 0.5
MAX_AGE = 32
HIDDEN = 64


def tokens(h_taps: torch.Tensor) -> torch.Tensor:
    return torch.cat([h_taps.abs() * GAIN_SCALE, torch.cos(h_taps.angle()), torch.sin(h_taps.angle())], -1).float()


def post(tok: torch.Tensor) -> torch.Tensor:
    g = F.softplus(tok[..., :TAPS])
    c, s = tok[..., TAPS:2 * TAPS], tok[..., 2 * TAPS:]
    d = torch.sqrt(c * c + s * s + EPS_NORM)
    return torch.cat([g, c / d, s / d], -1)


def to_complex(tok: torch.Tensor) -> torch.Tensor:
    g = tok[..., :TAPS] / GAIN_SCALE
    return torch.complex(g * tok[..., TAPS:2 * TAPS], g * tok[..., 2 * TAPS:])


class Seqs(torch.utils.data.Dataset):
    """(창, RU) 한 쌍 = 시퀀스 하나. 반환 tok (S,15) · feat (S,2)."""

    def __init__(self, npz: Path, train: bool):
        z = np.load(npz)
        self.X, self.A = z["X"], z["A"]
        self.train = train
        self.S = self.X.shape[1] // TAPS

    def __len__(self):
        return self.X.shape[0] * 3

    def __getitem__(self, i):
        w, ru = divmod(i, 3)
        h = torch.from_numpy(np.ascontiguousarray(self.X[w, :, ru]))
        if self.train:                                       # Summit 과 같은 전역 위상 증강
            h = h * torch.polar(torch.ones(()), torch.rand(()) * 2 * math.pi).to(h.dtype)
        tok = tokens(h[: self.S * TAPS].reshape(self.S, TAPS))
        feat = torch.from_numpy(bf.angle_feats(bf.step_mean_deg(self.A[w, :, ru], TAPS)))
        return tok, feat


class BeamGRU(nn.Module):
    """Summit SimpleChannelRNN + 빔 특징. 매개변수 이름(rnn.*, out_proj.*, init_tok)은 NumpyPredictor 와 맞춘다."""

    def __init__(self, beam_dim: int):
        super().__init__()
        self.beam_dim = beam_dim
        self.init_tok = nn.Parameter(torch.randn(TOK_DIM) * 0.02)
        self.rnn = nn.GRU(TOK_DIM + 1 + beam_dim, HIDDEN, batch_first=True)
        self.out_proj = nn.Linear(HIDDEN, TOK_DIM)

    def pack(self, tok, age, feat):
        parts = [tok, age] + ([feat] if self.beam_dim else [])
        return torch.cat(parts, -1)

    def rollout(self, tok, feat, t0, future_angle="hold", K=PRED_STEPS):
        """tok (B,S,D) GT · feat (B,S,2). GT[0..t0] 을 먹고 GT[t0+1..t0+K] 예측 (후처리 토큰) (B,K,D)."""
        B, S, D = tok.shape
        z1 = torch.zeros(B, 1, 1, device=tok.device)
        f0 = feat[:, :1]
        seq = [self.pack(self.init_tok.view(1, 1, D).expand(B, 1, D), z1, f0)]
        if t0 > 0:
            seq.append(self.pack(tok[:, :t0], torch.zeros(B, t0, 1, device=tok.device), feat[:, :t0]))
        _, h = self.rnn(torch.cat(seq, 1))
        x, out = tok[:, t0], []
        for i in range(K):
            age = torch.full((B, 1), min(i, MAX_AGE) / MAX_AGE, device=tok.device)
            f = feat[:, t0] if future_angle == "hold" else feat[:, min(t0 + i, S - 1)]
            y, h = self.rnn(self.pack(x, age, f).unsqueeze(1), h)
            p = post(self.out_proj(y[:, 0]))
            out.append(p)
            x = p.detach()
        return torch.stack(out, 1)


def run_loss(model, tok, feat, future_angle):
    S = tok.shape[1]
    t0 = int(torch.randint(0, S - PRED_STEPS, ()).item())
    pred = model.rollout(tok, feat, t0, future_angle)
    tgt = tok[:, t0 + 1:t0 + 1 + PRED_STEPS]
    w = torch.tensor([LOSS_GAMMA ** i for i in range(PRED_STEPS)], device=tok.device)
    mse = ((pred - tgt) ** 2).mean(dim=(0, 2))
    return (w * mse).sum() / w.sum()


@torch.no_grad()
def evaluate(model, ds, device, future_angle, t0_stride=4):
    """모든 시퀀스 × t0 격자에서 스텝별 NMSE [dB] (+25 … +125 ms)."""
    model.eval()
    err = np.zeros(PRED_STEPS)
    pw = np.zeros(PRED_STEPS)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
    for tok, feat in loader:
        tok, feat = tok.to(device), feat.to(device)
        S = tok.shape[1]
        for t0 in range(8, S - PRED_STEPS, t0_stride):
            p = to_complex(model.rollout(tok, feat, t0, future_angle))
            g = to_complex(tok[:, t0 + 1:t0 + 1 + PRED_STEPS])
            err += ((p - g).abs() ** 2).sum(dim=(0, 2)).cpu().numpy()
            pw += (g.abs() ** 2).sum(dim=(0, 2)).cpu().numpy()
    model.train()
    return 10 * np.log10(err / pw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CODE_ROOT / "data_ready_beam"))
    ap.add_argument("--no-angle", action="store_true", help="각도 없이 (기준 모델)")
    ap.add_argument("--future-angle", choices=["hold", "true"], default="hold")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--name", default="")
    ap.add_argument("--save-dir", default=str(CODE_ROOT / "checkpoints_beam"))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(a.data)
    tr, va, te = (Seqs(data / p / "X.npz", p == "train") for p in ("train", "validation", "test"))
    beam_dim = 0 if a.no_angle else bf.BEAM_FEAT_DIM
    model = BeamGRU(beam_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    loader = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=False)
    name = a.name or ("gru_noangle" if a.no_angle else f"gru_angle_{a.future_angle}")
    save = Path(a.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    print(f"[TRAIN] {name} · 각도 {'없음' if a.no_angle else a.future_angle} · train {len(tr)} 시퀀스 · {dev}")

    best, best_ep, t_start = float("inf"), -1, time.time()
    for ep in range(1, a.epochs + 1):
        for tok, feat in loader:
            loss = run_loss(model, tok.to(dev), feat.to(dev), a.future_angle)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep % a.eval_every == 0 or ep == a.epochs:
            v = evaluate(model, va, dev, a.future_angle)
            score = float(np.mean(v))
            mark = ""
            if score < best:
                best, best_ep, mark = score, ep, " *"
                torch.save({
                    "epoch": ep, "model": model.state_dict(),
                    "config": {"rnn_type": "GRU", "num_layers": 1, "hidden_size": HIDDEN, "samples_per_step": TAPS,
                               "tok_dim": TOK_DIM, "gain_scale": GAIN_SCALE, "eps_norm": EPS_NORM,
                               "pred_steps": PRED_STEPS, "loss_gamma": LOSS_GAMMA, "use_age_feature": True,
                               "max_age_feature": MAX_AGE, "beam_feat_dim": beam_dim,
                               "future_angle": a.future_angle, "data": str(data), "seed": a.seed},
                }, save / f"{name}.pt")
            print(f"  ep {ep:4d} loss {loss.item():.4f} · val NMSE {' '.join(f'{x:6.1f}' for x in v)} dB{mark}"
                  f" · {time.time() - t_start:.0f} s")

    ck = torch.load(save / f"{name}.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    t = evaluate(model, te, dev, a.future_angle)
    res = {"name": name, "best_epoch": best_ep, "angle": not a.no_angle, "future_angle": a.future_angle,
           "test_nmse_db": {f"+{25 * (k + 1)}ms": round(float(x), 2) for k, x in enumerate(t)},
           "val_nmse_db_mean": round(best, 2)}
    (save / f"{name}_result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[TEST] {name} (ep {best_ep}) NMSE " + " · ".join(f"{k} {v} dB" for k, v in res["test_nmse_db"].items()))
    print(f"[SAVE] {save / (name + '.pt')}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                        # noqa: BLE001
        pass
    main()
