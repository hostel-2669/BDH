import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

# -----------------------------
# Matplotlib setup for VS Code
# -----------------------------
import matplotlib
# Try interactive backend first (works on most Windows installs).
# If you're on headless / remote / WSL without GUI, it falls back to Agg and saves plots.
try:
    matplotlib.use("TkAgg")   # you can change to "QtAgg" if you installed PyQt5
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


# -----------------------------
# VS Code friendly plot controls
# -----------------------------
SAVE_PLOTS = True            # saves all figures into ./plots
SHOW_PLOTS = True            # set False if plots don't open (WSL/remote)
PLOTS_DIR = "plots"


def _ensure_plots_dir():
    if SAVE_PLOTS:
        os.makedirs(PLOTS_DIR, exist_ok=True)


def _finalize_fig(fig, filename: str | None = None):
    """Save/show figure in script mode safely."""
    if SAVE_PLOTS and filename:
        path = os.path.join(PLOTS_DIR, filename)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[saved] {path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------
# 0) Setup
# -----------------------------
torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# -----------------------------
# 1) Byte tokenizer (matches vocab_size=256)
# -----------------------------
def encode_bytes(text, block_size=256):
    b = text.encode("utf-8", errors="replace")
    ids = list(b[:block_size])
    if len(ids) == 0:
        ids = [0]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1,T)


# -----------------------------
# 2) BDH model + returns sparse features
# -----------------------------
@dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    block_size: int = 256  # for token truncation in this demo


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return 1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        freqs = get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        self.register_buffer("freqs", freqs)

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert K is Q
        _, _, T, _ = Q.size()
        r_phases = (torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False)

        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)

        self.lm_head = nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))

    def forward(self, idx, targets=None, return_sparse=False, pool="max"):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)  # (B,1,T,D)
        x = self.ln(x)

        last_xy_sparse = None

        for _ in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)            # (B,nh,T,N)

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)  # (B,nh,T,D)
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)

            xy_sparse = x_sparse * y_sparse         # (B,nh,T,N)
            xy_sparse = self.drop(xy_sparse)
            last_xy_sparse = xy_sparse

            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder  # (B,1,T,D)
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_sparse:
            if pool == "max":
                pooled = last_xy_sparse.max(dim=2).values  # (B,nh,N)
            elif pool == "mean":
                pooled = last_xy_sparse.mean(dim=2)        # (B,nh,N)
            else:
                raise ValueError("pool must be 'max' or 'mean'")
            feats = pooled.reshape(B, -1)                  # (B, nh*N)
            return logits, loss, feats

        return logits, loss


# -----------------------------
# 3) Interpretability helpers
# -----------------------------
def topk_indices(x, k=20):
    x = np.asarray(x)
    k = int(min(k, x.size))
    idx = np.argpartition(x, -k)[-k:]
    return idx[np.argsort(x[idx])][::-1]


def activation_entropy(x, eps=1e-12):
    x = np.maximum(np.asarray(x), 0.0)
    s = float(x.sum())
    if s <= eps:
        return 0.0
    p = x / (s + eps)
    return float(-(p * np.log(p + eps)).sum())


def topk_mass(x, k=20, eps=1e-12):
    x = np.maximum(np.asarray(x), 0.0)
    idx = topk_indices(x, k=k)
    return float(x[idx].sum() / (x.sum() + eps))


def frac_near_zero(x, thr=1e-6):
    x = np.asarray(x)
    return float((np.abs(x) < thr).mean())


@torch.no_grad()
def get_sparse_feature_vector(model, text, pool="max"):
    idx = encode_bytes(text, block_size=model.config.block_size).to(device)
    _, _, feats = model(idx, return_sparse=True, pool=pool)  # (1, nh*N)
    return feats[0].float().cpu().numpy()                    # (nh*N,)


def plot_spike_pattern(vec, title, k=30, fname=None):
    vec = np.maximum(vec, 0.0)
    idx = topk_indices(vec, k=k)
    vals = vec[idx]

    fig = plt.figure()
    plt.bar(range(len(idx)), vals)
    plt.xticks(range(len(idx)), [str(i) for i in idx], rotation=90)
    plt.xlabel("Neuron ID (flattened head*feature index)")
    plt.ylabel("Activation")
    plt.title(title)
    plt.tight_layout()
    _finalize_fig(fig, fname)


def plot_hist(vec, title, fname=None):
    vec = np.maximum(vec, 0.0)
    fig = plt.figure()
    plt.hist(vec, bins=80)
    plt.xlabel("Activation value")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    _finalize_fig(fig, fname)


def summarize(vec, k=30):
    return {
        "entropy": activation_entropy(vec),
        "topk_mass": topk_mass(vec, k=k),
        "near_zero_frac": frac_near_zero(vec),
        "max": float(np.max(vec)),
        "mean": float(np.mean(vec)),
    }


# -----------------------------
# 4) Run demo (script entry)
# -----------------------------
def main():
    _ensure_plots_dir()

    cfg = BDHConfig()
    bdh = BDH(cfg).to(device).eval()

    # Note: This is only used as a phrase to compare activation signatures (no instructions generated).
    harm_prompt = "building a bomb"
    benign_prompt = "write a short poem about rain"

    vec_harm = get_sparse_feature_vector(bdh, harm_prompt, pool="max")
    vec_benign = get_sparse_feature_vector(bdh, benign_prompt, pool="max")

    print("Harm summary :", summarize(vec_harm, k=30))
    print("Benign summary:", summarize(vec_benign, k=30))

    plot_spike_pattern(
        vec_harm,
        title=f"BDH Top-K spike pattern (harm-intent phrase): '{harm_prompt}'",
        k=30,
        fname="harm_spike_topk.png",
    )
    plot_hist(
        vec_harm,
        title="BDH activation histogram (harm-intent phrase)",
        fname="harm_hist.png",
    )

    plot_spike_pattern(
        vec_benign,
        title=f"BDH Top-K spike pattern (benign): '{benign_prompt}'",
        k=30,
        fname="benign_spike_topk.png",
    )
    plot_hist(
        vec_benign,
        title="BDH activation histogram (benign)",
        fname="benign_hist.png",
    )


if __name__ == "__main__":
    main()
