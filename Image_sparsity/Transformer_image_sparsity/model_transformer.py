# model_transformer.py
import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, emb_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            1,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)        # (B, E, H', W')
        x = x.flatten(2)        # (B, E, N)
        x = x.transpose(1, 2)   # (B, N, E)
        return x


class SimpleViT(nn.Module):
    def __init__(self, emb_dim=64, n_heads=4):
        super().__init__()

        self.patch_embed = PatchEmbedding()

        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=n_heads,
            batch_first=True
        )

        self._last_attention = None

    def forward(self, x):
        x = self.patch_embed(x)

        out, attn_weights = self.attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=False
        )

        # Shape: (B, heads, N, N)
        self._last_attention = attn_weights.detach()

        return out

