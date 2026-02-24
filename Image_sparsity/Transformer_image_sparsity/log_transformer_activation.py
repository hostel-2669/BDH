# log_transformer_activations.py
import torch
import numpy as np
from model_transformer import SimpleViT

images = torch.load("images.pt")

model = SimpleViT()
model.eval()

all_attn = []

with torch.no_grad():
    for img in images:
        img = img.unsqueeze(0)
        _ = model(img)

        attn = model._last_attention  # (1, heads, N, N)
        all_attn.append(attn.cpu().numpy())

all_attn = np.concatenate(all_attn, axis=0)
np.save("attentions.npy", all_attn)

print("Saved attention maps:", all_attn.shape)

