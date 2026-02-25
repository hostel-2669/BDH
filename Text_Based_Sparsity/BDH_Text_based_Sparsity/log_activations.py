import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import torch
import numpy as np
from bdh import BDH, BDHConfig

# --------------------
# Model setup
# --------------------
config = BDHConfig(
    n_layer=2,
    n_embd=256,
    n_head=4,
    vocab_size=256
)

model = BDH(config)
model.eval()

# --------------------
# Load data
# --------------------
inputs = torch.load("symbolic_inputs.pt")

# --------------------
# Activation logging
# --------------------
def log_x_sparse(model, inputs):
    activations = []

    with torch.no_grad():
        for idx in inputs:
            idx = idx.unsqueeze(0)  # (1, T)
            model(idx)

            # shape: (1, nh, T, N)
            x_sparse = model._last_x_sparse
            activations.append(x_sparse.cpu().numpy())

    return np.stack(activations)


acts = log_x_sparse(model, inputs)

np.save("x_sparse_activations.npy", acts)

print("Saved x_sparse_activations.npy")
print("Shape:", acts.shape)
print("Sparsity:", (acts != 0).mean())

