# log_activations.py
import torch
import numpy as np
from model import SimpleCNN

images = torch.load("images.pt")

model = SimpleCNN()
model.eval()

all_acts = []

with torch.no_grad():
    for img in images:
        img = img.unsqueeze(0)
        _ = model(img)
        acts = model._last_activations  # (1, C, H, W)
        all_acts.append(acts.cpu().numpy())

all_acts = np.concatenate(all_acts, axis=0)
np.save("activations.npy", all_acts)

print("Saved activations:", all_acts.shape)

