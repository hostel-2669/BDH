# inspect_cluster.py
import numpy as np
import torch
import matplotlib.pyplot as plt

CLUSTER_ID = 4

acts = np.load("activations.npy")
clusters = np.load("neuron_clusters.npy")
images = torch.load("images.pt")

neurons = np.where(clusters == CLUSTER_ID)[0]

scores = acts[:, neurons].mean(axis=(1,2,3))
top_idx = scores.argsort()[-5:]

for i in top_idx:
    plt.imshow(images[i][0], cmap="gray")
    plt.title(f"Activation score: {scores[i]:.2f}")
    plt.axis("off")
    plt.show()
