
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import umap
import matplotlib.pyplot as plt

# Load data
neuron_vectors = np.load("neuron_vectors.npy")
clusters = np.load("neuron_clusters.npy")

# Dimensionality reduction
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
coords = reducer.fit_transform(neuron_vectors)

np.save("atlas_coords.npy", coords)

# Quick static visualization
plt.figure(figsize=(8, 8))
plt.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap="tab10", s=5)
plt.title("Activation Atlas (Sparse Latent Neurons)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()

