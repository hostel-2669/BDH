import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.cluster import KMeans

HEAD_ID = 1          # which attention head to analyze

# -----------------------------
# Config
# -----------------------------
PERCENTILE = 95          # top 5% = firing
N_CLUSTERS = 8           # adjust later if needed

# -----------------------------
# Load activations
# -----------------------------
# Shape: (num_inputs, 1, nh, T, N)
acts = np.load("x_sparse_activations.npy")

num_inputs, _, nh, T, N = acts.shape
print("Loaded activations:", acts.shape)

# Remove the singleton dim
acts = acts[:, 0]  # (num_inputs, nh, T, N)

# -----------------------------
# Build per-neuron firing masks
# -----------------------------
# We will create neuron vectors of shape:
# (nh * N, num_inputs * T)

neuron_vectors = []

h = HEAD_ID
for n in range(N):
    values = acts[:, h, :, n].reshape(-1)  # (num_inputs * T,)
    threshold = np.percentile(values, PERCENTILE)
    firing = (values >= threshold).astype(np.float32)
    neuron_vectors.append(firing)


neuron_vectors = np.stack(neuron_vectors)  # (nh*N, num_inputs*T)

# -----------------------------
# Sanity check sparsity
# -----------------------------
sparsity = neuron_vectors.mean()
print(f"Post-threshold sparsity: {sparsity:.4f}")

# -----------------------------
# Cluster neurons by behavior
# -----------------------------
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
clusters = kmeans.fit_predict(neuron_vectors)

# -----------------------------
# Save results
# -----------------------------
np.save("neuron_vectors.npy", neuron_vectors)
np.save("neuron_clusters.npy", clusters)

print("Saved neuron_vectors.npy")
print("Saved neuron_clusters.npy")

