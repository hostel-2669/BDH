import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np

# ============================================================
# CONFIG
# ============================================================
CLUSTER_ID = 6     # change this to inspect other clusters
HEAD_ID = 1        # must match the head used in clustering
TOP_K = 10         # number of top activating inputs to show

# ============================================================
# TOKEN DECODING (MUST MATCH generate_data.py EXACTLY)
# ============================================================
REVERSE_TOKEN_MAP = {
    0: "x",
    1: "y",
    2: "n",
    3: "+",
    4: "-",
    5: "*",
    6: "/",
    7: "=",
    8: "1",
    9: "2",
    10: "3",
    11: "4",
    12: "5",
    13: "6",
    14: "7",
    15: "8",
    16: "9",
}

def decode_equation(token_tensor):
    """
    Decode a length-5 symbolic equation:
    [var, op, num1, '=', num2]
    """
    tokens = token_tensor.tolist()
    symbols = [REVERSE_TOKEN_MAP[t] for t in tokens]
    return f"{symbols[0]} {symbols[1]} {symbols[2]} {symbols[3]} {symbols[4]}"

# ============================================================
# LOAD DATA
# ============================================================
inputs = torch.load("symbolic_inputs.pt")

# Shape: (num_inputs, 1, nh, T, N)
acts = np.load("x_sparse_activations.npy")

# Shape: (num_neurons,) — single-head clustering
clusters = np.load("neuron_clusters.npy")

num_inputs, _, nh, T, N = acts.shape
acts = acts[:, 0]  # remove singleton dim → (num_inputs, nh, T, N)

# ============================================================
# IDENTIFY NEURONS IN THIS CLUSTER
# ============================================================
cluster_neurons = np.where(clusters == CLUSTER_ID)[0]

print(f"\nCluster {CLUSTER_ID} has {len(cluster_neurons)} neurons")
print("Top activating inputs for this cluster:\n")

if len(cluster_neurons) == 0:
    raise ValueError("Selected cluster has no neurons — choose another CLUSTER_ID.")

# ============================================================
# COMPUTE CLUSTER ACTIVATION PER INPUT
# ============================================================
cluster_scores = []

for i in range(num_inputs):
    # activations for one input, chosen head
    a = acts[i, HEAD_ID]              # shape: (T, N)
    a_cluster = a[:, cluster_neurons] # (T, |cluster|)
    score = a_cluster.sum()
    cluster_scores.append(score)

cluster_scores = np.array(cluster_scores)

# ============================================================
# SHOW TOP-ACTIVATING INPUTS
# ============================================================
top_indices = np.argsort(cluster_scores)[-TOP_K:][::-1]

for idx in top_indices:
    print(f"Input {idx}, score = {cluster_scores[idx]:.2f}")
    print(decode_equation(inputs[idx]))
    print("-" * 50)

