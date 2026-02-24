# cluster_heads.py
import numpy as np
from sklearn.cluster import KMeans

attn = np.load("attentions.npy")  # (B, H, N, N)
B, H, N, _ = attn.shape

head_vectors = []

for h in range(H):
    head_maps = attn[:, h, :, :]  # (B, N, N)
    mean_pattern = head_maps.mean(axis=0)
    head_vectors.append(mean_pattern.flatten())

head_vectors = np.stack(head_vectors)

kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(head_vectors)

np.save("head_vectors.npy", head_vectors)
np.save("head_clusters.npy", clusters)

print("Head clusters saved")
