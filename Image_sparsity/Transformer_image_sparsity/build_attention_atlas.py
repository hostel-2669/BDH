# build_attention_atlas.py
import numpy as np
import umap
import matplotlib.pyplot as plt

vectors = np.load("head_vectors.npy")
clusters = np.load("head_clusters.npy")

embedding = umap.UMAP(n_neighbors=5).fit_transform(vectors)

plt.figure(figsize=(6,6))
plt.scatter(
    embedding[:,0],
    embedding[:,1],
    c=clusters,
    cmap="tab10"
)

plt.colorbar(label="Cluster ID")
plt.title("Transformer Attention Head Atlas")
plt.show()
