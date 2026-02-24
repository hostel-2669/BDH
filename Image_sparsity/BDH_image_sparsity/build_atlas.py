# build_atlas.py
import numpy as np
import umap
import matplotlib.pyplot as plt

vectors = np.load("neuron_vectors.npy")
clusters = np.load("neuron_clusters.npy")

embedding = umap.UMAP(n_neighbors=10).fit_transform(vectors)

plt.figure(figsize=(6,6))
plt.scatter(embedding[:,0], embedding[:,1], c=clusters, cmap="tab10")
plt.colorbar(label="Cluster ID")
plt.title("Image Neuron Activation Atlas")
plt.show()
