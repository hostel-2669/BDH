# cluster_neurons.py
import numpy as np
from sklearn.cluster import KMeans

acts = np.load("activations.npy")  # (N, C, H, W)
N, C, H, W = acts.shape

neuron_vectors = []

for c in range(C):
    channel_acts = acts[:, c, :, :].reshape(N, -1)
    mean_pattern = channel_acts.mean(axis=0)
    neuron_vectors.append(mean_pattern)

neuron_vectors = np.stack(neuron_vectors)

kmeans = KMeans(n_clusters=8, random_state=0)
clusters = kmeans.fit_predict(neuron_vectors)

np.save("neuron_vectors.npy", neuron_vectors)
np.save("neuron_clusters.npy", clusters)

print("Neuron clusters saved")

