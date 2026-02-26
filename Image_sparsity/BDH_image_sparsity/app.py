import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap

from generate_data import load_data
from model import SimpleCNN
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Neuron Atlas Explorer",
    layout="wide"
)

st.title("🧠 CNN Neuron Activation Atlas")

# =====================================================
# STEP 1 — DATA
# =====================================================
st.header("1️⃣ Generate MNIST Data")

if st.button("Generate Dataset"):
    load_data()
    st.success("Images saved!")

# =====================================================
# STEP 2 — LOG ACTIVATIONS
# =====================================================
st.header("2️⃣ Extract Activations")

if st.button("Run Model + Log Activations"):

    images = torch.load("images.pt")

    model = SimpleCNN()
    model.eval()

    acts_all = []

    with torch.no_grad():
        for img in images:
            img = img.unsqueeze(0)
            _ = model(img)
            acts = model._last_activations
            acts_all.append(acts.cpu().numpy())

    acts_all = np.concatenate(acts_all, axis=0)
    np.save("activations.npy", acts_all)

    st.success(f"Saved activations {acts_all.shape}")

# =====================================================
# STEP 3 — CLUSTER NEURONS
# =====================================================
st.header("3️⃣ Cluster Neurons")

if st.button("Cluster Neurons"):

    acts = np.load("activations.npy")
    N, C, H, W = acts.shape

    neuron_vectors = []

    for c in range(C):
        ch = acts[:, c, :, :].reshape(N, -1)
        neuron_vectors.append(ch.mean(axis=0))

    neuron_vectors = np.stack(neuron_vectors)

    kmeans = KMeans(n_clusters=8, random_state=0)
    clusters = kmeans.fit_predict(neuron_vectors)

    np.save("neuron_vectors.npy", neuron_vectors)
    np.save("neuron_clusters.npy", clusters)

    st.success("Neuron clustering done!")

# =====================================================
# STEP 4 — ATLAS VISUALIZATION
# =====================================================
st.header("4️⃣ Neuron Atlas")

if st.button("Build Atlas"):

    vectors = np.load("neuron_vectors.npy")
    clusters = np.load("neuron_clusters.npy")

    embedding = umap.UMAP(
        n_neighbors=10
    ).fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(
        embedding[:,0],
        embedding[:,1],
        c=clusters,
        cmap="tab10"
    )

    plt.colorbar(sc)
    ax.set_title("Neuron Activation Atlas")

    st.pyplot(fig)

# =====================================================
# STEP 5 — CLUSTER INSPECTION
# =====================================================
st.header("5️⃣ Inspect Cluster")

cluster_id = st.slider(
    "Select Cluster",
    0,
    7,
    0
)

if st.button("Show Top Activating Images"):

    acts = np.load("activations.npy")
    clusters = np.load("neuron_clusters.npy")
    images = torch.load("images.pt")

    neurons = np.where(clusters == cluster_id)[0]

    scores = acts[:, neurons].mean(axis=(1,2,3))
    top_idx = scores.argsort()[-5:]

    cols = st.columns(5)

    for col, i in zip(cols, top_idx):
        col.image(
            images[i][0],
            caption=f"{scores[i]:.2f}",
            use_column_width=True
        )
