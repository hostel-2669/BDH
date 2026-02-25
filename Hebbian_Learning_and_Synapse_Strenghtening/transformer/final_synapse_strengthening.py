

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict

torch.manual_seed(42)



facts = [
    ["alice", "lives", "in", "london"],
    ["london", "is", "in", "england"],
    ["england", "is", "in", "europe"]
]

tokens = []
fact_boundaries = []
for fact in facts:
    fact_boundaries.append(len(tokens))
    tokens.extend(fact)

vocab = sorted(list(set(tokens)))
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}

sequence = torch.tensor([[word2idx[w] for w in tokens]])



class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward_with_tracking(self, idx):

        x = self.embed(idx)
        B, T, D = x.shape

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(D)

        mask = torch.tril(torch.ones(T,T))
        scores = scores.masked_fill(mask==0, float("-inf"))

        attention = F.softmax(scores, dim=-1)

        out = torch.matmul(attention, V)
        out = self.out_proj(out)

        snapshots = []

        for t in range(T):
            A_t = attention[0, :t+1, :t+1].detach()

            entropy = -(A_t * torch.log(A_t+1e-9)).sum(dim=-1).mean()
            energy = A_t.abs().sum()

            snapshots.append({
                "timestep": t,
                "attention": A_t,
                "entropy": entropy.item(),
                "energy": energy.item()
            })

        return snapshots




model = SimpleTransformer(len(vocab))
snapshots = model.forward_with_tracking(sequence)



pairs = [
    ("alice","london"),
    ("london","england"),
    ("england","europe")
]

pair_positions = {
    ("alice","london"): (0, 3),
    ("london","england"): (4, 7),
    ("england","europe"): (8, 11)
}

pair_time_series = defaultdict(list)

for snap in snapshots:
    t = snap["timestep"]
    A = snap["attention"]

    for pair, (i, j) in pair_positions.items():
        if i <= t and j <= t:
            pair_time_series[pair].append(A[j, i].item())
        else:
            pair_time_series[pair].append(0.0)


plt.figure(figsize=(10,6))
for pair, values in pair_time_series.items():
    plt.plot(values, marker='o', label=f"{pair[0]} → {pair[1]}")

for b in fact_boundaries[1:]:
    plt.axvline(b, linestyle="--")

plt.title("Transformer Baseline: Attention Links Over Time\n(Softmax-normalized 'synapses')")
plt.xlabel("Token Position")
plt.ylabel("Synapse Strength (Attention Prob)")
plt.legend()

plt.savefig("1_attention_over_time.png", dpi=300, bbox_inches="tight")
plt.close()


final_attention = snapshots[-1]["attention"].numpy()

plt.figure(figsize=(8,6))
plt.imshow(final_attention, cmap="YlOrRd")
plt.colorbar(label="Attention prob A(i,j)")
plt.title("Final Transformer Attention\nA(i,j)=softmax(QK^T/√d)")
plt.xlabel("Token Position j")
plt.ylabel("Token Position i")

plt.savefig("2_final_attention_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()


heat_data = np.array([pair_time_series[p] for p in pairs])

plt.figure(figsize=(10,4))
plt.imshow(heat_data, aspect="auto", cmap="YlOrRd")
plt.colorbar(label="Attention prob")
plt.yticks(range(len(pairs)),
           [f"{p[0]} → {p[1]}" for p in pairs])
plt.xlabel("Processing Step (Token Position)")
plt.title("Transformer: Temporal Evolution of Attention Links")

plt.savefig("3_temporal_evolution.png", dpi=300, bbox_inches="tight")
plt.close()



final_values = [pair_time_series[p][-1] for p in pairs]

plt.figure(figsize=(8,5))
bars = plt.bar(range(len(pairs)), final_values)

plt.xticks(range(len(pairs)),
           [f"{p[0]}\n↓\n{p[1]}" for p in pairs])

plt.ylabel("Final Attention prob")
plt.title("Transformer Final Link Strengths\n(after processing all facts)")

for i,v in enumerate(final_values):
    plt.text(i, v+0.005, f"{v:.5f}", ha="center")

plt.savefig("4_final_link_strengths.png", dpi=300, bbox_inches="tight")
plt.close()



energies = [s["energy"] for s in snapshots]
entropies = [s["entropy"] for s in snapshots]

plt.figure()
plt.plot(energies)
plt.title("Transformer Total Attention Energy Over Time")
plt.xlabel("Timestep")
plt.ylabel("Total Attention Energy")
plt.savefig("5_energy_over_time.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(entropies)
plt.title("Transformer Attention Entropy Over Time")
plt.xlabel("Timestep")
plt.ylabel("Entropy")
plt.savefig("6_entropy_over_time.png", dpi=300, bbox_inches="tight")
plt.close()



with open("7_interference_analysis.txt", "w") as f:
    f.write("Interference Analysis:\n")
    for pair in pairs:
        values = pair_time_series[pair]
        peak = max(values)
        final = values[-1]
        delta = final - peak
        f.write(f"{pair[0]} → {pair[1]} : Peak={peak:.4f}  Final={final:.4f}  Δ={delta:.4f}\n")

print("All transformer synapse-strengthening outputs saved successfully.")
