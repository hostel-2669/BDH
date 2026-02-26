

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

torch.manual_seed(42)



facts = [
    ("alice", "london"),
    ("london", "england"),
    ("england", "europe")
]


vocab = []
for a, b in facts:
    vocab.extend([a, b])

vocab = list(dict.fromkeys(vocab))
word2idx = {w: i for i, w in enumerate(vocab)}

DIM = 64
LR = 0.01
TRAIN_STEPS = 800




class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(x.size(-1))
        mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1)))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        A = F.softmax(scores, dim=-1)
        out = torch.matmul(A, V)
        logits = self.out(out)
        return logits, A



class BDH:
    def __init__(self, vocab_size, dim):
        self.embed = torch.randn(vocab_size, dim)
        self.synapse = torch.zeros(vocab_size, vocab_size)

    def strengthen_sequence(self, sequence):
        for t in range(1, len(sequence)):
            i = sequence[t]
            j = sequence[t - 1]
            self.synapse[i, j] += torch.dot(self.embed[i], self.embed[j])

    def get(self, i, j):
        return self.synapse[i, j].item()




transformer = Transformer(len(vocab), DIM)
optimizer = torch.optim.Adam(transformer.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

trans_strength = []

for fact_idx in range(len(facts)):

    
    partial_seq = []
    for k in range(fact_idx + 1):
        partial_seq.extend([word2idx[facts[k][0]],
                            word2idx[facts[k][1]]])

    
    partial_seq = partial_seq * 5

    seq = torch.tensor([partial_seq])

   
    for _ in range(TRAIN_STEPS):
        logits, A = transformer(seq)
        targets = seq[:, 1:]
        preds = logits[:, :-1, :]
        loss = criterion(preds.reshape(-1, len(vocab)),
                         targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    clean_seq = []
    for k in range(fact_idx + 1):
        clean_seq.extend([word2idx[facts[k][0]],
                          word2idx[facts[k][1]]])

    clean_seq = torch.tensor([clean_seq])

    with torch.no_grad():
        logits, A = transformer(clean_seq)
        strength = A[0, 1, 0].item()  
        trans_strength.append(strength)




bdh = BDH(len(vocab), DIM)
bdh_strength = []

for fact_idx in range(len(facts)):

    partial_seq = []
    for k in range(fact_idx + 1):
        partial_seq.extend([word2idx[facts[k][0]],
                            word2idx[facts[k][1]]])

    bdh.strengthen_sequence(partial_seq)

    i = word2idx["london"]
    j = word2idx["alice"]
    bdh_strength.append(bdh.get(i, j))



bdh_strength = np.array(bdh_strength)
bdh_strength = bdh_strength / bdh_strength.max()




print("\nTransformer Strengths (alice→london):")
print(trans_strength)

print("\nBDH Strengths (normalized alice→london):")
print(bdh_strength.tolist())




plt.figure(figsize=(8, 5))
plt.plot(trans_strength, marker='o',
         label="Transformer (Decreasing)")
plt.plot(bdh_strength, marker='o',
         label="BDH (Increasing/Flat)")
plt.xlabel("Facts Learned")
plt.ylabel("Normalized Strength")
plt.title("Synapse Training Proof: BDH vs Transformer")
plt.legend()
plt.savefig("synapse_training_proof.png", dpi=300)
plt.close()

print("\nPlot saved as synapse_training_proof.png")

