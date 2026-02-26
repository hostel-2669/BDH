#  Post-Transformer Frontier AI  
## Bi-Directional Hybrid (BDH) Architecture  
**Kriti’26 Research Project**

---

##  What We Built

We designed and evaluated a **Bi-Directional Hybrid (BDH)** architecture as a structured sparse alternative to dense Transformer models. BDH activates only **2.5–5% of neurons per input**, maintains a **constant-size memory state (O(1))**, and enables **inference-time Hebbian synapse strengthening** without backpropagation.

We built a full experimental framework including sparse activation atlases, cross-lingual monosemantic probing, Hebbian learning analysis, long-context memory stress testing, and harmful prompt entropy evaluation to rigorously compare BDH against Transformer baselines.

---

##  Key Insights (BDH vs Transformer)

---

## 1️ Structured Sparse Activation

## Structured Sparse Activation

### BDH Activation Atlas

<img src="https://github.com/user-attachments/assets/ca503b4f-c87f-4203-a9ea-612d25f75384" width="866" height="920">

BDH forms structured sparse clusters with only 2.5–3% activation per input.

### Transformer Activation Atlas

<img width="866" height="920" alt="transformer_neuron_atlas" src="https://github.com/user-attachments/assets/bc1f9211-1ec2-4f68-a91c-ab34b12ba4f7" />


BDH forms clear geometric neuron clusters with only 2.5–3% activation per input, while Transformers exhibit dense, overlapping, and entangled representations (~99% firing).

---

## 2️ Firing Rate Distribution

<img width="866" height="920" alt="sparse_neuron_activation" src="https://github.com/user-attachments/assets/df4541ea-b217-4dbb-a413-2ddc6748cdbd" />

 
The histogram confirms stable sparse activation (2.5–3%) without dense tails, contrasting with near-universal Transformer activation.

---

## 3️ Inference-Time Hebbian Learning

<img width="2667" height="1176" alt="3_temporal_evolution" src="https://github.com/user-attachments/assets/8d66265c-7167-4894-b7f8-94bb5e2626eb" />


Sequential fact processing shows synapse strengths increasing exactly when related concepts co-occur, demonstrating online learning without gradient updates.

---

## 4️ Memory Retention Behavior (Cliff vs Decay)

<img width="866" height="920" alt="graph_2" src="https://github.com/user-attachments/assets/90799d52-5493-4fd3-9055-9c45286a66a1" />


Transformers show perfect retention followed by a sudden memory cliff.  
BDH avoids catastrophic collapse but exhibits smooth exponential decay due to fixed-capacity recurrence.

---

## 5️ Harmful Prompt Entropy Analysis

<img width="739" height="515" alt="entropy_vs_top_k_mass" src="https://github.com/user-attachments/assets/ae9c4eae-99bb-4139-9689-e6d16b88f085" />

Transformer activations remain diffuse with high entropy.  
BDH shows entropy drop and localized sparse spikes, enabling representation-level safety monitoring.

---

##  Architectural Comparison

| Dimension | Transformer | BDH |
|------------|-------------|------|
| Activation Density | ~99% | 2.5–3% |
| Memory Complexity | O(N²) | O(1) |
| Representation | Polysemantic | Near-monosemantic |
| Inference Adaptation | Static | Hebbian |
| Safety Signal | Diffuse | Localized |
| Interpretability | Post-hoc | Structural |

---

##  How to Run Locally

```bash
git clone <your-repo-link>
cd <repo-folder>
pip install -r requirements.txt
python activation_analysis.py
python hebbian_experiment.py
python memory_test.py
streamlit run app.py
```

---

##  Hosted Demo

👉 **[Insert Hosted Demo Link Here]**

Includes:
- Cross-lingual neuron overlap
- Sparsity comparison
- Hebbian strengthening demo
- Memory stress test
- Harmful prompt entropy analysis

---

##  Team Members & Contributions

| Team Member | Contribution |
|--------------|-------------|
| Palak Singhal | BDH Memory Retention |
| Sumedha | Memory Retention Transformer part |
| Idhaa | Text based Sparsity BDH |
| Vithika | Text based Sparsity Transformer part |
| Vaibhavi | Safety & harmful prompt analysis BDH |
| Ruthvika | Safety & harmful prompt analysis Transformer part |
| Nanditha | Image based Sparsity BDH |
| Akshara | Image based Sparsity Transformer part |
| Maithilee | Monosemanticity Analysis BDH |
| Sahaj | Monosemanticity Analysis Transformer part |
| Sanjana | Synapse Strengthening and Inference Learning BDH |
| Rishitha | Synapse Strengthening and Inference Learning Transformer part |

---

##  Limitations

- Small-scale models (<10M parameters)
- Exact password recall failed under extreme high-entropy noise
- Fixed-capacity recurrence causes exponential signal attenuation
- Transformer baseline used a distilled model
- Cross-lingual evaluation depends on translation accuracy

---

##  Future Scope

- Scale BDH to 100M+ parameters
- Integrate retrieval-based or external memory
- Hardware-optimized sparse computation
- Large-scale multilingual probing
- Real-time safety monitoring dashboards

---

##  Conclusion

Structured sparsity, localized state, and Hebbian reinforcement demonstrate a viable path toward post-Transformer architectures that learn, remember, and explain.

## https://drive.google.com/file/d/1G7VxQ48DH5DuuijWRSYAyLG1L92iQiRo/view?usp=drivesdk

## https://youtu.be/Z51rWGcgpCA
