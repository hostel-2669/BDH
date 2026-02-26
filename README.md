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
<img width="737" height="541" alt="transformer_entropy_vs_top_k_mass" src="https://github.com/user-attachments/assets/93467b03-ab07-43a7-864f-33543292f8a0" />


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
git clone https://github.com/hostel-2669/BDH.git
cd BDH
```
```bash
cd Hebbian_Learning_and_Synapse_Strengthening
python BDH_Hebbian_Learning/BDH_inference_learning_experiment.py
python transformer_model_hebbian_learning/BDH_vs_transformer.py
python transformer_model_hebbian_learning/Transformer_Synapse_training.py
python transformer_model_hebbian_learning/final_synapse_strengthening.py
```
```bash
cd ..
```
```bash
cd Image_Sparsity
python BDH_image_sparsity/build_atlas.py
python BDH_image_sparsity/cluster_neurons.py
python BDH_image_sparsity/generate_data.py
python BDH_image_sparsity/inspect_cluster.py
python BDH_image_sparsity/log_activations.py
python BDH_image_sparsity/model.py
python Transformer_image_sparsity/*.py
```
```bash
cd ..
```
```bash
cd Text_Based_Sparsity
python BDH_Text_based_Sparsity/*.py
```
```bash
cd ..
```
```bash
cd Harmful_prompts
python BDH_Harmful_Prompts/*.py
python Transformer_Model_Harmful_prompts/plot.py
```
```bash
cd ..
```
---

## Instructions for running .ipynb files
Install jupyter lab in terminal
or
open code in vscode editor
run each cell sequentially

##  Hosted Demo

 HOSTED DEMO LINK:-
https://docs.google.com/document/d/1kJej2FM3fV0vq4wI-2shdNxPhF_AD51m1XdHCLYynDw/edit?tab=t.0

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

## Drive link for Report
https://drive.google.com/file/d/1G7VxQ48DH5DuuijWRSYAyLG1L92iQiRo/view?usp=drivesdk

## Youtube Video link
https://youtu.be/Z51rWGcgpCA
