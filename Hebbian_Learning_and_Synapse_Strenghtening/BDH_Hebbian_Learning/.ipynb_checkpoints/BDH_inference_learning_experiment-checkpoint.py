
import sys
import math
import dataclasses
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# BDH.PY CODE

@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR

        # Current attention scores matrix represents σ(i,j)
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V, scores


class BDH_WithTracking(nn.Module):
    """
    Tracks synapses
    """
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_with_tracking(self, idx):
        """
        Forward pass that tracks synapse evolution
        """
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        
        synapse_history = []

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)  # B, 1, T, D

        for level in range(C.n_layer):
            x_latent = x @ self.encoder  
            x_sparse = F.relu(x_latent)  

            
            yKV, attention_scores = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
            )
            
            # Store synapse snapshot
            for t in range(T):
                for h in range(nh):
                    synapse_history.append({
                        'layer': level,
                        'head': h,
                        'timestep': t,
                        'token_idx': idx[0, t].item(),
                        'sigma': attention_scores[0, h, :t+1, :t+1].clone().detach(),  
                        'x_sparse': x_sparse[0, h, t, :].clone().detach(),  
                    })
            
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse 

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        
        return logits, synapse_history


# HEBBIAN LEARNING EXPERIMENT
class HebbianExperiment:
    """
    Experiment demonstrating Hebbian learning
    """
    def __init__(self):
        # Vocabulary
        self.vocab = {
            '<PAD>': 0, 'alice': 1, 'is': 2, 'in': 3, 'london': 4,
            'england': 5, 'europe': 6, '.': 7
        }
        self.idx2word = {v: k for k, v in self.vocab.items()}
        
        # Concept pairs to track
        self.concept_pairs = [
            ('alice', 'london'),
            ('london', 'england'),
            ('england', 'europe')
        ]
        
        # Create BDH model with tracking
        config = BDHConfig(
            n_layer=4,
            n_embd=64,
            dropout=0.0,  
            n_head=2,
            mlp_internal_dim_multiplier=32,
            vocab_size=len(self.vocab)
        )
        
        self.model = BDH_WithTracking(config)
        self.model.eval()
        
        print(f"\n{'='*70}")
        print("MODEL ARCHITECTURE:")
        print(f"{'='*70}")
        print(f"  Layers: {config.n_layer}")
        print(f"  Embedding dim: {config.n_embd}")
        print(f"  Heads: {config.n_head}")
        print(f"  Neuron dim (N): {config.n_embd * config.mlp_internal_dim_multiplier // config.n_head}")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*70}\n")
        
    def tokenize(self, sentence):
        tokens = sentence.lower().split()
        return [self.vocab.get(t, 0) for t in tokens]
    
    def run_experiment(self):
        facts = [
            "alice is in london .",
            "london is in england .",
            "england is in europe ."
        ]
        
        print("PROCESSING FACTS:")
        for i, fact in enumerate(facts, 1):
            print(f"  {i}. {fact}")
        print()
        
        # Tokenize all facts
        all_tokens = []
        fact_boundaries = [0]
        for fact in facts:
            tokens = self.tokenize(fact)
            all_tokens.extend(tokens)
            fact_boundaries.append(len(all_tokens))
        
        tokens_tensor = torch.tensor([all_tokens])
        
        print(f"Token sequence: {[self.idx2word[t] for t in all_tokens]}")
        print(f"Total tokens: {len(all_tokens)}\n")
        
        # Run model
        with torch.no_grad():
            logits, synapse_history = self.model.forward_with_tracking(tokens_tensor)
        
        return synapse_history, facts, all_tokens, fact_boundaries
    
    def analyze_synapses(self, synapse_history, all_tokens):
        """
        Analyze synapse strengthening between concept pairs
        The σ matrix from attention_scores represents synapses
        """
        # Map words to token positions
        token_positions = {}
        for pos, token_idx in enumerate(all_tokens):
            word = self.idx2word[token_idx]
            if word not in token_positions:
                token_positions[word] = []
            token_positions[word].append(pos)
        
        print("TOKEN POSITIONS:")
        for word in ['alice', 'london', 'england', 'europe']:
            if word in token_positions:
                print(f"  {word}: positions {token_positions[word]}")
        print()
        
        synapse_evolution = defaultdict(list)
        
        final_layer = self.model.config.n_layer - 1
        
        for t in range(len(all_tokens)):
            
            snapshot = None
            for s in synapse_history:
                if s['layer'] == final_layer and s['head'] == 0 and s['timestep'] == t:
                    snapshot = s
                    break
            
            if snapshot is None:
                continue
                
            sigma = snapshot['sigma']  # T x T matrix of attention scores
            
           
            for word1, word2 in self.concept_pairs:
                if word1 in token_positions and word2 in token_positions:
                    
                    
                    pos1_list = [p for p in token_positions[word1] if p <= t]
                    pos2_list = [p for p in token_positions[word2] if p <= t]
                    
                    # Calculate strength only if both words have appeared
                    if pos1_list and pos2_list:
                        strengths = []
                        for p1 in pos1_list:
                            for p2 in pos2_list:
                                # Only check if p2 > p1 (causal constraint)
                                if p2 > p1 and p2 < sigma.size(0) and p1 < sigma.size(1):
                                    strengths.append(sigma[p2, p1].item())
                        
                        if strengths:
                            N = self.model.config.n_embd * self.model.config.mlp_internal_dim_multiplier // self.model.config.n_head
                            avg_strength = np.mean(np.abs(strengths)) / np.sqrt(N)
                        else:
                            
                            avg_strength = 0.0
                    else:
                        
                        avg_strength = 0.0
                    
                    
                    synapse_evolution[(word1, word2)].append({
                        'timestep': t,
                        'strength': avg_strength
                    })
        
        return synapse_evolution, token_positions
    
    def visualize(self, synapse_evolution, synapse_history, facts, fact_boundaries):
        
        final_layer = self.model.config.n_layer - 1
        final_snapshots = [s for s in synapse_history 
                          if s['layer'] == final_layer and s['head'] == 0]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        print("Creating visualizations...")
        
        # ============================================================
        # PLOT 1: Synapse Strength Evolution
        # ============================================================
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        
        for idx, (pair, data) in enumerate(synapse_evolution.items()):
            if data:
                timesteps = [d['timestep'] for d in data]
                strengths = [d['strength'] for d in data]
                ax1.plot(timesteps, strengths, marker='o', label=f'{pair[0]} → {pair[1]}',
                        color=colors[idx], linewidth=3, markersize=10, alpha=0.8)
        
        for i, boundary in enumerate(fact_boundaries[1:-1], 1):
            ax1.axvline(x=boundary-0.5, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            ax1.text(boundary-2.5, ax1.get_ylim()[1]*0.95, f'Fact {i}→{i+1}',
                    fontsize=10, rotation=90, va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax1.set_xlabel('Token Position', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Synapse Strength |σ(i,j)|', fontsize=14, fontweight='bold')
        ax1.set_title('Hebbian Learning: Synapse Strengthening Over Time\n' + 
                     'Attention Scores as Synapses',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.legend(fontsize=13, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        fig1.savefig('plot1_synapse_evolution.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("Saved: plot1_synapse_evolution.png")
        
        # ============================================================
        # PLOT 2: Final Attention Matrix Heatmap
        # ============================================================
        if final_snapshots:
            final_sigma = final_snapshots[-1]['sigma'].numpy()
            fig2, ax2 = plt.subplots(figsize=(10, 9))
            
            sns.heatmap(final_sigma, cmap='RdYlBu_r', center=0, ax=ax2,
                       cbar_kws={'label': 'Attention Score σ(i,j)', 'shrink': 0.8},
                       annot=False, square=True, linewidths=0.5)
            
            ax2.set_title(f'Final Attention Matrix (Layer {final_layer}, Head 0)\n' +
                         'σ(i,j) = (Q⊙RoPE) @ (K⊙RoPE)ᵀ with Causal Masking',
                         fontsize=16, fontweight='bold', pad=20)
            ax2.set_xlabel('Token Position j', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Token Position i', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            fig2.savefig('plot2_attention_matrix.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print("Saved: plot2_attention_matrix.png")
        
        
        # ============================================================
        # PLOT 3: Temporal Evolution Heatmap
        # ============================================================
        if synapse_evolution:
            fig4, ax4 = plt.subplots(figsize=(14, 6))
            
            n_steps = max(len(data) for data in synapse_evolution.values())
            n_pairs = len(self.concept_pairs)
            strength_matrix = np.zeros((n_pairs, n_steps))
            
            for idx, pair in enumerate(self.concept_pairs):
                if pair in synapse_evolution:
                    data = synapse_evolution[pair]
                    for i, d in enumerate(data):
                        if i < n_steps:
                            strength_matrix[idx, i] = d['strength']
            
            sns.heatmap(strength_matrix, cmap='YlOrRd', ax=ax4,
                       yticklabels=[f'{p[0]} → {p[1]}' for p in self.concept_pairs],
                       cbar_kws={'label': 'Synapse Strength', 'shrink': 0.8},
                       annot=False)
            
            ax4.set_xlabel('Processing Step (Token Position)', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Concept Pair', fontsize=14, fontweight='bold')
            ax4.set_title('Temporal Evolution of Synapse Strengths\n' +
                         'How Connections Strengthen as Facts are Processed',
                         fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            fig4.savefig('plot3_temporal_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close(fig4)
            print("Saved: plot3_temporal_heatmap.png")
        
        # ============================================================
        # PLOT 4: Final Synapse Strengths Comparison
        # ============================================================
        if synapse_evolution:
            fig5, ax5 = plt.subplots(figsize=(10, 8))
            
            pairs = list(synapse_evolution.keys())
            final_strengths = [synapse_evolution[pair][-1]['strength'] 
                             if synapse_evolution[pair] else 0 
                             for pair in pairs]
            
            bars = ax5.bar(range(len(pairs)), final_strengths, 
                          color=colors[:len(pairs)], alpha=0.8, edgecolor='black', linewidth=2)
            
            ax5.set_xticks(range(len(pairs)))
            ax5.set_xticklabels([f'{p[0]}\n↓\n{p[1]}' for p in pairs], 
                               fontsize=13, fontweight='bold')
            ax5.set_ylabel('Final Synapse Strength |σ(i,j)|', fontsize=14, fontweight='bold')
            ax5.set_title('Final Attention Strengths Between Concept Pairs\n' +
                         'After Processing All Three Facts',
                         fontsize=16, fontweight='bold', pad=20)
            ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Add value labels on bars
            for bar, strength in zip(bars, final_strengths):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{strength:.5f}', ha='center', va='bottom', 
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            fig5.savefig('plot4_final_strengths.png', dpi=300, bbox_inches='tight')
            plt.close(fig5)
            print("Saved: plot4_final_strengths.png")
        
        print("\n" + "="*70)
        print("All visualizations saved successfully!")
        print("="*70)


def main():
    print("="*70)
    print("BDH HEBBIAN LEARNING EXPERIMENT")
    print("="*70)
   
    
    # Run experiment
    experiment = HebbianExperiment()
    
    print("Running experiment...")
    synapse_history, facts, all_tokens, fact_boundaries = experiment.run_experiment()
    print(f"Collected {len(synapse_history)} synapse snapshots\n")
    
    print("Analyzing synapse evolution...")
    synapse_evolution, token_positions = experiment.analyze_synapses(synapse_history, all_tokens)
    
    if synapse_evolution:
        print("\nFINAL SYNAPSE STRENGTHS:")
        print("-" * 70)
        for (w1, w2), data in synapse_evolution.items():
            if data:
                print(f"  |σ({w1:8} → {w2:8})| = {data[-1]['strength']:8.6f}")
        print("-" * 70)
    
    print("\nGenerating visualizations...")
    experiment.visualize(synapse_evolution, synapse_history, facts, fact_boundaries)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. plot1_synapse_evolution.png  - Strength over time")
    print("  2. plot2_attention_matrix.png   - Final σ(i,j) heatmap")
    print("  3. plot3_temporal_heatmap.png    - Evolution across pairs")
    print("  4. plot4_final_strengths.png    - Final strength comparison")
    print("=" * 70)
    
    print("\nSaving results to JSON...")
    import json
    import os
    
    try:
        results = {
            'synapse_strengths': {},
            'model_config': {
                'n_layer': experiment.model.config.n_layer,
                'n_embd': experiment.model.config.n_embd,
                'n_head': experiment.model.config.n_head,
                'vocab_size': experiment.model.config.vocab_size,
                'total_params': sum(p.numel() for p in experiment.model.parameters())
            },
            'facts': facts,
            'total_snapshots': len(synapse_history)
        }
        
        for (w1, w2), data in synapse_evolution.items():
            if data:
                strength_value = data[-1]['strength']
                if hasattr(strength_value, 'item'):
                    strength_value = strength_value.item()
                
                results['synapse_strengths'][f'{w1}_to_{w2}'] = {
                    'final_strength': float(strength_value),
                    'pair': [w1, w2]
                }
        
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        if os.path.exists('results.json'):
            abs_path = os.path.abspath('results.json')
            file_size = os.path.getsize('results.json')
            print(f"✓ Successfully saved results.json")
            print(f"  Location: {abs_path}")
            print(f"  Size: {file_size} bytes")
        else:
            print("✗ ERROR: results.json was not created")
            
    except Exception as e:
        print(f"✗ ERROR saving results.json: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

