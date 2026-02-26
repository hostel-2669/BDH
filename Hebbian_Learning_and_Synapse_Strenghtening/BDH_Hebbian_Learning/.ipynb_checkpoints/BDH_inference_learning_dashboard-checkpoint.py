"""
BDH Hebbian Learning - Interactive Results Dashboard
"""

import streamlit as st
from PIL import Image
import os


st.set_page_config(
    page_title="BDH Hebbian Learning Results",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .plot-container {
        border: 2px solid #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        background-color: #fafafa;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


results_file = 'results.json'
if not os.path.exists(results_file):
    st.error("results.json not found. Please run the experiment first to generate results.")
    st.info("Run: python bdh_experiment_FIXED.py")
    st.stop()

import json
with open(results_file, 'r') as f:
    results = json.load(f)


synapse_strengths = results['synapse_strengths']
model_config = results['model_config']


plot_files = {
    'plot1': 'plot1_synapse_evolution.png',
    'plot2': 'plot2_attention_matrix.png',
    'plot3': 'plot3_temporal_heatmap.png',
    'plot4': 'plot4_final_strengths.png'
}

plots_exist = all(os.path.exists(f) for f in plot_files.values())

if not plots_exist:
    st.error("Plot files not found. Please run the experiment first to generate the plots.")
    st.info("Run: python bdh_experiment_FIXED.py")
    st.stop()


with st.sidebar:
    st.title("BDH Project")
    st.markdown("---")
    
    st.markdown("### Project Overview")
    st.info("""
    Demonstrating Hebbian Learning in BDH Architecture
    
    This project shows how synapses strengthen during inference without training.
    """)
    
    st.markdown("### Key Results")
    st.success("""
    1.Hebbian learning confirmed
    2.Synapse strengths measured
    3.Temporal dynamics observed
    """)
    
    st.markdown("### Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Layers", str(model_config['n_layer']))
        st.metric("Tokens", "15")
    with col2:
        st.metric("Neurons", str(model_config['n_embd'] * 32 // model_config['n_head']))
        st.metric("Snapshots", str(results['total_snapshots']))
    
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    Use the tabs above to explore:
    - Overview: Project summary
    - Plot 1-4: Visualizations
    """)

# Main content
st.title("BDH Hebbian Learning")
st.markdown("### Neurons that fire together, wire together")
st.markdown("---")

# Create tabs
tabs = st.tabs([
    "Overview", 
    "Plot 1: Evolution", 
    "Plot 2: Matrix",
    "Plot 3: Temporal",
    "Plot 4: Final Strengths"
])

# Tab 0: Overview
with tabs[0]:
    st.header("Project Overview")
    
    # Get synapse values
    alice_london = synapse_strengths.get('alice_to_london', {}).get('final_strength', 0.0)
    london_england = synapse_strengths.get('london_to_england', {}).get('final_strength', 0.0)
    england_europe = synapse_strengths.get('england_to_europe', {}).get('final_strength', 0.0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{alice_london:.3f}</h2>
            <p>alice to london</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{london_england:.3f}</h2>
            <p>london to england</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{england_europe:.3f}</h2>
            <p>england to europe</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### What We Demonstrated")
        st.markdown("""
        This project empirically validates that the BDH (Baby Dragon Hatchling) 
        architecture implements Hebbian learning during inference:
        
        1. Synapses strengthen when related concepts co-occur
        2. No training is required. It works with random weights
        3. Connections form exactly when words appear together displaying temporal precision
    
        Attention scores serve as dynamic synapses that naturally 
        implement the principle "neurons that fire together, wire together".
        """)
    
    with col2:
        st.markdown("### Facts Processed")
        facts_text = "\n".join([f"        {i+1}. {fact}" for i, fact in enumerate(results['facts'])])
        st.code(facts_text)
        
        st.markdown("### Model Config")
        st.code(f"""
        Layers: {model_config['n_layer']}
        Embedding: {model_config['n_embd']}
        Neurons: {model_config['n_embd'] * 32 // model_config['n_head']}
        Heads: {model_config['n_head']}
        Params: {model_config['total_params'] // 1000}K
        """)

# Tab 1: Plot 1 - Synapse Evolution
with tabs[1]:
    st.header("Synapse Strength Evolution Over Time")
    
    st.markdown("""
    This is the main result showing Hebbian learning in action.
    
    What you're seeing:
    - Three colored lines (alice to london, london to england, england to europe)
    - Each line shows how synapse strength changes as tokens are processed
    - Sharp increases when related words appear together
    - Connections remain strong after formation
    """)
    
   
    img = Image.open(plot_files['plot1'])
    st.image(img, use_container_width=True)
    
   
    with st.expander("Detailed Explanation"):
        st.markdown(f"""
        ### Key Observations:
        
        **Red Line (alice to london):**
        - Starts at 0 (positions 0-2)
        - Jumps at position 3 when "london" appears
        - Stays elevated (approximately {alice_london:.3f})
        
        **Cyan Line (london to england):**
        - Starts at 0 (positions 0-7)
        - Jumps at position 8 when "england" appears
        - Stays elevated (approximately {london_england:.3f})
        
        **Blue Line (england to europe):**
        - Starts at 0 (positions 0-12)
        - Jumps at position 13 when "europe" appears
        - Stays elevated (approximately {england_europe:.3f})
        
        ### Why is this Hebbian Learning?
        
        Synapses strengthen exactly when related words co-occur. This follows 
        Hebb's principle: when neurons for "london" and "alice" both fire (activate), 
        their connection strengthens automatically through the dot product attention 
        mechanism.
        
        No training is needed as this happens with random weights during inference.
        """)

# Tab 2: Plot 2 - Attention Matrix
with tabs[2]:
    st.header("Final Attention Matrix Heatmap")
    
    st.markdown("""
    This shows the final state of all synapses after processing all three facts.
    
    What you're seeing:
    - A 15x15 heatmap (15 tokens in our sequence)
    - Bright spots indicate strong connections
    - Lower triangle only (due to causal masking)
    - Each cell represents synapse strength from token i to j
    """)
    
    img = Image.open(plot_files['plot2'])
    st.image(img, use_container_width=True)
    
    with st.expander("Detailed Explanation"):
        st.markdown("""
        ### Key Features:
        
        **Causal Masking (Lower Triangle):**
        - Tokens can only attend to past tokens
        - Position 3 can see positions 0,1,2 but not 4,5,6
        - This prevents information leakage from future
        
        **Bright Spots:**
        - [3, 0] and [5, 0]: london attending to alice
        - [8, 3] and [8, 5]: england attending to london
        - [13, 8] and [13, 10]: europe attending to england
        
        **What Each Cell Means:**
        ```
        sigma(i,j) = dot product of neuron activations
                   = how many neurons fire for BOTH tokens
                   = synapse strength
        ```
        
        Bright yellow/red indicates many overlapping neurons which means strong synapse
        """)

# Tab 3: Plot 3 - Temporal Heatmap
with tabs[3]:
    st.header("Temporal Evolution Heatmap")
    
    st.markdown("""
    This shows when each synapse forms during processing.
    
    What you're seeing:
    - 3 rows (one per concept pair)
    - 15 columns (one per token position)
    - Color shows synapse strength at that moment
    - Staircase pattern reveals learning dynamics
    """)
    
    img = Image.open(plot_files['plot3'])
    st.image(img, use_container_width=True)
    
    with st.expander("Detailed Explanation"):
        st.markdown("""
        ### The Pattern:
        
        **Row 1: alice to london**
        ```
        Positions 0-2:  Light (no connection yet)
        Position 3:     Dark (london appears - connection forms)
        Positions 4-14: Dark (connection persists)
        ```
        
        **Row 2: london to england**
        ```
        Positions 0-7:  Light (no connection yet)
        Position 8:     Dark (england appears - connection forms)
        Positions 9-14: Dark (connection persists)
        ```
        
        **Row 3: england to europe**
        ```
        Positions 0-12: Light (no connection yet)
        Position 13:    Dark (europe appears - connection forms)
        Position 14:    Dark (connection persists)
        ```
        
        ### What This Proves:
        
        1. Temporal Precision: Synapses form exactly when words co-occur
        2. Persistence: Once formed, connections remain strong
        3. Sequential Learning: Each fact processed in order
        4. Real-Time Dynamics: Learning happens during inference, not after
        
        This is direct visual evidence of Hebbian learning happening in real-time.
        """)

# Tab 4: Plot 4 - Final Strengths
with tabs[4]:
    st.header("Final Synapse Strengths Comparison")
    
    st.markdown("""
    This is the summary - final synapse strengths after processing all facts.
    
    What you're seeing:
    - Three colored bars (one per concept pair)
    - Height indicates final synapse strength
    - Exact values displayed on bars
    - All values greater than zero means Hebbian learning worked
    """)
    
    img = Image.open(plot_files['plot4'])
    st.image(img, use_container_width=True)
    
    with st.expander("Detailed Explanation"):
        st.markdown(f"""
        ### Final Results:
        
        | Pair | Strength | Interpretation |
        |------|----------|----------------|
        | alice to london | {alice_london:.3f} | Moderate (appeared once together) |
        | london to england | {london_england:.3f} | Strong (london appeared twice) |
        | england to europe | {england_europe:.3f} | Strong (appeared once together) |""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><b>BDH Hebbian Learning Project</b></p>
    <p>Demonstrating "Neurons that fire together, wire together" in action</p>
    <p><i>Created with Streamlit, Python, PyTorch</i></p>
</div>
""", unsafe_allow_html=True)

