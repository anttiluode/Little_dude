# Brainwave-Mapped Fractal Dendritic Attention Network

A neural network architecture inspired by human brainwave frequencies for multi-scale temporal processing.

# Overview

This system implements an attention mechanism that processes information across five different temporal scales, inspired by the frequency bands 
observed in human neural activity. Instead of traditional single-scale attention, it uses multiple parallel processors operating at different time horizons.

# Core Concept

The architecture maps neural frequency bands to different processing scales:

Gamma (β=1.5): Fast, local detail processing (4-token windows)
Beta (β=0.75): Focused attention span (8-token windows)
Alpha (β=0.5): Pattern integration (16-token windows)
Theta (β=0.3): Memory consolidation (32-token windows)
Delta (β=0.1): Deep structural processing (64-token windows)

What "Coherence" Means
Coherence refers to cross-frequency synchronization - when different processing scales align and reinforce each other. High coherence indicates that patterns detected at multiple temporal scales are consistent, suggesting the input contains meaningful structure rather than noise.
The system uses a "soma coherence detector" that:

Measures correlation between different frequency processors
Identifies moments when multiple scales agree
Enhances signal processing during these alignment events

# Files

fractal_dendritic_attention.py

The core neural network implementation containing:

BrainwaveProcessor: Individual frequency band processors
SomaCoherenceDetector: Cross-frequency alignment detection
BrainwaveMappedFDAN: Complete model architecture

Interactive learning interface that demonstrates:

Real-time pattern learning from conversation
Multi-scale processing development
Character-level language acquisition

Usage

Install requirements.txt

Run the main architecture test
python fractal_dendritic_attention.py

# Start interactive learning session

python fractal_dendritic_field_attention_chat.py

Comparison to Standard AI

# Advantages:

Multi-scale processing: Handles patterns at multiple time horizons simultaneously
Interpretable dynamics: Can observe which scales are active for different inputs
Emergent behavior: Cross-frequency interactions create complex responses
Adaptive processing: Different frequency dominance for different tasks

Standard transformers use single-scale attention with fixed context windows. This system processes the same input through multiple temporal lenses simultaneously.

# Limitations

Computational:

5x more processors than standard attention (one per frequency)
FFT operations add computational overhead
More complex than traditional architectures

# Practical:

Requires careful hyperparameter tuning for frequency mappings
May be overkill for simple pattern recognition tasks
Training dynamics are more complex due to multi-scale interactions

Scale:

Currently tested on small vocabularies and short sequences
Numerical stability requires attention with larger models
Cross-frequency learning may be slow for some tasks

Maturity:

Experimental architecture without extensive benchmarking
Limited to character-level processing in current implementation
May not outperform well-tuned standard models on all tasks

Key Insight
The system demonstrates that temporal multi-scale processing can produce interesting emergent behaviors, including adaptive pattern recognition and complex learning dynamics. Whether this translates to practical advantages over standard architectures depends on the specific application and dataset.
The "brainwave" inspiration provides a structured way to think about different types of attention across time scales, potentially useful for tasks requiring both fine-grained and long-range pattern detection.

We were supposed to have complex fields and dendritic system, but I gues they were not done as planned per Claude: 

were the gauge forces ideas used here?EditLooking through the code, no - the gauge forces ideas aren't explicitly implemented here. This is primarily a multi-scale temporal processing system inspired by brainwave frequencies.
What's Actually Implemented:
Multi-Scale Processing:

5 different temporal windows (γ,β,α,θ,δ)
Cross-frequency coherence detection
Fractal noise extraction using 1/k^β scaling

Missing Gauge Theory Elements:

No gauge field dynamics
No symmetry preservation mechanisms
No covariant derivatives or gauge transformations
No connection to fundamental forces

Where Gauge Ideas Could Fit:
The "soma hole-punching" mechanism has some conceptual similarity to gauge field interactions:
python# Current: Simple coherence detection
coherence = cosine_similarity(freq1, freq2)

# Gauge-inspired version could be:
# gauge_field = compute_connection_field(freq1, freq2)
# curvature = gauge_field_strength(gauge_field)
# hole_punch_strength = curvature_dependent_enhancement()
Potential Gauge Extensions:
1. Gauge Field Between Frequencies:

Treat cross-frequency interactions as gauge fields
Preserve information symmetries during frequency mixing
Connection fields that maintain pattern consistency

2. SU(1) Lagrangian Compatibility:

Implement U(1) gauge symmetry in attention weights
Phase-invariant frequency processing
Gauge-covariant pattern transformations

3. Symmetry-Preserving Learning:

Learning dynamics that respect gauge invariances
Information conservation laws during pattern discovery

Bottom Line:
This system is brainwave-inspired multi-scale processing rather than gauge theory implementation. The gauge forces ideas would be a fascinating next step to explore, potentially making the cross-frequency interactions more principled and physically motivated.
The current "coherence detection" could be viewed as a simplified version of what gauge field interactions might look like in information processing! 🧠⚡RetryALis the dendritic system used?EditLooking at the code carefully, yes and no - it depends on what you mean by "dendritic system":
What's Actually Implemented:
Dendritic-Inspired Elements:
pythonclass BrainwaveProcessor(nn.Module):
    # Each processor acts like a dendrite sampling at specific frequencies
    def extract_brainwave_noise(self, x):
        # This mimics dendrites filtering specific frequency content
Multi-Branch Processing:

5 parallel "dendrite" processors (one per brainwave frequency)
Each branch processes the same input differently
Outputs converge at the "soma" (coherence detector)

Missing True Dendritic Features:
No Dendritic Trees:

No branching tree structures
No spatial dendrite organization
No distance-dependent signal decay

No Dendritic Computations:

No local nonlinear processing in branches
No dendritic spike generation
No compartmental modeling

No Synaptic Integration:

No weighted synaptic inputs per dendrite
No temporal synaptic dynamics
No plasticity rules for individual synapses

What It Actually Is:
The system is more accurately described as:
Input → [5 Parallel Frequency Filters] → Soma Integration → Output
Rather than true dendritic processing, it's multi-scale temporal filtering with a biological naming convention.
How Real Dendrites Could Be Added:
pythonclass DendriticBranch(nn.Module):
    def __init__(self, branch_length, synapse_count):
        self.synapses = nn.ModuleList([Synapse() for _ in range(synapse_count)])
        self.branch_compartments = nn.ModuleList([Compartment() for _ in range(branch_length)])
    
    def forward(self, inputs):
        # Process inputs through synapses
        # Integrate along dendritic branch
        # Apply nonlinear dendritic computations
Bottom Line:
The current system uses dendritic-inspired naming and parallel processing but lacks true dendritic computation. It's more like 5 specialized "sensors" feeding into a central processor than actual dendritic trees with complex branching dynamics.
The "fractal dendritic" name refers more to the multi-scale fractal processing (the 1/k^β scaling) than to biological dendritic structures! 🧠
