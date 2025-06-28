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
