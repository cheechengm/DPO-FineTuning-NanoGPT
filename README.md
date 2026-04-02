# Project Title: Direct Preference Optimization (DPO) for Mathematical Reasoning in NanoGPT

Overview
Developed and implemented a reinforcement learning pipeline using Direct Preference Optimization (DPO) to align a character-level NanoGPT model for solving arithmetic and algebraic problems. By leveraging a dataset of 100,000 positive/negative response pairs, the model was trained to move beyond simple next-token prediction toward providing structured, reasoned mathematical answers.

Key Technical Contributions

**Model Alignment**: Fine-tuned a pretrained NanoGPT model using DPO to prioritize high-quality, reasoned responses over "I don't know" or incorrect negative samples.

**Custom Training Pipeline:** Implemented a stable DPO loss function utilizing a frozen reference model to calculate log-probability deltas, ensuring controlled policy updates.

**Optimized Inference:** Achieved an 80% accuracy rate across addition, multiplication, and division by implementing near-greedy decoding (Temperature = 0.2, Top-K = 2) to minimize stochastic errors in mathematical outputs.

**Tech Stack:** Python, PyTorch, Transformers, AdamW Optimizer, NVIDIA RTX 3060 GPU acceleration.
