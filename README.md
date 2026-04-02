# Project Title: Direct Preference Optimization (DPO) for Mathematical Reasoning in NanoGPT

Overview
This project focuses on fine-tuning a pretrained NanoGPT model to solve simple algebra and arithmetic problems using reinforcement learning through Direct Preference Optimization (DPO). A dataset of 100,000 paired responses consisting of negative and positive response is used to train preference alignment. The DPO approach enables the model to learn to generate accurate mathematical answers such as addition, subtraction, multiplication, division, and basic algebra with a short reasoning given .Such as 5+7=? The answer is 12 because 5+7 equals 12. x/4=3,x=? The answer is 12 because 3*4 equals 12.
Key Technical Contributions

**Model Alignment**: Fine-tuned a pretrained NanoGPT model using DPO to prioritize high-quality, reasoned responses over "I don't know" or incorrect negative samples.

**Custom Training Pipeline:** Implemented a stable DPO loss function utilizing a frozen reference model to calculate log-probability deltas, ensuring controlled policy updates.

**Optimized Inference:** Achieved an 80% accuracy rate across addition, multiplication, and division by implementing near-greedy decoding (Temperature = 0.2, Top-K = 2) to minimize stochastic errors in mathematical outputs.

**Tech Stack:** Python, PyTorch, Transformers, AdamW Optimizer, NVIDIA RTX 3060 GPU acceleration.

Below is an image of our Finetuned NanoGPT that is optimised to do math and can give simple reasoning of how it derived the calculation. It achieved an accuracy of 80%, with larger database and more optimised training set it can definitely be optimised to achieve a higher accuracy. For example our finetuned nanogpt is noted to be weak in division, with more training it definitely can be finetuned to be better at division
<img width="919" height="1219" alt="image" src="https://github.com/user-attachments/assets/b64885c6-38a7-4157-8b4f-926bc95731e9" />
