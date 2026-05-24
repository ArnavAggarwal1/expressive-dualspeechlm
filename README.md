# Expressive DualSpeechLM (Phase 1 & 2)

An end-to-end, modular implementation of a dual-stream Speech Language Model designed to explicitly decouple semantic content from paralinguistic expression (pitch, energy, and speech rhythm). This framework models expressive speech conditioning by framing speech generation as an autoregressive token prediction task.

## 🚀 Key Features & Architecture

### Phase 1: Paralinguistic Tokenizer & Feature Extraction
* **Acoustic Profiling (`features.py`):** Extracts continuous frame-level paralinguistic features ($F_0$ pitch contours, speech energy, and frame pacing).
* **Robust Vector Quantization (`quantizer.py`):** Maps continuous style signals into discrete tokens across a 256-size codebook. 
* **Collapse Prevention:** Implements a custom **Orthogonal Sub-Space Embedding Initialization** strategy and a **Forced Batch Random Restart** mechanism to prevent codebook vector collapse, sustaining a healthy codebook perplexity (~31.15).

### Phase 2: Dual-Stream Sequence Interleaving & AcousticGPT
* **Prefix Style Prompting (`model.py`):** Interleaves streams by prepending paralinguistic style tokens as an explicit condition prompt before semantic/acoustic content sequence matrices `[B, T2 + T1]`.
* **Causal Autoregressive Backbone:** Uses a decoder-only causal Transformer architecture (`ExpressiveAcousticGPT`) configured with strict lower-triangular causal attention masking to prevent future-token leakage.
* **Target Vocabulary Header:** Projects hidden states to a standard `1024` discrete acoustic unit vocabulary (compatible with EnCodec/DAC neural audio structures).

### Phase 3: Autoregressive Token Cross-Entropy Trainer
* **Next-Token Prediction (`train_gpt.py`):** Features a robust optimization loop using an explicit causal matrix shift (`logits[..., :-1, :]` vs. `labels[..., 1:]`).
* **Padding-Aware Loss:** Enforces `nn.CrossEntropyLoss(ignore_index=-100)` to isolate padded frame boundaries and ensure clean gradient updates.

---

## 🛠️ Project Structure
```text
expressive_dualspeechlm_phase1/
│
├── __init__.py
├── dataset.py          # 24kHz multi-stream padding and dataloading
├── features.py         # Frame-level F0/energy extraction pipelines
├── quantizer.py        # ProsodyVQQuantizer with cluster initialization
├── model.py            # Sequence interleaving and ExpressiveAcousticGPT 
├── train_vq.py         # Codebook verification and optimization loop
└── train_gpt.py        # Autoregressive cross-entropy training pipeline
