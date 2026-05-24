from __future__ import annotations

import torch

from expressive_dualspeechlm_phase1.config import ProsodyFeatureConfig
from expressive_dualspeechlm_phase1.features import extract_paralinguistic_features
from expressive_dualspeechlm_phase1.model import ExpressiveAcousticGPT, ExpressiveAcousticGPTConfig, interleave_speech_sequences
from expressive_dualspeechlm_phase1.quantizer import ProsodyVQQuantizer, VQConfig


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy waveform: 2.0s of random-ish noise with a weak sinusoid for voiced frames
    sr = 24000
    duration_s = 2.0
    n = int(sr * duration_s)

    t = torch.arange(n, device=device, dtype=torch.float32) / sr
    wav = 0.02 * torch.randn(n, device=device) + 0.1 * torch.sin(2.0 * torch.pi * 160.0 * t)
    wav = wav.clamp(-1.0, 1.0)

    feature_cfg = ProsodyFeatureConfig(sample_rate=sr)
    feats = extract_paralinguistic_features(wav, sample_rate=sr, config=feature_cfg)
    # feats: [B, Frames, 3]
    assert feats.dim() == 3, f"Expected [B,Frames,D], got {tuple(feats.shape)}"

    # Quantizer expects embedding_dim_in == feats feature dim
    # Target acoustic vocabulary size (Phase 3) = 1024
    vq_cfg = VQConfig(vocab_size=1024, embedding_dim=64, commitment_beta=0.25)
    quantizer = ProsodyVQQuantizer(config=vq_cfg, embedding_dim_in=feats.shape[-1]).to(device=device)

    indices, z_q, vq_loss, codebook_loss, commitment_loss = quantizer(feats)

    # -------- Phase 2 (AcousticGPT backbone) quick shape validation --------
    # Use indices as "semantic" tokens for this mock test; create a dummy "paralinguistic/style" token stream.
    # semantic: [B, T]
    semantic_tokens = indices
    # paralinguistic/style tokens: [B, T2] (choose shorter length to exercise concatenation)
    T2 = max(1, semantic_tokens.shape[1] // 2)
    paralinguistic_tokens = torch.randint(
        low=0,
        high=vq_cfg.vocab_size,
        size=(semantic_tokens.shape[0], T2),
        device=device,
        dtype=torch.long,
    )

    combined_tokens, attn_mask = interleave_speech_sequences(
        semantic_tokens=semantic_tokens,
        paralinguistic_tokens=paralinguistic_tokens,
    )  # combined_tokens: [B, T2+T]

    model_cfg = ExpressiveAcousticGPTConfig(
        vocab_size=vq_cfg.vocab_size,
        hidden_dim=256,
        n_layers=2,
        n_heads=8,
        dropout=0.0,
        max_seq_len=1024,
    )
    model = ExpressiveAcousticGPT(model_cfg).to(device=device)

    logits = model(combined_tokens, attention_mask=attn_mask)
    assert logits.shape == (
        semantic_tokens.shape[0],
        combined_tokens.shape[1],
        vq_cfg.vocab_size,
    ), f"Expected logits [B,L,V], got {tuple(logits.shape)}"

    print("=== Expressive DualSpeechLM Phase 1 Demo ===")
    print(f"Input waveform shape: {tuple(wav.shape)} @ {sr} Hz")
    print(f"Continuous prosody features shape: {tuple(feats.shape)}")
    print(f"Discrete prosody/token indices shape: {tuple(indices.shape)} (dtype={indices.dtype})")
    print(f"Quantized prosody embeddings shape: {tuple(z_q.shape)}")
    print(f"VQ loss: {vq_loss.item():.6f} (codebook={codebook_loss.item():.6f}, commitment={commitment_loss.item():.6f})")
    print(f"Phase 2 mock logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
