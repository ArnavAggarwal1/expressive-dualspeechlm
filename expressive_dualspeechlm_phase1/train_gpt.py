from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from expressive_dualspeechlm_phase1.dataset import ExpressiveSpeechDataset, expressive_collate_fn
from expressive_dualspeechlm_phase1.features import extract_paralinguistic_features
from expressive_dualspeechlm_phase1.model import (
    ExpressiveAcousticGPT,
    ExpressiveAcousticGPTConfig,
    interleave_speech_sequences,
)
from expressive_dualspeechlm_phase1.config import ProsodyFeatureConfig
from expressive_dualspeechlm_phase1.quantizer import ProsodyVQQuantizer, VQConfig


@dataclass(frozen=True)
class TrainGPTConfig:
    # Requested hyperparameters
    lr: float = 5e-5
    batch_size: int = 8
    epochs: int = 5
    weight_decay: float = 0.01
    # Simple linear scheduler over total_steps
    warmup_ratio: float = 0.0

    # Token conditioning settings
    pad_id: int = 0
    use_paralinguistic_prompt: bool = True

    # Mock run settings
    dummy_num_samples: int = 32
    dummy_min_frames: int = 50
    dummy_max_frames: int = 120


def _linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear schedule:
      - warmup: LR increases linearly from 0 -> base_lr
      - decay: LR decreases linearly from base_lr -> 0 across remaining steps
    """
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # decay
        decay_step = step - warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - float(decay_step) / float(decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _build_combined_tokens_for_batch(
    semantic_tokens: torch.Tensor,  # [B, T1]
    attention_mask: torch.Tensor,  # [B, T1] (1 valid, 0 pad)
    vocab_size: int,
    pad_id: int,
    *,
    use_paralinguistic_prompt: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build combined token sequence [B, T2+T1] and its attention mask.

    We follow Phase 2: paralinguistic tokens first, then semantic tokens.
    """
    if not use_paralinguistic_prompt:
        combined_tokens = semantic_tokens
        combined_attn_mask = attention_mask
        return combined_tokens, combined_attn_mask

    # Choose prompt length proportional to semantic length for stable demo
    b, t1 = semantic_tokens.shape
    # prompt length in [1, max_prompt_len]
    max_prompt_len = max(1, t1 // 2)
    t2 = max_prompt_len

    # Create prompt tokens; keep them valid (no padding) for simplicity.
    paralinguistic_tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(b, t2),
        device=device,
        dtype=torch.long,
    )

    # interleave_speech_sequences builds mask as (token != 0) heuristic.
    # Since prompt is random, we can also explicitly build prompt mask as ones.
    combined_tokens, _heuristic_mask = interleave_speech_sequences(
        semantic_tokens=semantic_tokens,
        paralinguistic_tokens=paralinguistic_tokens,
    )

    # Build a correct attention mask:
    # prompt portion: all valid (1), semantic portion: use provided attention_mask.
    prompt_mask = torch.ones((b, t2), device=device, dtype=torch.long)
    combined_attn_mask = torch.cat([prompt_mask, attention_mask.to(device=device, dtype=torch.long)], dim=1)

    # Ensure pad ids in semantic portion are consistent with attention_mask
    # (optional safety)
    combined_tokens[:, t2:] = torch.where(
        attention_mask.to(device=device, dtype=torch.bool),
        semantic_tokens,
        torch.full_like(semantic_tokens, pad_id),
    )

    return combined_tokens, combined_attn_mask


def train_gpt(
    *,
    cfg: TrainGPTConfig,
    device: torch.device,
    model: ExpressiveAcousticGPT,
    dataset: Dataset,
) -> None:
    model.train()

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=expressive_collate_fn,
    )

    total_steps = cfg.epochs * max(1, len(loader))
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _linear_warmup_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    # ignore_index=-100 for padding labels
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    vocab_size = model.cfg.vocab_size

    global_step = 0
    for epoch in range(cfg.epochs):
        running_loss = 0.0
        running_tokens = 0

        for batch in loader:
            # dataset.py collate returns:
            #   prosody_token_ids: [B, Max_T]
            #   attention_mask:    [B, Max_T] (1 valid, 0 pad)
            semantic_tokens: torch.Tensor = batch["prosody_token_ids"].to(device=device, dtype=torch.long)
            attention_mask: torch.Tensor = batch["attention_mask"].to(device=device, dtype=torch.long)

            # Combined/interleaved sequence: [B, L]
            combined_tokens, combined_attn_mask = _build_combined_tokens_for_batch(
                semantic_tokens=semantic_tokens,
                attention_mask=attention_mask,
                vocab_size=vocab_size,
                pad_id=cfg.pad_id,
                use_paralinguistic_prompt=cfg.use_paralinguistic_prompt,
                device=device,
            )

            # Forward
            logits: torch.Tensor = model(combined_tokens, attention_mask=combined_attn_mask)  # [B, L, V]

            # Classic autoregressive shift
            shift_logits = logits[..., :-1, :].contiguous()          # [B, L-1, V]
            shift_labels = combined_tokens[..., 1:].contiguous()    # [B, L-1]

            # Mask padding positions in labels to be ignored by CE
            # combined_attn_mask: [B, L] => for labels positions [1..L-1] => mask [B, L-1]
            label_valid_mask = combined_attn_mask[..., 1:].to(device=device, dtype=torch.long)  # [B, L-1]
            ignore_mask = label_valid_mask == 0

            shift_labels = shift_labels.clone()
            shift_labels[ignore_mask] = -100

            # Flatten
            b, l1, v = shift_logits.shape
            loss = criterion(
                shift_logits.view(b * l1, v),   # [B*(L-1), V]
                shift_labels.view(b * l1),      # [B*(L-1)]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Optional: gradient clipping for stability in early phases
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1

            # Metrics
            with torch.no_grad():
                running_loss += loss.item()
                # Count non-ignored labels for reporting (optional)
                running_tokens += int((shift_labels != -100).sum().item())

        avg_loss = running_loss / max(1, len(loader))
        perplexity = torch.exp(torch.tensor(avg_loss, device=device)).item()

        print(
            f"Epoch {epoch + 1:02d}/{cfg.epochs} | "
            f"loss={avg_loss:.6f} | token_perplexity={perplexity:.3f}"
        )


class _DummyExpressiveSpeechDataset(Dataset):
    """
    Dummy dataset that creates variable-length token sequences and attention masks
    in-memory (no audio / feature extraction / quantization).
    This is for a fully local mock run that still exercises the GPT training loop.

    We return keys compatible with expressive_collate_fn expectations:
      - features_continuous: [T, 3] (dummy)
      - prosody_token_ids:   [T]   (dummy)
      - quantized_embeddings: [T, 64] (dummy)
      - frame_count: optional
      - file_path: string
    """
    def __init__(
        self,
        *,
        num_samples: int,
        min_frames: int,
        max_frames: int,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.vocab_size = vocab_size
        self.device = device

        # token id 0 reserved as PAD (collate uses pad_value=0 for tokens and embeddings)
        self.pad_id = 0

        self._rng = torch.Generator(device="cpu").manual_seed(1234)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Variable length
        t = int(torch.randint(self.min_frames, self.max_frames + 1, (1,), generator=self._rng).item())
        # Tokens: 1..V-1 to avoid pad_id=0 accidentally marking valid tokens
        tokens = torch.randint(
            low=1,
            high=self.vocab_size,
            size=(t,),
            generator=self._rng,
            dtype=torch.long,
        )
        # Dummy continuous features [T,3] and dummy embeddings [T,64]
        feats = torch.randn(t, 3)
        embeddings = torch.randn(t, 64)

        return {
            "file_path": f"dummy_{idx}.wav",
            "features_continuous": feats,
            "prosody_token_ids": tokens,
            "quantized_embeddings": embeddings,
            "frame_count": torch.tensor(t, dtype=torch.long),
        }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model config (token vocab matches Phase-1 VQ vocab size)
    # Target acoustic vocabulary size (Phase 3) = 1024
    vq_cfg = VQConfig(vocab_size=1024, embedding_dim=64, commitment_beta=0.25)
    # Lightweight training GPT
    model_cfg = ExpressiveAcousticGPTConfig(
        vocab_size=vq_cfg.vocab_size,
        hidden_dim=256,
        n_layers=2,
        n_heads=8,
        dropout=0.1,
        max_seq_len=2048,
    )
    model = ExpressiveAcousticGPT(model_cfg).to(device=device)

    cfg = TrainGPTConfig(
        lr=5e-5,
        batch_size=8,
        epochs=5,
        weight_decay=0.01,
        use_paralinguistic_prompt=True,
    )

    # --- Full mock run using dummy samples so you can verify loss drops steadily locally ---
    dummy_dataset = _DummyExpressiveSpeechDataset(
        num_samples=cfg.dummy_num_samples,
        min_frames=cfg.dummy_min_frames,
        max_frames=cfg.dummy_max_frames,
        vocab_size=vq_cfg.vocab_size,
        device=device,
    )

    print(f"Starting ExpressiveAcousticGPT training on {device} for {cfg.epochs} epochs...")
    train_gpt(cfg=cfg, device=device, model=model, dataset=dummy_dataset)


if __name__ == "__main__":
    main()
