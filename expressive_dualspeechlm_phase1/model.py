from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


def interleave_speech_sequences(
    semantic_tokens: torch.Tensor,
    paralinguistic_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Interleave tokens so the decoder is conditioned on paralinguistic (tone/style) tokens first.

    Args:
        semantic_tokens: [B, Seq_Len_Dim1] int64 token ids for "what was said"
        paralinguistic_tokens: [B, Seq_Len_Dim2] int64 token ids for "tone/style"

    Returns:
        combined_tokens: [B, Seq_Len_Dim1 + Seq_Len_Dim2] with:
            combined = concat([paralinguistic_tokens, semantic_tokens], dim=1)
        combined_attention_mask: [B, Seq_Len_Dim1 + Seq_Len_Dim2] where 1 indicates valid tokens.
            If inputs contain no padding (or use non-zero pad ids), pass your own attention_mask later.
            This helper simply assumes pad_id=0 is padding and builds a mask accordingly.
    """
    if semantic_tokens.dim() != 2 or paralinguistic_tokens.dim() != 2:
        raise ValueError(
            f"Expected semantic_tokens/paralinguistic_tokens to be [B, L]. "
            f"Got {tuple(semantic_tokens.shape)} and {tuple(paralinguistic_tokens.shape)}"
        )
    if semantic_tokens.shape[0] != paralinguistic_tokens.shape[0]:
        raise ValueError(
            "Batch mismatch: "
            f"semantic_tokens batch={semantic_tokens.shape[0]} vs paralinguistic_tokens batch={paralinguistic_tokens.shape[0]}"
        )

    combined_tokens = torch.cat([paralinguistic_tokens, semantic_tokens], dim=1)

    # Simple attention mask heuristic: pad_id == 0.
    # If your dataset uses a different pad id, construct attention_mask externally and pass it to model.
    combined_attention_mask = (combined_tokens != 0).to(dtype=torch.long)

    return combined_tokens, combined_attention_mask


@dataclass(frozen=True)
class ExpressiveAcousticGPTConfig:
    vocab_size: int
    hidden_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 2048


class ExpressiveAcousticGPT(nn.Module):
    """
    Lightweight decoder-only causal Transformer for acoustic codec token prediction.

    Input:
        token_ids: [B, L] int64 token ids
        attention_mask (optional): [B, L] where 1 indicates valid tokens, 0 indicates padding.

    Output:
        logits: [B, L, vocab_size] predicting the next token id for each position.

    Attention:
        Uses a causal look-ahead mask so token i can attend only to tokens <= i.
        - Causal mask shape: [L, L]
          where positions (i, j) with j > i are masked out (set to -inf).
        - Padding mask shape (key_padding_mask): [B, L] with True for padded keys.
    """

    def __init__(self, cfg: ExpressiveAcousticGPTConfig) -> None:
        super().__init__()
        if cfg.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if cfg.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")

        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Decoder-only causal behavior is achieved via causal self-attention mask.
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def _causal_look_ahead_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build causal mask for TransformerEncoderLayer/Encoder:
          - Shape: [L, L]
          - dtype: float
          - Masked positions are set to -inf so softmax becomes ~0.
        """
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        # upper triangular (excluding diagonal) are future positions -> mask
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask_cond = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        mask = torch.where(mask_cond, mask, torch.zeros_like(mask))
        return mask

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] int64
            attention_mask: optional [B, L] long/bool with 1 for valid tokens and 0 for pad.

        Returns:
            logits: [B, L, vocab_size]
        """
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be [B, L], got {tuple(token_ids.shape)}")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"token_ids must be integer dtype, got {token_ids.dtype}")

        b, l = token_ids.shape
        if l > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length L={l} exceeds max_seq_len={self.cfg.max_seq_len}")

        device = token_ids.device
        x = self.token_emb(token_ids)  # [B, L, H]

        pos_ids = torch.arange(l, device=device, dtype=torch.long).unsqueeze(0).expand(b, l)  # [B, L]
        x = x + self.pos_emb(pos_ids)  # [B, L, H]

        # Causal mask: [L, L]
        causal_mask = self._causal_look_ahead_mask(seq_len=l, device=device, dtype=x.dtype)

        # Padding mask for keys: [B, L] with True for padded positions.
        key_padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.shape != (b, l):
                raise ValueError(f"attention_mask must be [B,L]={b,l}, got {tuple(attention_mask.shape)}")
            # attention_mask: 1 valid, 0 pad
            key_padding_mask = attention_mask.to(dtype=torch.long) == 0  # True => pad

        h = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )  # [B, L, H]

        logits = self.lm_head(h)  # [B, L, vocab_size]
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mock sanity test:
    # Semantic tokens: [B, T1], Paralinguistic tokens: [B, T2]
    B = 2
    T1 = 12
    T2 = 6
    vocab_size = 256

    semantic = torch.randint(0, vocab_size, (B, T1), device=device, dtype=torch.long)
    paralinguistic = torch.randint(0, vocab_size, (B, T2), device=device, dtype=torch.long)

    combined, attn_mask = interleave_speech_sequences(semantic, paralinguistic)  # combined: [B, T2+T1]
    cfg = ExpressiveAcousticGPTConfig(
        vocab_size=vocab_size,
        hidden_dim=256,
        n_layers=2,
        n_heads=8,
        dropout=0.0,
        max_seq_len=512,
    )
    model = ExpressiveAcousticGPT(cfg).to(device=device)

    logits = model(combined, attention_mask=attn_mask)
    assert logits.shape == (B, T1 + T2, vocab_size), f"Bad logits shape: {tuple(logits.shape)}"

    print("ExpressiveAcousticGPT mock test passed.")
    print("combined tokens:", tuple(combined.shape))
    print("logits:", tuple(logits.shape))
