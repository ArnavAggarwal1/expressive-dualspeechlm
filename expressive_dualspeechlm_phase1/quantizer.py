from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VQConfig:
    """
    Vector Quantization settings.

    We implement a VQ-VAE style straight-through estimator with:
      - codebook loss: ||sg[z_e] - e||^2
      - commitment loss: beta * ||z_e - sg[e]||^2
    """

    vocab_size: int = 256
    embedding_dim: int = 64
    commitment_beta: float = 0.25


class ProsodyVQQuantizer(nn.Module):
    """
    Quantize continuous prosody features into discrete tone/prosody tokens.

    Input:
      - z_e: [B, T, embedding_dim_in] where embedding_dim_in can be != embedding_dim
             If != embedding_dim, an internal linear projection is applied.

    Output:
      - indices: [B, T] int64 codebook ids (Prosody/Tone tokens)
      - z_q: [B, T, embedding_dim] quantized embedding vectors
      - vq_loss: scalar tensor (codebook + commitment)
      - codebook_loss: scalar
      - commitment_loss: scalar

    Includes a deterministic sub-space initialization across the codebook and an
    explicit per-forward forced batch random restart for any codebook vector
    with zero usage in the current forward pass.
    """

    def __init__(self, config: VQConfig, embedding_dim_in: int) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if config.commitment_beta < 0:
            raise ValueError("commitment_beta must be >= 0")

        self.config = config
        self.embedding_dim_in = int(embedding_dim_in)
        self.embedding_dim = int(config.embedding_dim)

        # Optional projection so input feature dim matches codebook dim
        if self.embedding_dim_in != self.embedding_dim:
            self.proj = nn.Linear(self.embedding_dim_in, self.embedding_dim, bias=False)
        else:
            self.proj = nn.Identity()

        # Codebook: [K, D]
        self.embedding_dim = int(config.embedding_dim)
        self.vocab_size = int(config.vocab_size)

        # Keep backward-compat attribute name expected by train_vq.py
        # (train_vq.py uses quantizer.codebook.parameters()).
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.codebook = self.embedding

        # ---- Sub-Space Vector Initialization (requested) ----
        # Initialize across index ranges to force separated coordinate zones.
        with torch.no_grad():
            # Default random init
            nn.init.uniform_(self.embedding.weight, -1.0 / self.vocab_size, 1.0 / self.vocab_size)

            excited_center = 2.0
            sad_center = -2.0

            # Indices 0-63 excited cluster center
            self.embedding.weight.data[0:64].normal_(mean=excited_center, std=0.5)

            # Indices 64-127 sad cluster center
            self.embedding.weight.data[64:128].normal_(mean=sad_center, std=0.5)

            # Indices 128-191 fast-pace cluster center
            # Using high-frequency alternating values in embedding dimensions.
            center_block = self.embedding.weight.data[128:192]
            # Create alternating +/- pattern across embedding dim
            alt = torch.ones(self.embedding_dim, dtype=center_block.dtype, device=center_block.device)
            alt[::2] = 1.0
            alt[1::2] = -1.0
            fast_center_scale = 3.0
            center_block.copy_(alt.unsqueeze(0).repeat(center_block.shape[0], 1) * fast_center_scale)
            # Add small noise so it isn't perfectly deterministic
            center_block.add_(0.1 * torch.randn_like(center_block))
            # Leave remaining indices randomized (already initialized)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if z_e is None or not torch.is_tensor(z_e):
            raise TypeError("z_e must be a torch.Tensor")
        if z_e.dim() != 3:
            raise ValueError(f"z_e must be [B, T, C], got {tuple(z_e.shape)}")
        if z_e.numel() == 0:
            raise ValueError("z_e is empty")

        device = z_e.device
        b, t, c = z_e.shape
        z_e = z_e.contiguous()

        # Project to embedding_dim for distance computation
        z_e_proj = self.proj(z_e)  # [B, T, D]
        if z_e_proj.shape[-1] != self.embedding_dim:
            raise RuntimeError("Projection produced incorrect embedding dimension")

        # Compute distances to codebook entries:
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        # z: [B,T,D] -> [B*T, D]
        z_flat = z_e_proj.view(-1, self.embedding_dim)  # [B*T, D]
        e_weight = self.codebook.weight  # [K, D]

        z_sq = (z_flat ** 2).sum(dim=-1, keepdim=True)  # [B*T, 1]
        e_sq = (e_weight ** 2).sum(dim=-1).unsqueeze(0)  # [1, K]
        dist = z_sq + e_sq - 2.0 * (z_flat @ e_weight.t())  # [B*T, K]

        # Indices of nearest codebook vector
        indices = torch.argmin(dist, dim=-1)  # [B*T]

        # ---- Forced Batch Random Restart (requested) ----
        # For every codebook vector index with usage count == 0 in current forward pass,
        # replace its weight data by copying a random active frame and adding jitter.
        with torch.no_grad():
            flat_idx = indices.reshape(-1)
            if flat_idx.numel() > 0:
                counts = torch.bincount(flat_idx, minlength=self.vocab_size)  # [K], int64
                dead_mask = counts == 0  # [K] boolean
                dead_ids = torch.nonzero(dead_mask, as_tuple=False).squeeze(1)  # [n_dead]
                n_dead = int(dead_ids.numel())

                if n_dead > 0:
                    # Sample active frames (from z_flat) to replace dead vectors
                    active_count = int(flat_idx.numel())
                    rand_pos = torch.randint(0, active_count, (n_dead,), device=device)
                    sampled = z_flat[rand_pos]  # [n_dead, D]
                    jitter = 0.01 * torch.randn_like(sampled)
                    new_vecs = sampled + jitter
                    self.embedding.weight.data[dead_ids] = new_vecs

        indices_2d = indices.view(b, t)  # [B, T]

        # Gather quantized embeddings
        z_q = self.embedding(indices)  # [B*T, D]
        z_q = z_q.view(b, t, self.embedding_dim)  # [B, T, D]

        # Losses (VQ-VAE)
        # Codebook loss: ||sg[z_e] - z_q||^2
        # Commitment loss: beta * ||z_e - sg[z_q]||^2
        z_e_detached = z_e_proj.detach()
        z_q_detached = z_q.detach()

        codebook_loss = F.mse_loss(z_q, z_e_detached)
        commitment_loss = F.mse_loss(z_e_proj, z_q_detached) * self.config.commitment_beta
        vq_loss = codebook_loss + commitment_loss

        # Straight-through estimator
        z_q_st = z_e_proj + (z_q - z_e_proj).detach()

        # Ensure output device/dtype safety
        indices_2d = indices_2d.to(device=device, dtype=torch.long)
        z_q_st = z_q_st.to(device=device, dtype=z_e.dtype)

        return indices_2d, z_q_st, vq_loss, codebook_loss, commitment_loss
