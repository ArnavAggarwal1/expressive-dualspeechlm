from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from expressive_dualspeechlm_phase1.config import ProsodyFeatureConfig
from expressive_dualspeechlm_phase1.dataset import ExpressiveSpeechDataset, expressive_collate_fn
from expressive_dualspeechlm_phase1.quantizer import ProsodyVQQuantizer, VQConfig


@dataclass
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 16
    epochs: int = 10
    num_workers: int = 0
    seed: int = 1234
    device_preference: str = "cuda"  # "cuda" or "cpu"

    # dummy run
    dummy_num_epochs: int = 5
    dummy_num_samples: int = 8
    dummy_audio_dir: str = "./_dummy_vq_audio"


def _select_device(cfg: TrainConfig) -> torch.device:
    if cfg.device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _codebook_perplexity_from_indices(indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Compute codebook perplexity from token indices.

    indices: [B, T] int64
    perplexity = exp(-sum(p * log(p)))
    """
    # Flatten, count frequencies
    flat = indices.reshape(-1)
    # guard empty
    if flat.numel() == 0:
        return torch.tensor(0.0, device=indices.device, dtype=torch.float32)

    counts = torch.bincount(flat, minlength=vocab_size).float()  # [K]
    probs = counts / counts.sum().clamp_min(1e-12)
    # avoid log(0)
    logp = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * logp).sum()
    return torch.exp(entropy)


def train_vq(
    train_paths: List[str],
    feature_config: ProsodyFeatureConfig,
    vq_config: VQConfig,
    cfg: TrainConfig,
) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = _select_device(cfg)
    print(f"Using device: {device}")

    dataset = ExpressiveSpeechDataset(
        audio_paths=train_paths,
        feature_config=feature_config,
        return_waveform=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=expressive_collate_fn,
        drop_last=False,
    )

    quantizer = ProsodyVQQuantizer(config=vq_config, embedding_dim_in=feature_config.feature_dim).to(device)
    # Only codebook parameters should be updated; projection/probably can be included.
    # Requirement: update only the VQ embedding codebook parameters.
    optimizer = torch.optim.AdamW(quantizer.codebook.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        quantizer.train()

        running: Dict[str, float] = {
            "vq_loss": 0.0,
            "codebook_loss": 0.0,
            "commitment_loss": 0.0,
        }
        n_batches = 0

        for batch in loader:
            feats = batch["features_continuous"].to(device=device, dtype=torch.float32)  # [B, T, 3]
            # attention mask is [B, T], but quantizer currently quantizes all frames.
            # We still compute loss normally; for stability, could mask padded frames.
            # To match the existing quantizer behavior, we keep it as-is.

            optimizer.zero_grad(set_to_none=True)

            # quantizer outputs: indices, z_q, vq_loss, codebook_loss, commitment_loss
            indices, z_q, vq_loss, codebook_loss, commitment_loss = quantizer(feats)

            # Ensure all scalar losses are finite
            if not torch.isfinite(vq_loss).all():
                raise RuntimeError(f"Non-finite vq_loss detected at epoch={epoch}")
            if not torch.isfinite(commitment_loss).all():
                raise RuntimeError(f"Non-finite commitment_loss detected at epoch={epoch}")

            vq_loss.backward()
            optimizer.step()

            running["vq_loss"] += float(vq_loss.detach().cpu())
            running["codebook_loss"] += float(codebook_loss.detach().cpu())
            running["commitment_loss"] += float(commitment_loss.detach().cpu())
            n_batches += 1

        avg_vq = running["vq_loss"] / max(1, n_batches)
        avg_cb = running["codebook_loss"] / max(1, n_batches)
        avg_com = running["commitment_loss"] / max(1, n_batches)

        # Evaluation-style codebook perplexity on the last epoch batches
        quantizer.eval()
        with torch.no_grad():
            all_perplexities: List[torch.Tensor] = []
            for batch in loader:
                feats = batch["features_continuous"].to(device=device, dtype=torch.float32)
                indices, _, _, _, _ = quantizer(feats)
                perplex = _codebook_perplexity_from_indices(indices, vocab_size=vq_config.vocab_size)
                all_perplexities.append(perplex)

            mean_perplex = torch.stack(all_perplexities).mean() if len(all_perplexities) else torch.tensor(0.0)

        warn = ""
        if float(mean_perplex.item()) < (0.1 * vq_config.vocab_size):
            warn = " [WARNING: potential codebook collapse (low perplexity)]"

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"vq_loss={avg_vq:.6f} codebook_loss={avg_cb:.6f} commitment_loss={avg_com:.6f} | "
            f"Codebook Perplexity={mean_perplex.item():.3f}{warn}"
        )


def _make_dummy_audio_files(out_dir: str, n_files: int, sample_rate: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    for i in range(n_files):
        # random duration between 0.3s and 0.9s
        dur = random.uniform(0.3, 0.9)
        n = int(sample_rate * dur)
        if n < sample_rate // 10:
            n = int(sample_rate * 0.1)

        t = torch.arange(n, dtype=torch.float32, device='cpu') / sample_rate

        # Dynamic prosody simulator:
        # - oscillating pitch contour between 80Hz and 300Hz
        # - varying amplitude (energy + pauses)
        f_lo, f_hi = 80.0, 300.0
        pitch_lfo_hz = random.uniform(0.4, 1.6)
        pitch_phase = random.uniform(0, 2 * math.pi)
        # pitch(t) in [f_lo, f_hi]
        pitch = f_lo + (f_hi - f_lo) * (0.5 + 0.5 * torch.sin(2 * math.pi * pitch_lfo_hz * t + pitch_phase))

        # integrate instantaneous frequency to get phase
        dt = 1.0 / sample_rate
        phase = 2 * math.pi * torch.cumsum(pitch * dt, dim=0)

        # amplitude envelope with occasional pauses
        # create a few random control points and interpolate
        n_ctrl = max(3, int(dur * 4))
        ctrl_t = torch.linspace(0, 1, steps=n_ctrl)
        ctrl_amp = torch.rand(n_ctrl) * 0.9 + 0.1  # [0.1, 1.0]

        # add pauses: multiply by near-zero segments
        if n_ctrl >= 3:
            for _ in range(random.randint(1, 2)):
                pause_center = random.uniform(0.15, 0.85)
                pause_width = random.uniform(0.05, 0.12)
                pause_mask = torch.exp(-0.5 * ((ctrl_t - pause_center) / pause_width) ** 2)
                ctrl_amp = ctrl_amp * (1.0 - 0.85 * pause_mask)

        # linear interpolation of ctrl_amp to per-sample envelope
        # torch.interp is not available; do linear interpolation manually.
        xt = (t / float(dur)).clamp(0.0, 1.0)  # [N] in [0,1]
        # ctrl_t is uniform in [0,1], so index math is straightforward.
        pos = xt * (n_ctrl - 1)
        i0 = torch.floor(pos).long().clamp(0, n_ctrl - 2)
        i1 = i0 + 1
        w = (pos - i0.float()).unsqueeze(0).squeeze(0)
        env = ctrl_amp[i0] * (1.0 - w) + ctrl_amp[i1] * w
        env = env.clamp(0.0, 1.0)


        # voiced carrier + breath/noise
        carrier = torch.sin(phase)
        noise = torch.randn_like(carrier) * 0.03
        waveform = (0.35 * env * carrier + noise).clamp(-1.0, 1.0)


        # torchaudio expects [channels, time]
        waveform = waveform.unsqueeze(0)

        path = os.path.join(out_dir, f"dummy_{i:02d}.wav")
        # torchaudio.save may require torchcodec; if unavailable, write a minimal PCM16 WAV ourselves.
        try:
            torchaudio.save(path, waveform, sample_rate)
        except Exception:
            import wave
            import numpy as np

            x = waveform.squeeze(0).clamp(-1.0, 1.0).cpu().numpy()
            pcm16 = (x * 32767.0).astype(np.int16)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # PCM16
                wf.setframerate(sample_rate)
                wf.writeframes(pcm16.tobytes())

        paths.append(path)


    return paths


def main() -> None:
    cfg = TrainConfig()

    # feature extractor config
    feature_cfg = ProsodyFeatureConfig(
        sample_rate=24000,
        frame_hop_ms=10.0,
        frame_length_ms=30.0,
    )

    # VQ config: keep modest sizes for the dummy run
    vq_cfg = VQConfig(
        vocab_size=128,
        embedding_dim=64,
        commitment_beta=0.25,
    )

    # Create dummy data so the script is end-to-end runnable.
    device = _select_device(cfg)
    print(f"Preparing dummy audio data under: {cfg.dummy_audio_dir} (device={device})")

    dummy_paths = _make_dummy_audio_files(
        out_dir=os.path.abspath(cfg.dummy_audio_dir),
        n_files=cfg.dummy_num_samples,
        sample_rate=feature_cfg.sample_rate,
    )

    # For the simulated run, use 5 mini-epochs.
    cfg_sim = TrainConfig(
        lr=cfg.lr,
        batch_size=min(cfg.batch_size, 8),
        epochs=cfg.dummy_num_epochs,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        device_preference=cfg.device_preference,
        dummy_num_epochs=cfg.dummy_num_epochs,
        dummy_num_samples=cfg.dummy_num_samples,
        dummy_audio_dir=cfg.dummy_audio_dir,
    )

    train_vq(
        train_paths=dummy_paths,
        feature_config=feature_cfg,
        vq_config=vq_cfg,
        cfg=cfg_sim,
    )


if __name__ == "__main__":
    main()

