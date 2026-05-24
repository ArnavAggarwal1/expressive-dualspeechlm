from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torchaudio

from .config import ProsodyFeatureConfig


def _frame_signal(x: torch.Tensor, frame_length: int, frame_hop: int) -> torch.Tensor:
    """
    Convert waveform to framed tensor using as_strided (no copy).

    Args:
        x: [B, N]
        frame_length: int
        frame_hop: int

    Returns:
        frames: [B, T, frame_length]
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [B, N], got {tuple(x.shape)}")
    bsz, n = x.shape
    if n < frame_length:
        # Produce a single padded frame
        pad = frame_length - n
        x = F.pad(x, (0, pad))
        n = x.shape[1]

    t = 1 + (n - frame_length) // frame_hop
    if t <= 0:
        raise ValueError("Computed non-positive number of frames. Check frame params.")

    # as_strided frame view
    stride_b, stride_n = x.stride()
    frames = x.as_strided(
        (bsz, t, frame_length),
        (stride_b, frame_hop * stride_n, stride_n),
    )
    return frames


def _safe_rms_energy(frames: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    RMSE energy per frame.

    Args:
        frames: [B, T, L]

    Returns:
        energy_rmse: [B, T]
    """
    # energy = sqrt(mean(x^2))
    return torch.sqrt(torch.mean(frames.float().pow(2), dim=-1).clamp_min(eps))


def _zcr(frames: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Zero crossing rate per frame (proxy for voicedness).

    Args:
        frames: [B, T, L]

    Returns:
        zcr: [B, T] (normalized by frame length)
    """
    # sign changes
    x = frames.float()
    s = torch.sign(x)
    # treat zeros as previous sign by forcing zero to 1 (minimizes spurious crossings)
    s = torch.where(s == 0, torch.ones_like(s), s)
    crossings = (s[..., 1:] * s[..., :-1] < 0).float().sum(dim=-1)
    return crossings / (frames.shape[-1] - 1 + eps)


def _yin_f0(
    frames: torch.Tensor,
    sample_rate: int,
    f0_min_hz: float,
    f0_max_hz: float,
    threshold: float,
) -> torch.Tensor:
    """
    YIN-like fundamental frequency estimation (vectorized across batch/time).

    This is a simplified, production-friendly variant:
      - Compute autocorrelation-like difference function using FFT convolution
        is complex; instead we compute difference function for candidate lags directly.
      - For efficiency, we only consider lags in [tau_min, tau_max].

    Args:
        frames: [B, T, L]
        sample_rate: int
        f0_min_hz: float
        f0_max_hz: float
        threshold: float in (0,1)

    Returns:
        f0_hz: [B, T] with unvoiced set to 0
    """
    if frames.dim() != 3:
        raise ValueError(f"frames must be [B, T, L], got {tuple(frames.shape)}")

    b, t, l = frames.shape
    x = frames.float()

    # tau = lag in samples
    tau_min = int(sample_rate / f0_max_hz)
    tau_max = int(sample_rate / f0_min_hz)
    tau_min = max(1, tau_min)
    tau_max = min(l - 2, tau_max)
    if tau_max <= tau_min:
        raise ValueError("Invalid tau range; check f0_min_hz/f0_max_hz vs frame length.")

    # Prepare lag candidates
    taus = torch.arange(tau_min, tau_max + 1, device=x.device)
    m = taus.numel()

    # Compute difference function d(tau) = sum_{j=1..L-tau} (x[j]-x[j+tau])^2
    # We want: [B, T, m]
    # Build shifted views: x[..., 0:l-tau] and x[..., tau:l]
    # Use broadcasting by indexing for each tau in a loop over m could be heavy;
    # but m is bounded by f0 range. We'll still do a bounded loop with no Python
    # per-batch/time, only per tau (acceptable for production if params are sane).
    d = torch.zeros((b, t, m), device=x.device, dtype=torch.float32)

    for i, tau in enumerate(taus.tolist()):
        x1 = x[..., : l - tau]
        x2 = x[..., tau:]
        diff = (x1 - x2).pow(2).sum(dim=-1)
        d[..., i] = diff

    # Normalize difference function: d'(tau) = d(tau) / (1/mau * sum_{j=1..tau} d(j))
    cumsum = torch.cumsum(d, dim=-1)
    # Avoid div by zero
    denom = cumsum.clamp_min(1e-12)
    d_prime = d / (denom / torch.arange(1, m + 1, device=x.device, dtype=d.dtype))

    # Find first tau where d'(tau) < threshold
    # y is index over tau, unvoiced if never crosses.
    mask = d_prime < threshold  # [B, T, m]
    # For each [B,T], get first index where mask is True.
    # If none, argmax gives 0; need to detect none.
    first_idx = mask.float().argmax(dim=-1)  # [B, T]
    any_cross = mask.any(dim=-1)  # [B, T]

    # Convert tau value
    tau_est = taus[first_idx]  # [B,T]
    f0 = torch.zeros((b, t), device=x.device, dtype=torch.float32)
    f0[any_cross] = (sample_rate / tau_est[any_cross].float()).clamp_min(0.0)

    # Optional refinement could be added; keep conservative here.
    return f0


def _estimate_speech_rate(voiced_mask: torch.Tensor, window_frames: int) -> torch.Tensor:
    """
    Speech rate proxy: voiced density over a sliding window.

    Args:
        voiced_mask: [B, T] bool/float
        window_frames: int

    Returns:
        speech_rate: [B, T] float in [0,1]
    """
    x = voiced_mask.float().unsqueeze(1)  # [B, 1, T]
    kernel = torch.ones((1, 1, window_frames), device=x.device, dtype=x.dtype)
    # Use symmetric padding then force exact output length to match input T.
    # conv1d length can become T+1 depending on kernel/window parity.
    padding = window_frames // 2
    num = F.conv1d(x, kernel, padding=padding)

    # num: [B, 1, T_out] -> trim/pad to [B, 1, T]
    t_in = voiced_mask.shape[-1]
    if num.shape[-1] > t_in:
        num = num[..., :t_in]
    elif num.shape[-1] < t_in:
        num = F.pad(num, (0, t_in - num.shape[-1]))

    denom = float(window_frames)
    return (num / denom).squeeze(1).clamp(0.0, 1.0)


@torch.no_grad()
def extract_paralinguistic_features(
    waveform: torch.Tensor,
    sample_rate: int,
    config: ProsodyFeatureConfig = ProsodyFeatureConfig(),
) -> torch.Tensor:
    """
    Extract continuous paralinguistic features and normalize.

    Output tensor shape:
        [Batch, Frames, 3] where dims = [f0_hz, energy_rmse, speech_rate]

    Args:
        waveform: [N] or [B, N] tensor
        sample_rate: int of waveform
        config: ProsodyFeatureConfig

    Returns:
        features: [B, T, 3] float tensor
    """
    if waveform is None:
        raise ValueError("waveform is None")
    if not torch.is_tensor(waveform):
        raise TypeError("waveform must be a torch.Tensor")

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(f"waveform must be [N] or [B,N], got {tuple(waveform.shape)}")

    if waveform.numel() == 0 or waveform.shape[1] == 0:
        raise ValueError("Empty audio tensor")

    if sample_rate != config.sample_rate:
        # Use torchaudio resample if needed
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config.sample_rate)
        waveform = resampler(waveform)
        sample_rate = config.sample_rate

    # Detect near-silence (all zeros or tiny amplitude)
    peak = waveform.abs().max().item()
    if peak < 1e-6:
        raise ValueError("Audio appears silent/near-silent (peak amplitude too low).")

    device = waveform.device
    waveform = waveform.contiguous()

    frame_length = config.frame_length_samples
    frame_hop = config.frame_hop_samples

    frames = _frame_signal(waveform, frame_length=frame_length, frame_hop=frame_hop)  # [B,T,L]
    b, t, _ = frames.shape

    # Energy
    energy_rmse = _safe_rms_energy(frames, eps=config.unvoiced_energy_floor)  # [B,T]

    # ZCR proxy for voicedness
    zcr = _zcr(frames)  # [B,T]
    # Convert zcr + energy into voiced_mask heuristically:
    # - low zcr tends to be voiced
    # - high energy tends to be speech
    # Normalize zcr per utterance to robustly threshold.
    zcr_mean = zcr.mean(dim=-1, keepdim=True)
    zcr_std = zcr.std(dim=-1, keepdim=True).clamp_min(1e-6)
    zcr_norm = (zcr - zcr_mean) / zcr_std
    energy_norm = (energy_rmse - energy_rmse.mean(dim=-1, keepdim=True)) / energy_rmse.std(dim=-1, keepdim=True).clamp_min(1e-6)

    # Heuristic voicedness threshold:
    voiced_mask = (zcr_norm < 0.0) & (energy_norm > -0.5)  # [B,T]

    # Pitch via YIN on full frames, then mask unvoiced by voiced_mask
    yin_frame_len = config.yin_frame_length_samples
    # If yin length differs, resample frames by slicing/padding
    if yin_frame_len != frame_length:
        if yin_frame_len < frame_length:
            frames_for_yin = frames[..., :yin_frame_len]
        else:
            pad = yin_frame_len - frame_length
            frames_for_yin = F.pad(frames, (0, pad))
    else:
        frames_for_yin = frames

    f0_hz = _yin_f0(
        frames_for_yin,
        sample_rate=sample_rate,
        f0_min_hz=config.f0_min_hz,
        f0_max_hz=config.f0_max_hz,
        threshold=config.yin_threshold,
    )  # [B,T]
    f0_hz = torch.where(voiced_mask, f0_hz, torch.zeros_like(f0_hz))

    # Speech rate proxy over voiced density
    window_frames = max(3, int(round(config.speech_rate_window_s * config.sample_rate / frame_hop)))
    speech_rate = _estimate_speech_rate(voiced_mask, window_frames=window_frames)  # [B,T]

    # Normalize features:
    #   - f0: unvoiced are 0; normalize voiced part only
    #   - energy: log1p then z-score
    #   - speech_rate: keep as-is (already [0,1]) optionally z-score
    f0 = f0_hz
    voiced_f0 = f0[f0 > 0]
    if voiced_f0.numel() == 0:
        # utterance without voiced frames: return zeros except energy/speech_rate
        f0_norm = torch.zeros_like(f0)
    else:
        # per-utterance z-norm over voiced frames; avoid dividing by 0
        mean = voiced_f0.mean()
        std = voiced_f0.std().clamp_min(1e-6)
        f0_norm = torch.where(f0 > 0, (f0 - mean) / std, torch.zeros_like(f0))

    energy_log = torch.log1p(energy_rmse)
    energy_mean = energy_log.mean(dim=-1, keepdim=True)
    energy_std = energy_log.std(dim=-1, keepdim=True).clamp_min(1e-6)
    energy_norm = (energy_log - energy_mean) / energy_std

    # speech_rate already bounded; z-score per utterance for robustness
    sr_mean = speech_rate.mean(dim=-1, keepdim=True)
    sr_std = speech_rate.std(dim=-1, keepdim=True).clamp_min(1e-6)
    speech_rate_norm = (speech_rate - sr_mean) / sr_std

    feat = torch.stack([f0_norm, energy_norm, speech_rate_norm], dim=-1)  # [B,T,3]
    if feat.shape != (b, t, config.feature_dim):
        raise RuntimeError(f"Unexpected feature shape: got {tuple(feat.shape)}")

    return feat.to(dtype=torch.float32, device=device)
