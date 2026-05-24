from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class ProsodyFeatureConfig:
    """
    Configuration for paralinguistic prosody feature extraction.

    Features produced (float32):
      - f0_hz: Continuous F0 with unvoiced frames set to 0
      - energy_rmse: Frame-level energy (RMSE)
      - speech_rate: Proxy based on voiced-frame density within a sliding window
    """
    sample_rate: int = 24000
    frame_hop_ms: float = 10.0
    frame_length_ms: float = 30.0

    # Pitch tracker settings (YIN-like)
    yin_frame_ms: Optional[float] = None  # if None, uses frame_length_ms
    f0_min_hz: float = 50.0
    f0_max_hz: float = 450.0
    yin_threshold: float = 0.15

    # Energy / VAD proxy
    unvoiced_energy_floor: float = 1e-7
    speech_rate_window_s: float = 1.0  # window size to estimate voiced density

    dtype: Literal["float32", "float64"] = "float32"

    def __post_init__(self) -> None:
        if self.sample_rate not in (16000, 24000):
            raise ValueError(f"sample_rate must be 16000 or 24000, got {self.sample_rate}")
        if self.frame_hop_ms <= 0 or self.frame_length_ms <= 0:
            raise ValueError("frame_hop_ms and frame_length_ms must be > 0")
        if self.f0_min_hz <= 0 or self.f0_max_hz <= self.f0_min_hz:
            raise ValueError("Invalid f0 range")
        if not (0.0 < self.yin_threshold < 1.0):
            raise ValueError("yin_threshold must be in (0, 1)")
        if self.speech_rate_window_s <= 0:
            raise ValueError("speech_rate_window_s must be > 0")

    @property
    def frame_hop_samples(self) -> int:
        return int(round(self.frame_hop_ms * 1e-3 * self.sample_rate))

    @property
    def frame_length_samples(self) -> int:
        return int(round(self.frame_length_ms * 1e-3 * self.sample_rate))

    @property
    def yin_frame_length_samples(self) -> int:
        yin_ms = self.yin_frame_ms if self.yin_frame_ms is not None else self.frame_length_ms
        return int(round(yin_ms * 1e-3 * self.sample_rate))

    @property
    def feature_dim(self) -> int:
        # [f0_hz, energy_rmse, speech_rate]
        return 3
