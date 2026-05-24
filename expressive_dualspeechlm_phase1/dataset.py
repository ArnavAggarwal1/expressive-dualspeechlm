from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset
import torchaudio

from expressive_dualspeechlm_phase1.config import ProsodyFeatureConfig
from .features import extract_paralinguistic_features
from .quantizer import ProsodyVQQuantizer


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".wma", ".aac", ".webm"}


def _list_audio_files(path: str) -> List[str]:
    files: List[str] = []
    for root, _, fnames in os.walk(path):
        for f in fnames:
            ext = os.path.splitext(f)[1].lower()
            if ext in AUDIO_EXTS:
                files.append(os.path.join(root, f))
    files.sort()
    return files


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert [C, N] -> [1, N] by averaging channels."""
    if waveform.dim() != 2:
        raise ValueError(f"Expected waveform [C, N], got {tuple(waveform.shape)}")
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


@dataclass
class DatasetConfig:
    feature_config: ProsodyFeatureConfig
    return_waveform: bool = False


class ExpressiveSpeechDataset(Dataset):
    """Dataset that loads audio, resamples to config.sample_rate, extracts prosody features,
    and quantizes them to discrete embeddings.

    Returns items as dictionaries. Time dimension is frames (T) produced by feature extraction.
    """

    def __init__(
        self,
        audio_paths: Union[str, Sequence[str]],
        feature_config: Optional[ProsodyFeatureConfig] = None,
        *,
        return_waveform: bool = False,
        device: Optional[torch.device] = None,
        quantizer: Optional[ProsodyVQQuantizer] = None,
    ) -> None:
        super().__init__()

        if feature_config is None:
            feature_config = ProsodyFeatureConfig()

        if isinstance(audio_paths, str):
            if os.path.isdir(audio_paths):
                audio_paths_list = _list_audio_files(audio_paths)
            else:
                audio_paths_list = [audio_paths]
        else:
            audio_paths_list = list(audio_paths)

        if len(audio_paths_list) == 0:
            raise ValueError("No audio paths provided/found.")

        self.audio_paths: List[str] = audio_paths_list
        self.feature_config = feature_config
        self.return_waveform = return_waveform
        self.device = device

        # Quantizer is required to quantize features in __getitem__.
        # If not provided, we create a sensible default based on feature_config.
        if quantizer is not None:
            self.quantizer = quantizer
        else:
            # Lazy import to avoid circulars
            default_vq_cfg = None
            try:
                # If VQConfig is available in this module's quantizer implementation
                from .quantizer import VQConfig

                default_vq_cfg = VQConfig()
            except Exception:
                default_vq_cfg = None

            if default_vq_cfg is None:
                # Fall back to typical defaults matching ProsodyFeatureConfig.feature_dim=3
                from .quantizer import VQConfig
                default_vq_cfg = VQConfig()

            self.quantizer = ProsodyVQQuantizer(config=default_vq_cfg, embedding_dim_in=self.feature_config.feature_dim)


    def __len__(self) -> int:
        return len(self.audio_paths)

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.audio_paths[idx]

        # torchaudio.load may require torchcodec; provide a fallback for simple PCM WAV.
        try:
            waveform, sr = torchaudio.load(path)  # [C, N]
        except Exception:
            import wave
            import numpy as np

            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())

            if sampwidth != 2:
                raise RuntimeError(f"Unsupported WAV sample width: {sampwidth * 8} bits")

            x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            x = x.reshape(-1, n_channels).T  # [C, N]
            waveform = torch.from_numpy(x)

        waveform = _to_mono(waveform)

        if sr != self.feature_config.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.feature_config.sample_rate)
            waveform = resampler(waveform)
            sr = self.feature_config.sample_rate

        # features: [B=1, T, 3]
        feat = extract_paralinguistic_features(
            waveform.squeeze(0),
            sample_rate=sr,
            config=self.feature_config,
        )
        # normalize to [T, 3] for easier batching
        feat = feat.squeeze(0)

        # Quantize continuous features into codebook embeddings.
        # We expect embeddings: [T, 64] (or [B,T,64] depending on quantizer implementation).
        q = self.quantizer(feat.unsqueeze(0))  # try batch-first

        # Support multiple possible quantizer return types.
        # Expected (common) keys: token_ids or indices, quantized_embeddings or embeddings, and vq_loss.
        if isinstance(q, dict):
            token_ids = q.get("token_ids")
            if token_ids is None:
                token_ids = q.get("indices")
            embeddings = q.get("embeddings")
            if embeddings is None:
                embeddings = q.get("quantized_embeddings")
        else:
            # If quantizer returns a tuple-like, infer by position.
            # (tokens, embeddings, ...)
            token_ids = q[0]
            embeddings = q[1]

        if token_ids is None or embeddings is None:
            raise RuntimeError("Quantizer output missing token ids and/or embeddings.")

        # token_ids should end up [T]
        if token_ids.dim() == 2:
            token_ids = token_ids.squeeze(0)
        if token_ids.dim() != 1:
            raise RuntimeError(f"Unexpected token_ids shape: {tuple(token_ids.shape)}")

        # embeddings should end up [T, 64]
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        if embeddings.dim() != 2:
            raise RuntimeError(f"Unexpected embeddings shape: {tuple(embeddings.shape)}")

        item: Dict[str, Any] = {
            "file_path": path,
            "features_continuous": feat,  # [T, 3]
            "prosody_token_ids": token_ids.long(),  # [T]
            "quantized_embeddings": embeddings,  # [T, 64]
            "frame_count": torch.tensor(feat.shape[0], dtype=torch.long),
        }
        if self.return_waveform:
            item["waveform"] = waveform.squeeze(0)  # [N]
            item["sample_rate"] = torch.tensor(sr, dtype=torch.long)

        return item


def _pad_1d(tokens: torch.Tensor, max_len: int, pad_value: int = 0) -> torch.Tensor:
    # tokens: [T]
    t = tokens.shape[0]
    if t == max_len:
        return tokens
    out = torch.full((max_len,), pad_value, dtype=tokens.dtype, device=tokens.device)
    out[:t] = tokens
    return out


def _pad_2d(x: torch.Tensor, max_len: int, pad_value: float = 0.0) -> torch.Tensor:
    # x: [T, D]
    t, d = x.shape
    if t == max_len:
        return x
    out = torch.full((max_len, d), pad_value, dtype=x.dtype, device=x.device)
    out[:t] = x
    return out


def _pad_2d_mask(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: [B, T, D]
    # returns padded x and mask [B,T]
    raise NotImplementedError


def expressive_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length prosody sequences in a batch.

    Expects each item contains:
      - features_continuous: [T, 3]
      - prosody_token_ids: [T]
      - quantized_embeddings: [T, 64]

    Returns:
      - features_continuous: [B, Max_T, 3]
      - prosody_token_ids: [B, Max_T]
      - quantized_embeddings: [B, Max_T, 64]
      - attention_mask: [B, Max_T] (1 valid, 0 pad)
      - frame_count: [B]
      - file_path: list[str]

    Optionally includes waveform/sample_rate if present.
    """

    if len(batch) == 0:
        raise ValueError("Empty batch")

    device = batch[0]["features_continuous"].device

    lengths = torch.tensor([int(item["features_continuous"].shape[0]) for item in batch], dtype=torch.long)
    max_t = int(lengths.max().item())

    features_padded = []
    tokens_padded = []
    embeddings_padded = []
    attention_mask = torch.zeros((len(batch), max_t), dtype=torch.long, device=device)

    file_paths: List[str] = []

    waveform_list: List[torch.Tensor] = []
    sample_rates_list: List[torch.Tensor] = []
    return_waveform = "waveform" in batch[0]

    for i, item in enumerate(batch):
        t = int(item["features_continuous"].shape[0])
        attention_mask[i, :t] = 1

        features_padded.append(_pad_2d(item["features_continuous"], max_t, pad_value=0.0))  # [Max_T,3]
        tokens_padded.append(_pad_1d(item["prosody_token_ids"], max_t, pad_value=0))  # [Max_T]
        embeddings_padded.append(_pad_2d(item["quantized_embeddings"], max_t, pad_value=0.0))  # [Max_T,64]

        file_paths.append(str(item["file_path"]))

        if return_waveform:
            waveform_list.append(item["waveform"])  # variable, keep list; downstream can ignore
            sample_rates_list.append(item["sample_rate"])  # scalar tensor

    out: Dict[str, Any] = {
        "file_path": file_paths,
        "features_continuous": torch.stack(features_padded, dim=0),
        "prosody_token_ids": torch.stack(tokens_padded, dim=0),
        "quantized_embeddings": torch.stack(embeddings_padded, dim=0),
        "attention_mask": attention_mask,
        "frame_count": lengths,
    }

    if return_waveform:
        out["waveform"] = waveform_list
        out["sample_rate"] = torch.stack(sample_rates_list, dim=0)

    return out


if __name__ == "__main__":
    # Mock test: validate variable-length padding and mask correctness.
    # This block avoids file I/O by using a tiny dummy dataset.

    class _DummyDataset(Dataset):
        def __init__(self) -> None:
            self.items = [
                {"file_path": "a.wav", "features_continuous": torch.randn(5, 3), "prosody_token_ids": torch.randint(0, 10, (5,)), "quantized_embeddings": torch.randn(5, 64)},
                {"file_path": "b.wav", "features_continuous": torch.randn(8, 3), "prosody_token_ids": torch.randint(0, 10, (8,)), "quantized_embeddings": torch.randn(8, 64)},
                {"file_path": "c.wav", "features_continuous": torch.randn(3, 3), "prosody_token_ids": torch.randint(0, 10, (3,)), "quantized_embeddings": torch.randn(3, 64)},
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return self.items[idx]

    from torch.utils.data import DataLoader

    loader = DataLoader(_DummyDataset(), batch_size=3, shuffle=False, collate_fn=expressive_collate_fn)
    batch = next(iter(loader))

    print("features_continuous:", tuple(batch["features_continuous"].shape))  # [B,Max_T,3]
    print("prosody_token_ids:", tuple(batch["prosody_token_ids"].shape))  # [B,Max_T]
    print("quantized_embeddings:", tuple(batch["quantized_embeddings"].shape))  # [B,Max_T,64]
    print("attention_mask:", tuple(batch["attention_mask"].shape))  # [B,Max_T]
    print("attention_mask rows:", batch["attention_mask"].tolist())

    # Basic asserts
    B = 3
    max_t_expected = 8
    assert batch["features_continuous"].shape == (B, max_t_expected, 3)
    assert batch["prosody_token_ids"].shape == (B, max_t_expected)
    assert batch["quantized_embeddings"].shape == (B, max_t_expected, 64)
    assert batch["attention_mask"].shape == (B, max_t_expected)
    assert batch["attention_mask"][0, :5].sum().item() == 5
    assert batch["attention_mask"][0, 5:].sum().item() == 0
    assert batch["attention_mask"][1, :8].sum().item() == 8
    assert batch["attention_mask"][2, :3].sum().item() == 3
    print("Dataset padding test passed.")

