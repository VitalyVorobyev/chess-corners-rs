"""Dataset helpers for ML refiner training."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def list_shards(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("shard_*.npz"))


def split_shards(
    shards: List[Path], val_split: float, split: str
) -> List[Path]:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    shards = sorted(shards)
    if val_split <= 0.0 or len(shards) == 0:
        return shards if split == "train" else []

    val_count = int(round(len(shards) * val_split))
    val_count = max(1, min(len(shards), val_count))

    if split == "val":
        return shards[-val_count:]
    return shards[: len(shards) - val_count]


@dataclass
class ShardData:
    patches: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    conf: np.ndarray
    is_pos: np.ndarray


class ShardCache:
    def __init__(self, max_items: int = 2) -> None:
        self.max_items = max(1, int(max_items))
        self._cache: Dict[int, ShardData] = {}
        self._order: List[int] = []

    def get(self, idx: int, loader) -> ShardData:
        if idx in self._cache:
            self._order.remove(idx)
            self._order.append(idx)
            return self._cache[idx]

        data = loader()
        self._cache[idx] = data
        self._order.append(idx)
        if len(self._order) > self.max_items:
            evict = self._order.pop(0)
            self._cache.pop(evict, None)
        return data


def _normalize_patches(patches: np.ndarray) -> np.ndarray:
    if patches.dtype == np.uint8:
        patches = patches.astype(np.float32) / 255.0
    else:
        patches = patches.astype(np.float32)
    return np.clip(patches, 0.0, 1.0)


class ShardDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        val_split: float,
        patch_size: int | None = None,
        max_cache: int = 2,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.val_split = float(val_split)
        self.patch_size = patch_size
        self._cache = ShardCache(max_items=max_cache)

        all_shards = list_shards(self.data_dir)
        if not all_shards:
            raise FileNotFoundError(f"No shards found in {self.data_dir}")

        self._shards = split_shards(all_shards, self.val_split, self.split)
        if not self._shards:
            raise FileNotFoundError(f"No shards found for split '{self.split}'")

        self._sizes: List[int] = []
        self._offsets: List[int] = [0]

        for shard in self._shards:
            with np.load(shard) as data:
                if "patches" not in data:
                    raise KeyError(f"Missing 'patches' in {shard}")
                count = int(data["patches"].shape[0])
                self._sizes.append(count)
                self._offsets.append(self._offsets[-1] + count)
                if self.patch_size is not None:
                    patch_shape = data["patches"].shape[1:]
                    expected = (self.patch_size, self.patch_size)
                    if patch_shape != expected:
                        raise ValueError(
                            f"Unexpected patch size {patch_shape} in {shard}"
                        )

    def __len__(self) -> int:
        return self._offsets[-1]

    def _load_shard(self, shard_idx: int) -> ShardData:
        path = self._shards[shard_idx]
        with np.load(path) as data:
            patches = _normalize_patches(data["patches"])
            dx = np.asarray(data["dx"], dtype=np.float32)
            dy = np.asarray(data["dy"], dtype=np.float32)
            conf = np.asarray(data["conf"], dtype=np.float32)
            conf = np.clip(conf, 0.0, 1.0)
            if "is_pos" in data:
                is_pos = np.asarray(data["is_pos"], dtype=np.float32)
            else:
                is_pos = (conf > 0.0).astype(np.float32)
        return ShardData(patches=patches, dx=dx, dy=dy, conf=conf, is_pos=is_pos)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError("index out of range")

        shard_idx = bisect.bisect_right(self._offsets, index) - 1
        local_idx = index - self._offsets[shard_idx]

        shard = self._cache.get(shard_idx, lambda: self._load_shard(shard_idx))

        patch = shard.patches[local_idx]
        x = torch.from_numpy(patch[None, ...])
        y = torch.tensor(
            [
                float(shard.dx[local_idx]),
                float(shard.dy[local_idx]),
                float(shard.conf[local_idx]),
                float(shard.is_pos[local_idx]),
            ],
            dtype=torch.float32,
        )
        return x, y
