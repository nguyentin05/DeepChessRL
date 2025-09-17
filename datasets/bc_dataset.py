# datasets/bc_dataset.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

class BCDataset(Dataset):
    """
    Đọc nhiều shard .npz:
      - planes: uint8 [N,8,8,P]
      - mask  : bool  [N,A]
      - y_hard: int16 [N]
    Duy trì index map -> (shard_idx, local_idx)
    """
    def __init__(self, shard_paths: List[str], mmap: bool = True):
        self.shard_paths = shard_paths
        self.mmap = mmap
        self._shards = []
        self._sizes = []
        self._index: List[Tuple[int,int]] = []
        self._load_index()

    def _load_index(self):
        cum = 0
        for i, p in enumerate(self.shard_paths):
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
            shard = np.load(p, mmap_mode="r" if self.mmap else None)
            self._shards.append(shard)
            n = shard["y_hard"].shape[0]
            self._sizes.append(n)
            self._index.extend([(i, j) for j in range(n)])
            cum += n

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        si, li = self._index[idx]
        sh = self._shards[si]
        x  = sh["planes"][li]   # (8,8,P) uint8
        m  = sh["mask"][li]     # (A) bool
        y  = int(sh["y_hard"][li])
        # to torch tensors; model collate sẽ transpose sang (P,8,8)
        return {
            "x": torch.from_numpy(x.astype(np.float32) / 255.0),  # (8,8,P) float
            "mask": torch.from_numpy(m.astype(np.bool_)),         # (A) bool
            "y": torch.tensor(y, dtype=torch.long),
        }

def bc_collate(batch: List[Dict[str,Any]]) -> Dict[str, torch.Tensor]:
    xs = [b["x"] for b in batch]     # list of (8,8,P)
    masks = [b["mask"] for b in batch]
    ys = [b["y"] for b in batch]

    x = torch.stack(xs, dim=0)  # (B,8,8,P)
    x = x.permute(0, 3, 1, 2).contiguous()  # -> (B,P,8,8)
    mask = torch.stack(masks, dim=0)  # (B,A)
    y = torch.stack(ys, dim=0)
    return {"x": x, "mask": mask, "y": y}
