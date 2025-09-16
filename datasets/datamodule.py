# d/datamodule.py
from __future__ import annotations

from typing import Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .move_encoder import MoveEncoder


class ChessDataset(Dataset):
    def __init__(self, data_path: str, split: str = "train") -> None:
        super().__init__()
        self.data_path = data_path
        z = np.load(data_path, allow_pickle=True)
        if split not in z:
            raise ValueError(f"Split '{split}' not in {list(z.keys())}")
        self.data = z[split]  # array of dict (object dtype)
        self.encoder = MoveEncoder()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx].item()  # dict: obs, legal, act
        return item


def build_dataloaders(
    data_path: str,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = ChessDataset(data_path, split="train")
    val_ds = ChessDataset(data_path, split="val")

    enc = MoveEncoder()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=enc.collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=enc.collate,
        pin_memory=True,
    )
    return train_loader, val_loader
