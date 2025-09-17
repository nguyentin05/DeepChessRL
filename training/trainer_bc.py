# training/trainer_bc.py
from __future__ import annotations
import os
import time
import math
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.bc_dataset import BCDataset, bc_collate
from models.bc_policy_net import BCPolicyNet
from config.config_bc_dagger import BCDaggerConfig

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,A), target: (B,), mask: (B,A) bool
    Set invalid logits to -inf before log-softmax, then NLL on targets.
    """
    large_neg = torch.finfo(logits.dtype).min / 2
    masked_logits = torch.where(mask, logits, torch.full_like(logits, large_neg))
    logp = F.log_softmax(masked_logits, dim=-1)
    nll = F.nll_loss(logp, target, reduction="mean")
    return nll

def topk_acc(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, ks=(1,3,5)) -> Dict[str,float]:
    res = {}
    with torch.no_grad():
        large_neg = torch.finfo(logits.dtype).min / 2
        masked_logits = torch.where(mask, logits, torch.full_like(logits, large_neg))
        for k in ks:
            topk = torch.topk(masked_logits, k=k, dim=-1).indices  # (B,k)
            correct = (topk == target.unsqueeze(-1)).any(dim=-1).float().mean().item()
            res[f"top{k}"] = correct
    return res

class BCTrainer:
    def __init__(self, cfg: BCDaggerConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        # Datasets
        train_paths = [os.path.join(cfg.shards_dir, p) for p in (cfg.train_shards or []) if p.endswith(".npz") or p.endswith(".npy")]
        val_paths   = [os.path.join(cfg.shards_dir, p) for p in (cfg.val_shards or []) if p.endswith(".npz") or p.endswith(".npy")]
        if not train_paths:
            # fallback: load any shard in folder as train
            for fn in sorted(os.listdir(cfg.shards_dir)):
                if fn.endswith(".npz"):
                    train_paths.append(os.path.join(cfg.shards_dir, fn))
        self.train_ds = BCDataset(train_paths, mmap=True)
        self.val_ds   = BCDataset(val_paths,   mmap=True) if val_paths else None

        self.train_loader = DataLoader(self.train_ds, batch_size=cfg.batch_size, shuffle=True,
                                       num_workers=2, pin_memory=True, collate_fn=bc_collate)
        self.val_loader   = DataLoader(self.val_ds, batch_size=cfg.batch_size, shuffle=False,
                                       num_workers=2, pin_memory=True, collate_fn=bc_collate) if self.val_ds else None

        # Model
        self.model = BCPolicyNet(cfg.in_planes, cfg.trunk_channels, cfg.residual_blocks, cfg.action_space_size).to(cfg.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device.startswith("cuda")))
        self.best_val = float("inf")
        self.global_step = 0

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        total = 0
        meter = {"top1":0.0, "top3":0.0, "top5":0.0}
        tic = time.time()
        for it, batch in enumerate(self.train_loader):
            x = batch["x"].to(self.cfg.device, non_blocking=True)      # (B,P,8,8)
            mask = batch["mask"].to(self.cfg.device, non_blocking=True)# (B,A)
            y = batch["y"].to(self.cfg.device, non_blocking=True)      # (B,)

            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.cfg.device.startswith("cuda")):
                logits = self.model(x, mask=None)  # mask only inside loss
                loss = masked_cross_entropy(logits, y, mask)

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.scaler.step(self.opt)
            self.scaler.update()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            self.global_step += 1

            if (it+1) % self.cfg.log_interval == 0:
                accs = topk_acc(logits, y, mask)
                for k,v in accs.items(): meter[k] = v
                print(f"[Train ep{epoch} it{it+1}] loss={loss.item():.4f} top1={meter['top1']:.3f} top3={meter['top3']:.3f} top5={meter['top5']:.3f}")

        return {
            "loss": total_loss / max(1,total),
            "time": time.time() - tic
        }

    def validate(self, epoch: int) -> Dict[str, Any]:
        if self.val_loader is None: return {"loss": None}
        self.model.eval()
        total_loss = 0.0
        total = 0
        top = {"top1":0.0, "top3":0.0, "top5":0.0}
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["x"].to(self.cfg.device, non_blocking=True)
                mask = batch["mask"].to(self.cfg.device, non_blocking=True)
                y = batch["y"].to(self.cfg.device, non_blocking=True)
                logits = self.model(x, mask=None)
                loss = masked_cross_entropy(logits, y, mask)
                total_loss += loss.item() * x.size(0)
                total += x.size(0)
                accs = topk_acc(logits, y, mask)
                for k in top.keys(): top[k] += accs[k] * x.size(0)

        avg_loss = total_loss / max(1,total)
        for k in top.keys(): top[k] = top[k] / max(1,total)
        print(f"[Val   ep{epoch}] loss={avg_loss:.4f} top1={top['top1']:.3f} top3={top['top3']:.3f} top5={top['top5']:.3f}")
        return {"loss": avg_loss, **top}

    def fit(self):
        for ep in range(1, self.cfg.epochs_bc + 1):
            tr = self.train_epoch(ep)
            if (ep % self.cfg.val_interval) == 0:
                val = self.validate(ep)
                if val["loss"] is not None and val["loss"] < self.best_val:
                    self.best_val = val["loss"]
                    path = os.path.join(self.cfg.ckpt_dir, f"bc_best.pt")
                    self.model.save(path)
                    print(f"[Save] {path} (val_loss={self.best_val:.4f})")
        # Final save
        self.model.save(os.path.join(self.cfg.ckpt_dir, "bc_last.pt"))

def main():
    cfg = BCDaggerConfig()
    trainer = BCTrainer(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()
