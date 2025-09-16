# f/train_bc.py
from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import torch
import torch.optim as optim
import torch.nn.functional as F

from datasets.datamodule import build_dataloaders
from datasets.move_encoder import MoveEncoder
from models.policy_net import PolicyNet
from agents.bc_agent import BCAgent, BCConfig


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help=".npz tạo từ preprocess_pgn.py")
    p.add_argument("--in-ch", type=int, default=14)
    p.add_argument("--action-dim", type=int, default=4864)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", default="runs/BC")
    p.add_argument("--save-every", type=int, default=1)
    return p.parse_args()

def evaluate(model: torch.nn.Module, loader, device: str) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(device)
            legal = batch["legal"].to(device)
            act = batch["act"].to(device)
            logits = model(obs, legal_mask=None).masked_fill(legal == 0, -1e9)
            pred = torch.argmax(logits, dim=-1)
            total += act.numel()
            correct += (pred == act).sum().item()
    acc = correct / max(1, total)
    return {"val_acc": acc}

def main() -> None:
    args = _parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader = build_dataloaders(args.data, batch_size=args.batch)

    model = PolicyNet(in_channels=args.in_ch, action_dim=args.action_dim)
    device = args.device
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    step = 0
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            obs = batch["obs"].to(device)
            legal = batch["legal"].to(device)
            act = batch["act"].to(device)

            logits = model(obs, legal_mask=None)
            logits = logits.masked_fill(legal == 0, -1e9)
            loss = F.cross_entropy(logits, act)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1

        metrics = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch}] val_acc={metrics['val_acc']:.4f}")

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"bc_epoch{epoch}.pt")
            torch.save({"model_state": model.state_dict(),
                        "config": {"in_channels": args.in_ch, "action_dim": args.action_dim}}, ckpt_path)
            print(f"Saved {ckpt_path}")

        if metrics["val_acc"] > best_acc:
            best_acc = metrics["val_acc"]
            best_path = os.path.join(args.save_dir, "best.pt")
            torch.save({"model_state": model.state_dict(),
                        "config": {"in_channels": args.in_ch, "action_dim": args.action_dim}}, best_path)
            print(f"Updated best: {best_path} (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()
