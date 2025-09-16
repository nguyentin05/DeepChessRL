# agents/bc_agent.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict

import torch
import torch.nn.functional as F
import chess

from .base_agent import BaseAgent
from datasets.move_encoder import MoveEncoder
from models.policy_net import PolicyNet


@dataclass
class BCConfig:
    in_channels: int = 14
    action_dim: int = 4864
    device: str = "cpu"
    temperature: float = 1.0
    topk: Optional[int] = None
    checkpoint: Optional[str] = None


class BCAgent(BaseAgent):
    def __init__(
        self,
        model: PolicyNet,
        encoder: MoveEncoder,
        device: str = "cpu",
        temperature: float = 1.0,
        topk: Optional[int] = None,
    ) -> None:
        self.model = model.to(device)
        self.encoder = encoder
        self.device = device
        self.temperature = max(1e-6, float(temperature))
        self.topk = topk
        self._train = False

    @classmethod
    def from_config(cls, cfg: Any) -> "BCAgent":
        if isinstance(cfg, dict):
            cfg = BCConfig(**cfg)
        elif not isinstance(cfg, BCConfig):
            raise TypeError("cfg must be dict or BCConfig")

        encoder = MoveEncoder(action_dim=cfg.action_dim)
        model = PolicyNet(in_channels=cfg.in_channels, action_dim=cfg.action_dim)
        agent = cls(model=model, encoder=encoder, device=cfg.device,
                    temperature=cfg.temperature, topk=cfg.topk)
        if cfg.checkpoint:
            agent.load(cfg.checkpoint)
        agent.eval_mode()
        return agent

    # ---------- Runtime (inference) ----------
    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        self.model.eval()
        obs = self.encoder.board_to_tensor(board)                       # (C,8,8) np
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)  # (1,C,8,8)
        legal_mask_np = self.encoder.legal_mask(board)                  # (A,)
        legal_mask = torch.from_numpy(legal_mask_np).to(self.device)    # (A,)

        logits = self.model(obs_t, legal_mask=None)[0]                  # (A,)
        # mask illegal
        illegal = (legal_mask == 0)
        logits = logits.masked_fill(illegal, -1e9)

        # temperature & (optional) top-k
        if self.temperature != 1.0:
            logits = logits / self.temperature

        if self.topk is not None and self.topk > 0:
            topk = min(self.topk, logits.numel())
            values, idxs = torch.topk(logits, k=topk, dim=-1)
            probs = F.softmax(values, dim=-1)
            choice = torch.multinomial(probs, num_samples=1).item()
            act_idx = idxs[choice].item()
        else:
            act_idx = torch.argmax(logits).item()

        move = self.encoder.index_to_move(act_idx, board)
        if move not in board.legal_moves:
            # Phòng hờ: chọn fallback = nước hợp lệ có xác suất cao nhất
            sorted_idx = torch.argsort(logits, descending=True).tolist()
            for i in sorted_idx:
                mv = self.encoder.index_to_move(int(i), board)
                if mv in board.legal_moves:
                    return mv
            raise RuntimeError("No legal move found by BCAgent (mask mismatch).")
        return move

    # ---------- Training ----------
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        batch: {
          'obs': FloatTensor (B,C,8,8),
          'legal': FloatTensor/Bool (B,A),
          'act': LongTensor (B,)
        }
        Trả về: {'loss': ..., 'acc': ...}
        """
        self.model.train()
        obs = batch["obs"].to(self.device)
        legal = batch["legal"].to(self.device)
        act = batch["act"].to(self.device)

        logits = self.model(obs, legal_mask=None)  # (B,A)
        # mask illegal
        logits = logits.masked_fill(legal == 0, -1e9)

        loss = F.cross_entropy(logits, act)

        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            acc = (pred == act).float().mean().item()

        loss.backward()
        # optimizer step nằm ngoài agent (do bạn quản lý trong train script)
        return {"loss": float(loss.detach().cpu()), "acc": float(acc)}

    # ---------- Modes ----------
    def eval_mode(self) -> None:
        self._train = False
        self.model.eval()

    def train_mode(self) -> None:
        self._train = True
        self.model.train()

    # ---------- IO ----------
    def save(self, path: str) -> None:
        ckpt = {
            "model_state": self.model.state_dict(),
            "encoder": {"action_dim": self.encoder.action_dim, "planes": self.encoder.planes},
            "config": {
                "device": self.device,
                "temperature": self.temperature,
                "topk": self.topk,
            }
        }
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        enc = ckpt.get("encoder", {})
        if enc:
            # đảm bảo khớp
            assert enc.get("action_dim", self.encoder.action_dim) == self.encoder.action_dim
            assert enc.get("planes", self.encoder.planes) == self.encoder.planes
