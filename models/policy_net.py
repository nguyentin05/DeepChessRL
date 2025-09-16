# f/policy_net.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class PolicyNet(nn.Module):
    """
    CNN đơn giản cho (C,8,8) -> logits (A=4864).
    Bạn có thể thay bằng ResNet/Transformer sau.
    """
    def __init__(self, in_channels: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.planes = action_dim // 64

        # trunk CNN
        self.b1 = ConvBlock(in_channels, 128)
        self.b2 = ConvBlock(128, 128)
        self.b3 = ConvBlock(128, 128)

        # head
        self.fc = nn.Sequential(
            nn.Flatten(),              # 128*8*8 = 8192
            nn.Linear(8192, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.b1(x)
        h = self.b2(h)
        h = self.b3(h)
        logits = self.fc(h)  # (B,A)
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask == 0, -1e9)
        return logits
