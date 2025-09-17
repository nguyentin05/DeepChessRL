# models/bc_policy_net.py
import torch, torch.nn as nn, torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x))); y = self.b2(self.c2(y))
        return F.relu(x + y)

class BCPolicyNet(nn.Module):
    def __init__(self, in_planes: int, trunk_channels: int, residual_blocks: int, action_space_size: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, trunk_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels), nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResidualBlock(trunk_channels) for _ in range(residual_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(trunk_channels, trunk_channels, 1, bias=False),
            nn.BatchNorm2d(trunk_channels), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(trunk_channels, action_space_size)
        nn.init.kaiming_normal_(self.stem[0].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.head[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, mask=None):
        y = self.stem(x); y = self.res(y); y = self.head(y); y = y.mean(dim=(2,3))
        logits = self.fc(y)
        if mask is not None: logits = logits.masked_fill(~mask, -1e9)
        return logits
