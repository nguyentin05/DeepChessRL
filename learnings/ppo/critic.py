# learnings/ppo/critic.py
import torch as T, torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_layers=(256,256)) -> None:
        super().__init__()
        layers = []; in_f = state_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_f, h), nn.ReLU()]; in_f = h
        layers += [nn.Linear(in_f, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: T.Tensor):
        return self.model(x).squeeze(-1)
