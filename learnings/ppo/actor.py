# learnings/ppo/actor.py
import torch as T, torch.nn as nn
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers=(256,256)) -> None:
        super().__init__()
        layers = []; in_f = state_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_f, h), nn.ReLU()]; in_f = h
        layers += [nn.Linear(in_f, action_dim)]  # logits
        self.base = nn.Sequential(*layers)

    def forward(self, states: T.Tensor, action_mask: T.Tensor) -> Categorical:
        logits = self.base(states)
        big_neg = T.finfo(logits.dtype).min / 4
        masked_logits = T.where(action_mask.bool(), logits, T.full_like(logits, big_neg))
        return Categorical(logits=masked_logits)
