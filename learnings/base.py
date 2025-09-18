# learnings/base.py
import numpy as np, torch as T, torch.nn as nn
from abc import ABC, abstractmethod

class Learning(nn.Module, ABC):
    def __init__(self, environment, epochs: int, gamma: float, learning_rate: float) -> None:
        super().__init__()
        self.state_dim = environment.observation_space.shape[0]
        self.action_dim = environment.action_space.n
        self.gamma = gamma; self.epochs = epochs; self.learning_rate = learning_rate
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    @abstractmethod
    def take_action(self, state: np.ndarray, *args): ...
    @abstractmethod
    def learn(self): ...
    @abstractmethod
    def remember(self, *args): ...
    @abstractmethod
    def save(self, folder: str, name: str): ...
