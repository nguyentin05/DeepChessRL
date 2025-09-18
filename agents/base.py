# agents/base.py
from abc import ABC, abstractmethod
import os, csv, numpy as np
from buffer.episode import Episode

class BaseAgent(ABC):
    def __init__(self, env, learner, episodes: int, train_on: int, result_folder: str) -> None:
        super().__init__()
        self.env = env; self.learner = learner
        self.episodes = episodes; self.train_on = train_on
        self.result_folder = result_folder
        os.makedirs(self.result_folder, exist_ok=True)

        # thống kê theo episode (agent-centric)
        self.rewards = np.zeros((episodes,), dtype=np.float32)
        self.moves   = np.zeros((episodes,), dtype=np.int32)
        self.checks  = np.zeros((episodes,), dtype=np.int32)
        self.mates   = np.zeros((episodes,), dtype=np.int32)

    @abstractmethod
    def train(self, render_each: int, save_on_learn: bool = True): ...
