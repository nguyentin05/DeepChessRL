# buffer/episode.py
import numpy as np

class Episode:
    def __init__(self) -> None:
        self.states = []; self.rewards = []; self.actions = []
        self.logprobs = []; self.values = []; self.masks = []; self.dones = []
        self.checks = 0; self.mates = 0; self.moves = 0  # thống kê cho agent

    def add(self, state, reward, action, done, logprob=None, value=None, mask=None,
            move_inc=0, check_inc=0, mate_inc=0):
        self.states.append(state); self.rewards.append(reward); self.actions.append(action)
        self.dones.append(bool(done))
        if logprob is not None: self.logprobs.append(logprob)
        if value   is not None: self.values.append(value)
        if mask    is not None: self.masks.append(mask)
        self.moves += move_inc
        self.checks += check_inc
        self.mates  += mate_inc

    def calc_advantage(self, gamma: float, lam: float):
        r  = np.asarray(self.rewards, dtype=np.float32)
        v  = np.asarray(self.values,  dtype=np.float32)
        dn = np.asarray(self.dones,   dtype=np.float32)
        n  = len(r)
        if len(v) == n: v = np.concatenate([v, [0.0]], axis=0)
        adv = np.zeros(n, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(n)):
            nonterm = 1.0 - dn[t]
            delta = r[t] + gamma * v[t+1] * nonterm - v[t]
            lastgaelam = delta + gamma * lam * nonterm * lastgaelam
            adv[t] = lastgaelam
        return adv.tolist()

    def __len__(self): return len(self.rewards)
    def total_reward(self) -> float: return float(sum(self.rewards))
