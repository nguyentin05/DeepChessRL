# buffer/ppo/module.py
import numpy as np
from collections import deque
from buffer.base import Buffer
from buffer.episode import Episode
import ultis

class BufferPPO(Buffer):
    def __init__(self, max_size, batch_size, gamma, gae_lambda, shuffle=True) -> None:
        super().__init__(max_size, batch_size, shuffle)
        self.gamma = gamma; self.gae_lambda = gae_lambda
        self.episodes = deque(maxlen=max_size)
        self.advantages = deque(maxlen=max_size)

    def add(self, episode: Episode):
        self.episodes.append(episode)
        self.advantages.append(episode.calc_advantage(self.gamma, self.gae_lambda))

    def clear(self):
        self.episodes.clear(); self.advantages.clear()

    def get_len(self): return len(self.episodes)

    def sample(self):
        states  = sum((e.states   for e in self.episodes), [])
        actions = sum((e.actions  for e in self.episodes), [])
        rewards = sum((e.rewards  for e in self.episodes), [])
        logps   = sum((e.logprobs for e in self.episodes), [])
        values  = sum((e.values   for e in self.episodes), [])
        masks   = sum((e.masks    for e in self.episodes), [])
        dones   = sum((e.dones    for e in self.episodes), [])
        advs    = sum(self.advantages, [])

        batches = ultis.make_batch_ids(n=len(states), batch_size=self.batch_size, shuffle=self.shuffle)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(dones, dtype=np.bool_),
                np.array(logps), np.array(values), np.array(masks), np.array(advs), batches)
