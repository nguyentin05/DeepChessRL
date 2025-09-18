# learnings/ppo/ppo.py
import os, numpy as np, torch as T, torch.optim as optim
from tqdm import tqdm
from buffer.ppo.module import BufferPPO
from buffer.episode import Episode
from learnings.base import Learning
from learnings.ppo.actor import Actor
from learnings.ppo.critic import Critic

class PPO(Learning):
    def __init__(self, environment, hidden_layers, epochs, buffer_size, batch_size,
                 gamma=0.99, gae_lambda=0.95, policy_clip=0.2, learning_rate=2.5e-4) -> None:
        super().__init__(environment, epochs, gamma, learning_rate)
        self.gae_lambda = gae_lambda; self.policy_clip = policy_clip
        self.buffer = BufferPPO(gamma=gamma, max_size=buffer_size, batch_size=batch_size, gae_lambda=gae_lambda)
        self.actor = Actor(self.state_dim, self.action_dim, hidden_layers)
        self.critic = Critic(self.state_dim, hidden_layers)
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=learning_rate, eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=learning_rate, eps=1e-5)
        self.to(self.device)

    @T.no_grad()
    def take_action(self, state: np.ndarray, action_mask: np.ndarray):
        s = T.tensor(state, dtype=T.float32, device=self.device).unsqueeze(0)
        m = T.tensor(action_mask, dtype=T.bool,   device=self.device).unsqueeze(0)
        dist = self.actor(s, m)
        a    = dist.sample()
        logp = T.squeeze(dist.log_prob(a)).item()
        v    = T.squeeze(self.critic(s)).item()
        return int(a.item()), float(logp), float(v)

    def epoch(self):
        (S, A, R, D, oldLP, V, M, ADV, batches) = self.buffer.sample()
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        for batch in batches:
            states = T.tensor(S[batch], dtype=T.float32, device=self.device)
            actions= T.tensor(A[batch], dtype=T.long,    device=self.device)
            masks  = T.tensor(M[batch], dtype=T.bool,    device=self.device)
            old_lp = T.tensor(oldLP[batch], dtype=T.float32, device=self.device)
            adv    = T.tensor(ADV[batch],   dtype=T.float32, device=self.device)
            vals   = T.tensor(V[batch],     dtype=T.float32, device=self.device)

            dist = self.actor(states, masks)
            critic_value = self.critic(states)
            new_lp = dist.log_prob(actions)
            ratio = (new_lp - old_lp).exp()

            s1 = ratio * adv
            s2 = T.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * adv
            actor_loss = -T.min(s1, s2).mean()
            returns = adv + vals
            critic_loss = 0.5 * (returns - critic_value).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            T.nn.utils.clip_grad_norm_(list(self.actor.parameters())+list(self.critic.parameters()), 0.5)
            self.actor_opt.step(); self.critic_opt.step()

    def learn(self):
        for _ in tqdm(range(self.epochs), desc="PPO Learning", ncols=64, leave=False):
            self.epoch()
        self.buffer.clear()

    def remember(self, ep: Episode): self.buffer.add(ep)

    def save(self, folder: str, name: str):
        os.makedirs(folder, exist_ok=True)
        T.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
               os.path.join(folder, f"{name}"))
