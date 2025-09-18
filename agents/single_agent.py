# agents/single_agent.py
import os, numpy as np
from tqdm import tqdm
import chess
from buffer.episode import Episode
from agents.base import BaseAgent

class SingleAgent(BaseAgent):
    def __init__(self, env, learner, episodes, train_on, result_folder, agent_plays_both_sides=True):
        super().__init__(env, learner, episodes, train_on, result_folder)
        self.agent_plays_both_sides = agent_plays_both_sides

    def train(self, render_each: int, save_on_learn: bool = True):
        current_ep = 0
        for ep in (pbar := tqdm(range(self.episodes))):
            agent_is_white = True
            if self.agent_plays_both_sides:
                agent_is_white = (ep % 2 == 0)
            state, mask = self.env.reset(agent_is_white=agent_is_white)
            episode = Episode()

            done = False
            while not done:
                action, logp, value = self.learner.take_action(state, mask)
                next_state, next_mask, reward, done, info = self.env.step_agent(action)
                # tăng thống kê (mỗi nước đi của agent là 1 step ghi vào episode)
                move_inc = 1
                check_inc = 1 if info.get("after") in ("agent", "opp_end") and reward > 0 and "check" in info.get("events","") else 0
                mate_inc  = 0  # mate đếm ở env (gán vào episode khi done)
                episode.add(state, reward, action, done, logp, value, mask,
                            move_inc=move_inc, check_inc=0, mate_inc=0)
                state, mask = next_state, next_mask

            # cập nhật episode theo env (checks/mates đã đếm trong env)
            episode.moves  = self.env.agent_moves
            episode.checks = self.env.agent_checks
            episode.mates  = self.env.agent_mates

            self.learner.remember(episode)
            self.rewards[ep] = episode.total_reward()
            self.moves[ep]   = episode.moves
            self.checks[ep]  = episode.checks
            self.mates[ep]   = episode.mates

            pbar.set_postfix({"ep": ep, "R": self.rewards[ep], "M": int(self.moves[ep]),
                              "Ck": int(self.checks[ep]), "Mt": int(self.mates[ep])})

            if (ep + 1) % self.train_on == 0:
                self.learner.learn()
                if save_on_learn:
                    self.learner.save(self.result_folder, "ppo.pt")
                    np.save(os.path.join(self.result_folder, "rewards.npy"), self.rewards)
                    np.save(os.path.join(self.result_folder, "moves.npy"),   self.moves)
                    np.save(os.path.join(self.result_folder, "checks.npy"),  self.checks)
                    np.save(os.path.join(self.result_folder, "mates.npy"),   self.mates)
