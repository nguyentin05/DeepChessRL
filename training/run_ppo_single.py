# training/run_ppo_single.py
import os, csv
from config.config import TrainConfig
from chess_env.env_py_chess import PyChessEnv
from learnings.ppo.ppo import PPO
from agents.single_agent import SingleAgent
from ultis import set_cpu_threads

def main():
    cfg = TrainConfig()
    set_cpu_threads(cfg.device_threads)
    os.makedirs(os.path.dirname(cfg.csv_episode_log), exist_ok=True)

    env = PyChessEnv(max_halfmoves=cfg.max_halfmoves)
    learner = PPO(
        environment=env,
        hidden_layers=cfg.hidden_layers,
        epochs=cfg.epochs,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        policy_clip=cfg.policy_clip,
        learning_rate=cfg.learning_rate,
    )
    agent = SingleAgent(env, learner, cfg.episodes, cfg.train_on, cfg.result_folder,
                        agent_plays_both_sides=cfg.agent_plays_both_sides)
    agent.train(render_each=cfg.render_each, save_on_learn=True)

    # lưu CSV per-episode
    with open(cfg.csv_episode_log, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["episode","reward","moves","checks","mates"])
        for ep in range(cfg.episodes):
            wr.writerow([ep, agent.rewards[ep], agent.moves[ep], agent.checks[ep], agent.mates[ep]])
    print("[done] training & log saved:", cfg.csv_episode_log)

if __name__ == "__main__":
    main()
