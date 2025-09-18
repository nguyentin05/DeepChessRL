# config/config.py
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Env
    max_halfmoves: int = 300          # số ply tối đa (đôi bên) để chặn ván quá dài
    render_mode: str = "rgb_array"    # không dùng GUI
    agent_plays_both_sides: bool = True  # xen kẽ: chẵn=White, lẻ=Black

    # PPO & buffer
    hidden_layers: tuple = (256, 256)
    epochs: int = 3
    buffer_size: int = 256
    batch_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    learning_rate: float = 2.5e-4

    # Training loop
    episodes: int = 5000
    train_on: int = 32                 # gom N ván rồi mới học
    render_each: int = 10**9           # tắt render

    # Buckets đánh giá (0-500, 500-1000, ...)
    bucket_size: int = 500

    # Kết quả
    result_folder: str = "results"
    csv_episode_log: str = "results/episodes_log.csv"
    csv_bucket_summary: str = "results/bucket_summary.csv"
    plots_dir: str = "results/plots"

    # Device
    device_threads: int = 4
