# config/config_bc_dagger.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BCDaggerConfig:
    # data
    data_dir: str = "data"
    shards_dir: str = "data/shards"
    splits_dir: str = "data/splits"
    train_shards: List[str] = None
    val_shards: List[str] = None
    test_shards: List[str] = None

    # PGN filter
    min_elo: int = 1800
    min_ply: int = 40
    allow_time_controls: Optional[List[str]] = None  # e.g. ["classical","rapid","blitz"]
    max_games: int = 200000

    # Expert (Stockfish) - chỉ dùng khi bạn muốn soft labels/DAgger
    stockfish_path: str = "stockfish"
    stockfish_skill: int = 3
    stockfish_depth: int = 10
    multipv_topk: int = 5
    temperature: float = 1.0

    # Action space: 64*64 * (1 + 4 promotions) = 20480
    include_promotions: bool = True
    action_space_size: int = 64 * 64 * 5

    # Model
    in_planes: int = 26  # 12 piece planes + 1 side + 4 castling + 8 ep-file + 1 halfmove
    trunk_channels: int = 128
    residual_blocks: int = 2

    # Train
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs_bc: int = 30
    epochs_dagger_per_iter: int = 8
    seed: int = 1337
    device: str = "cuda"

    # DAgger
    dagger_iters: int = 5
    episodes_per_iter: int = 100
    max_positions_per_iter: int = 20000

    # Eval
    games_vs_stockfish: int = 100
    time_per_move_ms: int = 300

    # IO
    ckpt_dir: str = "ckpts"
    log_interval: int = 100
    val_interval: int = 1

    def __post_init__(self):
        if self.train_shards is None: self.train_shards = []
        if self.val_shards is None: self.val_shards = []
        if self.test_shards is None: self.test_shards = []
