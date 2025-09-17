# config/config.py
from dataclasses import dataclass
from typing import List

def milestone_names(max_positions: int = 1_000_000, step: int = 20_000) -> List[str]:
    # bc_000000, bc_020000, ..., bc_1000000
    return [f"bc_{i:06d}.pt" for i in range(0, max_positions + 1, step)]

@dataclass
class EvalConfig:
    # state/action space
    include_promotions: bool = True
    in_planes: int = 26
    action_space_size: int = 64 * 64 * (1 + 4)  # (from,to) x {none,q,r,b,n}

    # model
    trunk_channels: int = 128
    residual_blocks: int = 2

    # evaluation
    games_per_checkpoint: int = 200   # tăng lên 300–500 để CI hẹp hơn
    temperature: float = 1.0
    seeds: List[int] = (1337, 2027, 7)
    device: str = "cuda"              # hoặc "cpu"

    # files
    ckpt_dir: str = "ckpts"
    checkpoints: List[str] = None     # tự sinh từ milestone_names nếu None
    out_csv: str = "results/progress_vs_heuristic.csv"
    out_png: str = "results/progress_vs_heuristic.png"

    def __post_init__(self):
        if self.checkpoints is None:
            self.checkpoints = milestone_names()
            self.checkpoints.append("dagger_final.pt")  # thêm cột cuối nếu có DAgger
        # prefix with ckpt_dir
        self.checkpoints = [f"{self.ckpt_dir}/{n}" for n in self.checkpoints]
