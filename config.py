"""Centralized config for the chess RL-style project."""
from dataclasses import dataclass

@dataclass
class AgentSelect:
    name: str
    params: dict

@dataclass
class HeuristicParams:
    weights: dict[str, float] | None = None
    depth: int = 2
    random_tiebreak: bool = True
    seed: int | None = None

@dataclass
class BCParams:
    checkpoint: str | None = None
    device: str = "cpu"
    temperature: float = 1.0
    topk: int | None = None

@dataclass
class PPOParams:
    checkpoint: str | None = None
    device: str = "cpu"
    clip_ratio: float = 0.2

@dataclass
class AZLiteParams:
    checkpoint: str | None = None
    simulations: int = 64
    cpuct: float = 1.0
    temperature: float = 1.0
    device: str = "cpu"
# Engine defaults
ENGINE_PATH = "stockfish" # or absolute path to stockfish(.exe)
ENGINE_THREADS = 2
ENGINE_HASH_MB = 256
ENGINE_LIMIT_STRENGTH = True
ENGINE_ELO = 1350 # clamp to [800, 2800]
ENGINE_MOVETIME_S = 0.7

# Env defaults
REWARD_WIN = 1.0
REWARD_LOSS = -1.0
REWARD_DRAW = 0.0
REWARD_INTERMEDIATE = 0.0 # sparse reward (0 until terminal)

# Rendering
SHOW_LEGAL_MOVES_HINT = True