from __future__ import annotations
from typing import Callable, Any, Dict, Type

from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .heuristic_agent import HeuristicAgent
from .bc_agent import BCAgent, BCConfig
from .ppo_agent import PPOAgent
from .azlite_agent import AZLiteAgent

# Nếu bạn có StockfishAgent, mở import bên dưới:
# from .stockfish_agent import StockfishAgent

def _make_bc(**kwargs: Any) -> BaseAgent:
    """
    Hỗ trợ: in_channels, action_dim, device, temperature, topk, checkpoint
    Ví dụ:
      create_agent("bc", checkpoint="runs/BC/best.pt", device="cpu")
    """
    cfg = BCConfig(
        in_channels=kwargs.get("in_channels", 14),
        action_dim=kwargs.get("action_dim", 4864),
        device=kwargs.get("device", "cpu"),
        temperature=kwargs.get("temperature", 1.0),
        topk=kwargs.get("topk", None),
        checkpoint=kwargs.get("checkpoint", None),
    )
    return BCAgent.from_config(cfg)

# Registry: tên ngắn gọn -> callable tạo Agent
AGENT_REGISTRY: Dict[str, Callable[..., BaseAgent]] = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "bc": _make_bc,
    "ppo": PPOAgent,
    "azlite": AZLiteAgent,
    # "stockfish": StockfishAgent,  # bật nếu dùng kèm
}

def register_agent(name: str, ctor: Callable[..., BaseAgent]) -> None:
    AGENT_REGISTRY[name.strip().lower()] = ctor

def create_agent(name: str, **kwargs: Any) -> BaseAgent:
    """Tạo agent theo tên đã đăng ký; ví dụ: create_agent('heuristic', depth=2)."""
    key = (name or "").strip().lower()
    if key not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent '{name}'. Available: {', '.join(sorted(AGENT_REGISTRY.keys()))}")
    return AGENT_REGISTRY[key](**kwargs)

def available_agents() -> list[str]:
    """Liệt kê các agent khả dụng."""
    return sorted(AGENT_REGISTRY.keys())
