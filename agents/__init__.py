"""Agent package exports."""
from .heuristic_agent import HeuristicAgent
from .single_agent import SingleAgent
from .stockfish_agent import StockfishAgent, StockfishConfig

__all__ = ["StockfishAgent", "StockfishConfig"]
__version__ = "0.1.0"