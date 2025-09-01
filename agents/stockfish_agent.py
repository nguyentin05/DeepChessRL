# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional
import chess
import chess.engine


@dataclass
class StockfishConfig:
    engine_path: str = "stockfish"
    threads: int = 2
    hash_mb: int = 256
    limit_strength: bool = True
    elo: int = 1350 # clamp to [800, 2800]
    movetime_s: float = 0.7


class StockfishAgent:
    """A thin wrapper that turns Stockfish (UCI) into an 'agents' with select_move()."""


    def __init__(self, cfg: Optional[StockfishConfig] = None):
        self.cfg = cfg or StockfishConfig()
        self._engine = chess.engine.SimpleEngine.popen_uci(self.cfg.engine_path)
        self._configure()


    def _configure(self):
        # Build config dict and set tolerantly (some builds lack certain options)
        options = {
        "Threads": self.cfg.threads,
        "Hash": self.cfg.hash_mb,
        }
        if self.cfg.limit_strength:
            options["UCI_LimitStrength"] = True
            # clamp elo
            elo = max(800, min(int(self.cfg.elo), 2800))
            options["UCI_Elo"] = elo
        for k, v in options.items():
            try:
                self._engine.configure({k: v})
            except Exception:
                pass


    def select_move(self, board: chess.Board) -> chess.Move:
        """Return best move under the configured time limit."""
        result = self._engine.play(board, chess.engine.Limit(time=self.cfg.movetime_s))
        return result.move


    def analyse_hint(self, board: chess.Board, time_limit: float = 1.0) -> Optional[chess.Move]:
        try:
            info = self._engine.analyse(board, chess.engine.Limit(time=time_limit))
            if "pv" in info and len(info["pv"]) > 0:
                return info["pv"][0]
        except Exception:
            return None
        return None


    def close(self):
        try:
            self._engine.close()
        except Exception:
            pass


    def __del__(self):
        self.close()