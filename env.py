#-*- coding: utf-8 -*-
from typing import Dict, Tuple, Optional
import chess


Color = bool # chess.WHITE or chess.BLACK


class ChessEnv:
    """A lightweight Gym-like chess environment using python-chess.


    Observations: FEN string (simple & framework-agnostic).
    Actions: chess.Move objects (caller ensures legality).


    Reward (sparse): +1 win / -1 loss / 0 draw / 0 otherwise,
    computed from the perspective of `agent_color`.
    """


    def __init__(self, agent_color: Color = chess.BLACK,
    reward_win: float = 1.0,
    reward_loss: float = -1.0,
    reward_draw: float = 0.0,
    reward_intermediate: float = 0.0):
        self.agent_color = agent_color
        self._r_win = reward_win
        self._r_loss = reward_loss
        self._r_draw = reward_draw
        self._r_mid = reward_intermediate
        self.board = chess.Board()


    # --- core API ---
    def reset(self) -> str:
        """Reset board to the initial position. Returns initial observation (FEN)."""
        self.board = chess.Board()
        return self._observation()


    def step(self, move: chess.Move) -> Tuple[str, float, bool, Dict]:
        """Apply one legal move. Returns (obs, reward, done, info)."""
        if move not in self.board.legal_moves:
            raise ValueError("Illegal move for current position: %s" % move)
        self.board.push(move)


        # Terminal check
        if self.board.is_game_over():
            reward = self._terminal_reward()
            return self._observation(), reward, True, self._terminal_info()
        else:
            return self._observation(), self._r_mid, False, {}


    # --- helpers ---
    def legal_moves(self):
        return list(self.board.legal_moves)

    def legal_moves_san(self):
        out = []
        for mv in self.board.legal_moves:
            try:
                out.append(self.board.san(mv))
            except Exception:
                pass
        return out

    def render(self) -> None:
        print(self.__str__())

    def __str__(self) -> str:
        turn = "Trắng" if self.board.turn == chess.WHITE else "Đen"
        return f"\n{self.board}\nLượt: {turn}\n"

    def _observation(self) -> str:
        return self.board.fen()

    def _terminal_reward(self) -> float:
        oc = self.board.outcome()
        if oc is None:
            return self._r_mid
        if oc.winner is None:
            return self._r_draw
        return self._r_win if oc.winner == self.agent_color else self._r_loss

    def _terminal_info(self) -> Dict:
        oc = self.board.outcome()
        if oc is None:
            return {"done": False}
        res = self.board.result()  # '1-0', '0-1', '1/2-1/2'
        winner = oc.winner  # True/False/None
        term = oc.termination.name if oc.termination else "UNKNOWN"
        if winner is True:
            who = "Trắng thắng"
        elif winner is False:
            who = "Đen thắng"
        else:
            who = "Hòa"
        return {"done": True, "result": res, "who": who, "termination": term}