# expert/action_encoder.py
import numpy as np, chess

PROMO_MAP = {None: 0, "q": 1, "r": 2, "b": 3, "n": 4}
REV_PROMO_MAP = {v: k for k, v in PROMO_MAP.items()}

class ActionEncoder:
    def __init__(self, include_promotions: bool = True):
        self.include_promotions = include_promotions
        self.base = 64 * 64
        self.action_space_size = self.base * (5 if include_promotions else 1)

    def encode(self, from_sq: int, to_sq: int, promo: str | None) -> int:
        idx = from_sq * 64 + to_sq
        return idx if (not self.include_promotions or promo is None) else PROMO_MAP[promo] * self.base + idx

    def decode(self, aid: int) -> tuple[int, int, str | None]:
        if not self.include_promotions: return aid // 64, aid % 64, None
        bucket, inner = aid // self.base, aid % self.base
        return inner // 64, inner % 64, REV_PROMO_MAP.get(bucket, None)

    def _promo_str(self, mv: chess.Move) -> str | None:
        return {chess.QUEEN:"q", chess.ROOK:"r", chess.BISHOP:"b", chess.KNIGHT:"n"}.get(mv.promotion, None)

    def valid_action_mask(self, board: chess.Board) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=bool)
        for mv in board.legal_moves:
            mask[self.encode(mv.from_square, mv.to_square, self._promo_str(mv))] = True
        return mask

    def to_move(self, aid: int, board: chess.Board) -> chess.Move:
        f,t,pr = self.decode(aid)
        prom = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}.get(pr)
        mv = chess.Move(f,t,promotion=prom)
        if mv not in board.legal_moves and prom is None:
            if board.piece_at(f) and board.piece_at(f).piece_type == chess.PAWN and chess.square_rank(t) in (0,7):
                mv = chess.Move(f,t,promotion=chess.QUEEN)
        return mv
