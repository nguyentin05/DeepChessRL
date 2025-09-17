# agents/heuristic_agent.py
import chess
VAL = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:0}

class HeuristicAgent:
    def __init__(self, mobility_weight=0.2, max_nodes=2000):
        self.mobility_weight = mobility_weight; self.max_nodes=max_nodes

    def _material(self, b: chess.Board) -> int:
        s=0
        for sq in chess.SQUARES:
            pc=b.piece_at(sq)
            if pc: s += (1 if pc.color==b.turn else -1)*VAL[pc.piece_type]
        return s

    def _mob(self, b: chess.Board) -> int:
        b.push(chess.Move.null()); opp=b.legal_moves.count(); b.pop()
        return b.legal_moves.count() - opp

    def _score(self, b: chess.Board) -> float:
        return float(self._material(b) + self.mobility_weight*self._mob(b))

    def select_move(self, b: chess.Board) -> chess.Move:
        best=None; val=-1e18; cnt=0
        for mv in b.legal_moves:
            b.push(mv); v=self._score(b); b.pop()
            if v>val: val=v; best=mv
            cnt+=1
            if cnt>=self.max_nodes: break
        return best or next(iter(b.legal_moves))
