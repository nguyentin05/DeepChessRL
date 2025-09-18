# agents/heuristic_agent.py
import chess

PV = {  # material weights
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
}

class HeuristicAgent:
    def score_move(self, board: chess.Board, move: chess.Move) -> float:
        # Ưu tiên bắt quân + nước chiếu + phát triển quân nhẹ
        score = 0.0
        if board.is_capture(move):
            cap_sq = move.to_square
            cap_piece = board.piece_at(cap_sq)
            if cap_piece:
                score += PV.get(cap_piece.piece_type, 0) * 10.0
        if board.gives_check(move):
            score += 1.0
        # ưu tiên đưa quân ra trung tâm nhẹ
        to_r, to_c = divmod(move.to_square, 8)
        center_bonus = (3.5 - abs(to_r - 3.5)) + (3.5 - abs(to_c - 3.5))
        score += 0.01 * center_bonus
        return score
