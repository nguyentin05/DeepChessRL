# chess_env/rewards.py
import chess

# scale dịu tay để dễ học trên CPU
MOVE_PENALTY = -0.01

CHECK_BONUS = +0.05
CHECK_PENALTY = -0.05

MATE_WIN  = +1.0
MATE_LOSE = -1.0

# Optional: material gain/loss nhỏ
PIECE_VALUE = {
    chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
    chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0
}
MATERIAL_SCALE = 0.02  # nhỏ thôi (9 * 0.02 = 0.18 khi ăn queen)

def material_delta(before: chess.Board, after: chess.Board, agent_color: bool) -> float:
    # đơn giản: đếm tổng vật chất hai bên, reward = (mat_after - mat_before) cho bên agent
    def side_score(bd: chess.Board, color: bool) -> float:
        s = 0.0
        for sq in chess.SQUARES:
            pc = bd.piece_at(sq)
            if not pc: continue
            v = PIECE_VALUE[pc.piece_type]
            s += v if pc.color == color else 0.0
        return s
    mat_before = side_score(before, agent_color) - side_score(before, not agent_color)
    mat_after  = side_score(after,  agent_color) - side_score(after,  not agent_color)
    return (mat_after - mat_before) * MATERIAL_SCALE
