# chess_env/encode.py
import numpy as np
import chess

# ----- STATE ENCODING -----
# 12 planes: [WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK]
# + 1 plane side-to-move -> 13 planes => 8*8*13 = 832 floats
PIECE_TO_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((13, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None: continue
        p_idx = PIECE_TO_PLANE[piece.piece_type]
        if piece.color == chess.WHITE:
            plane = p_idx
        else:
            plane = 6 + p_idx
        r, c = divmod(square, 8)
        planes[plane, 7 - r, c] = 1.0  # flip rank so white at bottom
    # side to move plane
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return planes.flatten()  # (832,)

# ----- ACTION ENCODING (fixed 64*64*5) -----
# index = from*64*5 + to*5 + promo_id ; promo_id: 0=None,1=Q,2=R,3=B,4=N
PROMO_MAP = {
    None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4
}
REV_PROMO = {v: k for k, v in PROMO_MAP.items()}

ACTION_DIM = 64 * 64 * 5

def move_to_index(move: chess.Move) -> int:
    promo_id = PROMO_MAP[move.promotion if move.promotion in PROMO_MAP else None]
    return move.from_square * 320 + move.to_square * 5 + promo_id

def index_to_move(idx: int) -> tuple[int, int, int]:
    from_sq = idx // 320
    rem = idx % 320
    to_sq = rem // 5
    promo_id = rem % 5
    return from_sq, to_sq, promo_id

def legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
    for mv in board.legal_moves:
        idx = move_to_index(mv)
        mask[idx] = True
    return mask

def index_to_chess_move(idx: int) -> chess.Move:
    f, t, p = index_to_move(idx)
    promo_piece = REV_PROMO[p]
    if promo_piece is None:
        return chess.Move(f, t)
    return chess.Move(f, t, promotion=promo_piece)
