# expert/board_encoding.py
import numpy as np, chess

PIECE_PLANES = [
    (chess.PAWN,   chess.WHITE), (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE), (chess.ROOK,   chess.WHITE),
    (chess.QUEEN,  chess.WHITE), (chess.KING,   chess.WHITE),
    (chess.PAWN,   chess.BLACK), (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK), (chess.ROOK,   chess.BLACK),
    (chess.QUEEN,  chess.BLACK), (chess.KING,   chess.BLACK),
]

def board_to_planes(board: chess.Board) -> np.ndarray:
    planes = np.zeros((8, 8, 26), dtype=np.uint8)
    # 12 planes quân
    for i, (ptype, color) in enumerate(PIECE_PLANES):
        for sq in board.pieces(ptype, color):
            r, c = divmod(sq, 8)
            planes[7 - r, c, i] = 1
    # side-to-move
    planes[:, :, 12] = 1 if board.turn == chess.WHITE else 0
    # castling
    planes[:, :, 13] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    planes[:, :, 14] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    planes[:, :, 15] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    planes[:, :, 16] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    # en-passant file (8)
    ep = board.ep_square
    if ep is not None:
        planes[:, :, 17 + chess.square_file(ep)] = 1
    # halfmove clock (0..255)
    planes[:, :, 25] = int(round(min(board.halfmove_clock, 100) / 100.0 * 255))
    return planes
