import chess

def parse_user_move(board: chess.Board, s: str):
    s = (s or "").strip()
    if not s:
        return None
    # Try SAN first
    try:
        mv = board.parse_san(s)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass
    # Try UCI
    try:
        mv = board.parse_uci(s)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass
    return None

def list_legal_moves_san(board: chess.Board):
    out = []
    for mv in board.legal_moves:
        try:
            out.append(board.san(mv))
        except Exception:
            pass
    return out