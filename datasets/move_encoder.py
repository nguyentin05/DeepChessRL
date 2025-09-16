# d/move_encoder.py
from __future__ import annotations

from typing import Tuple, List, Dict, Optional
import numpy as np
import chess
import torch


class MoveEncoder:
    """
    Mã hoá hành động theo (from_square, plane):
    planes = 76 = 56 (slide) + 8 (knight) + 12 (promotions)
      - Slide: 8 hướng × 7 bước (dist=1..7)
      - Knight: 8 vector
      - Promotion: 3 hướng (F, FL, FR) × 4 loại (Q,R,B,N) = 12
    action_dim = 64 * 76 = 4864
    """
    SLIDE_DIRS: List[Tuple[int, int]] = [
        (0, 1),  (0, -1), (1, 0),  (-1, 0),   # N,S,E,W
        (1, 1),  (-1, 1), (1, -1), (-1, -1)   # NE,NW,SE,SW
    ]
    KNIGHT_DELTAS: List[Tuple[int, int]] = [
        (1, 2), (2, 1), (-1, 2), (-2, 1),
        (1, -2), (2, -1), (-1, -2), (-2, -1),
    ]
    PROMO_PIECES: List[int] = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    PROMO_DIRS: List[str] = ["F", "FL", "FR"]  # Forward, Forward-Left, Forward-Right

    def __init__(self, action_dim: int = 4864, planes: int = 76) -> None:
        self.action_dim = action_dim
        self.planes = planes
        assert action_dim == 64 * planes, "action_dim must be 64 * planes"

    # ---------- Board encoding ----------
    def board_to_tensor(self, board: chess.Board, planes: int = 14) -> np.ndarray:
        """
        Trả về (C,8,8) float32:
          - 12 plane quân: [P,N,B,R,Q,K] cho White, rồi cho Black
          - 1 plane side-to-move (1 nếu White đến lượt, ngược lại 0)
          - 1 plane tổng hợp castle/en passant đơn giản (tuỳ chọn)
        """
        C = planes
        arr = np.zeros((C, 8, 8), dtype=np.float32)

        piece_order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

        # White pieces (0..5), Black pieces (6..11)
        for i, p in enumerate(piece_order):
            for sq in board.pieces(p, chess.WHITE):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                arr[i, r, c] = 1.0
            for sq in board.pieces(p, chess.BLACK):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                arr[6 + i, r, c] = 1.0

        # side to move
        arr[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

        # simple castle/en passant plane (optional heuristic)
        castle_val = 0.0
        if board.has_kingside_castling_rights(chess.WHITE): castle_val += 0.25
        if board.has_queenside_castling_rights(chess.WHITE): castle_val += 0.25
        if board.has_kingside_castling_rights(chess.BLACK): castle_val += 0.25
        if board.has_queenside_castling_rights(chess.BLACK): castle_val += 0.25
        arr[13, :, :] = castle_val

        return arr

    # ---------- Mask ----------
    def legal_mask(self, board: chess.Board) -> np.ndarray:
        mask = np.zeros((self.action_dim,), dtype=np.float32)
        for mv in board.legal_moves:
            idx = self.move_to_index(mv, board)
            if idx is not None:
                mask[idx] = 1.0
        return mask

    # ---------- Move <-> index ----------
    def move_to_index(self, move: chess.Move, board: chess.Board) -> Optional[int]:
        frm = move.from_square
        to = move.to_square

        f_file, f_rank = chess.square_file(frm), chess.square_rank(frm)
        t_file, t_rank = chess.square_file(to), chess.square_rank(to)
        df, dr = t_file - f_file, t_rank - f_rank

        # Knight?
        if (df, dr) in self.KNIGHT_DELTAS:
            k_idx = self.KNIGHT_DELTAS.index((df, dr))  # 0..7
            plane = 56 + k_idx
            return frm * self.planes + plane

        # Promotion?
        if move.promotion is not None:
            dir_code = self._promo_dir_code(df, dr, board.turn)
            if dir_code is None:
                return None
            try:
                p_idx = self.PROMO_PIECES.index(move.promotion)  # 0..3
            except ValueError:
                return None
            plane = 56 + 8 + (dir_code * 4 + p_idx)  # 64..75
            return frm * self.planes + plane

        # Slide (R,B,Q,K,P one-step etc.) encode as direction + distance
        # Normalize step to unit direction if on straight/diagonal line
        dir_idx, dist = self._slide_dir_and_dist(df, dr)
        if dir_idx is None or dist is None or dist < 1 or dist > 7:
            # e.g., king knight-like? (king 1-step will be captured by dist=1)
            return None
        plane = dir_idx * 7 + (dist - 1)  # 0..55
        return frm * self.planes + plane

    def index_to_move(self, idx: int, board: chess.Board) -> chess.Move:
        frm = idx // self.planes
        plane = idx % self.planes

        f_file, f_rank = chess.square_file(frm), chess.square_rank(frm)

        if plane < 56:
            # slide
            dir_idx = plane // 7
            dist = (plane % 7) + 1
            df, dr = self.SLIDE_DIRS[dir_idx]
            t_file = f_file + df * dist
            t_rank = f_rank + dr * dist
            if not (0 <= t_file <= 7 and 0 <= t_rank <= 7):
                return chess.Move.null()
            to = chess.square(t_file, t_rank)
            return chess.Move(frm, to)

        elif plane < 64:
            # knight
            k_idx = plane - 56
            df, dr = self.KNIGHT_DELTAS[k_idx]
            t_file = f_file + df
            t_rank = f_rank + dr
            if not (0 <= t_file <= 7 and 0 <= t_rank <= 7):
                return chess.Move.null()
            to = chess.square(t_file, t_rank)
            return chess.Move(frm, to)

        else:
            # promotions
            promo_plane = plane - 64  # 0..11
            dir_code = promo_plane // 4        # 0..2
            p_idx = promo_plane % 4            # 0..3
            df, dr = self._promo_delta(dir_code, board.turn)
            t_file = f_file + df
            t_rank = f_rank + dr
            if not (0 <= t_file <= 7 and 0 <= t_rank <= 7):
                return chess.Move.null()
            to = chess.square(t_file, t_rank)
            promo_piece = self.PROMO_PIECES[p_idx]
            return chess.Move(frm, to, promotion=promo_piece)

    # ---------- helpers ----------
    def _slide_dir_and_dist(self, df: int, dr: int) -> Tuple[Optional[int], Optional[int]]:
        # On straight/diagonal line?
        if df == 0 and dr != 0:
            dir_vec = (0, 1 if dr > 0 else -1)
            dist = abs(dr)
        elif dr == 0 and df != 0:
            dir_vec = (1 if df > 0 else -1, 0)
            dist = abs(df)
        elif abs(df) == abs(dr) and df != 0:
            dir_vec = (1 if df > 0 else -1, 1 if dr > 0 else -1)
            dist = abs(df)
        else:
            return None, None

        try:
            dir_idx = self.SLIDE_DIRS.index(dir_vec)
        except ValueError:
            return None, None
        return dir_idx, dist

    def _promo_dir_code(self, df: int, dr: int, stm_white: bool) -> Optional[int]:
        # forward direction depends on side to move
        if stm_white:
            if (df, dr) == (0, 1): return 0       # F
            if (df, dr) == (-1, 1): return 1      # FL
            if (df, dr) == (1, 1): return 2       # FR
        else:
            if (df, dr) == (0, -1): return 0
            if (df, dr) == (1, -1): return 1
            if (df, dr) == (-1, -1): return 2
        return None

    def _promo_delta(self, dir_code: int, stm_white: bool) -> Tuple[int, int]:
        if stm_white:
            return [(0, 1), (-1, 1), (1, 1)][dir_code]
        else:
            return [(0, -1), (1, -1), (-1, -1)][dir_code]

    # ---------- dataloader collate ----------
    def collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        obs = torch.from_numpy(np.stack([b["obs"] for b in batch], axis=0)).float()  # (B,C,8,8)
        legal = torch.from_numpy(np.stack([b["legal"] for b in batch], axis=0)).float()  # (B,A)
        act = torch.from_numpy(np.array([b["act"] for b in batch], dtype=np.int64))
        return {"obs": obs, "legal": legal, "act": act}
