# d/pgn_parser.py
from __future__ import annotations

from typing import Iterable, Optional, List
import os
import io
import gzip

try:
    import zstandard as zstd  # optional; dùng cho .zst
except Exception:
    zstd = None

import chess
import chess.pgn
import numpy as np

from .move_encoder import MoveEncoder


class PGNParser:
    """
    Đọc một hoặc nhiều file PGN (hỗ trợ .pgn, .pgn.gz, .pgn.zst nếu zstd có).
    Sinh ra các mẫu (obs, legal_mask, action_idx) cho mỗi ply trong game.
    """
    def __init__(self, src_paths: List[str],
                 min_elo: Optional[int] = None,
                 variants: Optional[set[str]] = None,
                 max_moves_per_game: Optional[int] = None) -> None:
        self.src_paths = src_paths
        self.min_elo = min_elo
        self.variants = variants or {"Standard"}
        self.max_moves_per_game = max_moves_per_game

    def _open_any(self, path: str) -> io.TextIOBase:
        if path.endswith(".gz"):
            return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
        if path.endswith(".zst"):
            if zstd is None:
                raise RuntimeError("Install 'zstandard' to read .zst files")
            dctx = zstd.ZstdDecompressor()
            stream = dctx.stream_reader(open(path, "rb"))
            return io.TextIOWrapper(stream, encoding="utf-8", newline="")
        return open(path, "rt", encoding="utf-8", newline="")

    def iter_games(self) -> Iterable[chess.pgn.Game]:
        for p in self.src_paths:
            with self._open_any(p) as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    yield game

    def to_examples(self, out_path: str, encoder: MoveEncoder,
                    split_ratio: tuple[float, float, float] = (0.96, 0.02, 0.02)) -> None:
        """
        Ghi ra .npz có 3 split: train/val/test — mỗi split là list các dict (obs, legal, act)
        Do .npz không lưu list dict trực tiếp tốt, ta lưu dưới dạng mảng object numpy.
        """
        train, val, test = [], [], []

        for game in self.iter_games():
            # lọc variant
            variant = game.headers.get("Variant", "Standard")
            if variant not in self.variants:
                continue

            # lọc ELO (nếu đặt)
            try:
                we = int(game.headers.get("WhiteElo", "0"))
                be = int(game.headers.get("BlackElo", "0"))
                if self.min_elo is not None and (we < self.min_elo or be < self.min_elo):
                    continue
            except ValueError:
                if self.min_elo is not None:
                    continue

            board = game.board()
            node = game

            ply_count = 0
            for mv in game.mainline_moves():
                if self.max_moves_per_game and ply_count >= self.max_moves_per_game:
                    break

                obs = encoder.board_to_tensor(board)             # (C,8,8)
                legal = encoder.legal_mask(board)                # (A,)
                act_idx = encoder.move_to_index(mv, board)       # int

                # chỉ nhận nếu mapping được
                if act_idx is not None:
                    example = {"obs": obs, "legal": legal, "act": np.int64(act_idx)}
                    # split đơn giản theo modulo (để không cần giữ hết vào RAM)
                    r = (ply_count % 100)
                    if r < split_ratio[0] * 100:
                        train.append(example)
                    elif r < (split_ratio[0] + split_ratio[1]) * 100:
                        val.append(example)
                    else:
                        test.append(example)

                board.push(mv)
                ply_count += 1

        # Lưu ra npz (mảng object)
        np.savez_compressed(
            out_path,
            train=np.array(train, dtype=object),
            val=np.array(val, dtype=object),
            test=np.array(test, dtype=object),
        )
