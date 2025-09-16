# f/preprocess_pgn.py
from __future__ import annotations

import argparse
from datasets.pgn_parser import PGNParser
from datasets.move_encoder import MoveEncoder

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True,
                   help="Danh sách file PGN (.pgn | .pgn.gz | .pgn.zst)")
    p.add_argument("--out", required=True, help="Đường dẫn output .npz")
    p.add_argument("--min-elo", type=int, default=None, help="Lọc người chơi Elo >= min_elo")
    p.add_argument("--max-moves", type=int, default=None, help="Giới hạn số ply mỗi game")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    parser = PGNParser(
        src_paths=args.src,
        min_elo=args.min_elo,
        max_moves_per_game=args.max_moves,
    )
    encoder = MoveEncoder()
    parser.to_examples(out_path=args.out, encoder=encoder)
    print(f"Wrote dataset to {args.out}")

if __name__ == "__main__":
    main()
