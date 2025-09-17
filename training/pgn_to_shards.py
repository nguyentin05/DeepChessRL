# training/pgn_to_shards.py
from __future__ import annotations
import os, chess, chess.pgn, numpy as np
from dataclasses import dataclass
from typing import List, Optional
from expert.board_encoding import board_to_planes
from expert.action_encoder import ActionEncoder

@dataclass
class PgnToShardsConfig:
    input_paths: List[str]
    out_dir: str = "data/shards"
    positions_per_shard: int = 50_000
    max_shards: int = 50
    min_elo: int = 1800
    min_ply: int = 40
    allow_time_controls: Optional[List[str]] = None
    include_promotions: bool = True

class PgnShardBuilder:
    def __init__(self, cfg: PgnToShardsConfig):
        self.cfg=cfg; os.makedirs(cfg.out_dir, exist_ok=True)
        self.enc = ActionEncoder(cfg.include_promotions)
        self.P=[]; self.M=[]; self.Y=[]; self.sid=0; self.total=0

    def _save(self):
        if not self.Y: return
        out = os.path.join(self.cfg.out_dir, f"bc_{self.sid:05d}.npz")
        np.savez_compressed(out,
            planes=np.stack(self.P,0),
            mask=np.stack(self.M,0),
            y_hard=np.array(self.Y, np.int16))
        self.total += len(self.Y)
        print(f"[save] {out}  +{len(self.Y)}  total={self.total}")
        self.sid += 1; self.P.clear(); self.M.clear(); self.Y.clear()

    def _ok(self, g: chess.pgn.Game)->bool:
        try:
            we=int(g.headers.get("WhiteElo","0") or 0); be=int(g.headers.get("BlackElo","0") or 0)
        except: return False
        if min(we,be) < self.cfg.min_elo: return False
        if self.cfg.allow_time_controls:
            tc=(g.headers.get("TimeControl") or "").lower()
            if not any(k in tc for k in self.cfg.allow_time_controls): return False
        ply=0; node=g
        while node.variations: node=node.variations[0]; ply+=1
        return ply>=self.cfg.min_ply

    def _promo(self, mv: chess.Move):
        return {chess.QUEEN:"q", chess.ROOK:"r", chess.BISHOP:"b", chess.KNIGHT:"n"}.get(mv.promotion,None)

    def _add(self, b: chess.Board, mv: chess.Move):
        self.P.append(board_to_planes(b))
        self.M.append(self.enc.valid_action_mask(b))
        self.Y.append(self.enc.encode(mv.from_square, mv.to_square, self._promo(mv)))

    def build(self):
        for path in self.cfg.input_paths:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    g = chess.pgn.read_game(f)
                    if g is None: break
                    if not self._ok(g): continue
                    b=g.board(); node=g
                    while node.variations:
                        mv=node.variations[0].move
                        self._add(b,mv); b.push(mv)
                        if len(self.Y)>=self.cfg.positions_per_shard:
                            self._save()
                            if self.sid>=self.cfg.max_shards: break
                    if self.sid>=self.cfg.max_shards: break
            if self.sid>=self.cfg.max_shards: break
        self._save()

if __name__=="__main__":
    cfg=PgnToShardsConfig(
        input_paths=["data/raw_pgn/lichess_sample.pgn"],
        out_dir="data/shards", positions_per_shard=50_000, max_shards=20,
        min_elo=1800, min_ply=40, allow_time_controls=None
    ); PgnShardBuilder(cfg).build()
