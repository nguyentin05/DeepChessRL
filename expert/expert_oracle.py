# expert/expert_oracle.py
import chess, chess.engine
from expert.action_encoder import ActionEncoder
MATE_CP=100000

class ExpertOracle:
    def __init__(self, engine_path="stockfish", skill=3, depth=10, multipv=5, encoder: ActionEncoder|None=None):
        self.path=engine_path; self.skill=skill; self.depth=depth; self.multipv=multipv
        self.enc=encoder or ActionEncoder(True); self.eng=None
    def start(self):
        self.eng=chess.engine.SimpleEngine.popen_uci(self.path)
        try: self.eng.configure({"Skill Level": self.skill})
        except: pass
    def stop(self):
        if self.eng: self.eng.quit(); self.eng=None
    def _cp(self, score: chess.engine.PovScore, wtm: bool)->float:
        pov=score.white() if wtm else score.black()
        return (1 if (pov.is_mate() and pov.mate()>0) else -1)*MATE_CP if pov.is_mate() else float(pov.score())
    def best_action_id(self, board: chess.Board)->int:
        info=self.eng.analyse(board,limit=chess.engine.Limit(depth=self.depth),multipv=self.multipv)
        if not isinstance(info,list): info=[info]
        best=None; sc=-1e18
        for ln in info:
            if "pv" not in ln or "score" not in ln: continue
            mv=ln["pv"][0]; cp=self._cp(ln["score"], board.turn==chess.WHITE)
            aid=self.enc.encode(mv.from_square, mv.to_square,
                "q" if mv.promotion==chess.QUEEN else "r" if mv.promotion==chess.ROOK else
                "b" if mv.promotion==chess.BISHOP else "n" if mv.promotion==chess.KNIGHT else None)
            if cp>sc: sc=cp; best=aid
        if best is None:
            mask=self.enc.valid_action_mask(board); best=int(list(mask.nonzero()[0])[0])
        return best
