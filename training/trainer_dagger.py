# training/trainer_dagger.py
from __future__ import annotations
import os, time, numpy as np, torch, torch.nn.functional as F, chess
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset, DataLoader

from models.bc_policy_net import BCPolicyNet
from agents.bc_agent import BCAgent
from agents.heuristic_agent import HeuristicAgent
from expert.action_encoder import ActionEncoder
from expert.board_encoding import board_to_planes
from expert.expert_oracle import ExpertOracle

@dataclass
class DAggerConfig:
    in_planes:int=26; trunk_channels:int=128; residual_blocks:int=2; action_space_size:int=64*64*5
    device:str="cuda"; init_ckpt:str="ckpts/bc_1000000.pt"
    iters:int=5; games_per_iter:int=100; max_positions_per_iter:int=20_000
    batch_size:int=512; lr:float=5e-4; weight_decay:float=1e-4; epochs_per_iter:int=6
    stockfish_path:str="stockfish"; skill:int=3; depth:int=10; multipv:int=5
    out_ckpt:str="ckpts/dagger_final.pt"

class MemDS(Dataset):
    def __init__(self): self.X=[]; self.M=[]; self.Y=[]
    def __len__(self): return len(self.Y)
    def __getitem__(self,i):
        x=torch.from_numpy(self.X[i].astype(np.float32)/255.0).permute(2,0,1)
        m=torch.from_numpy(self.M[i].astype(np.bool_)); y=torch.tensor(int(self.Y[i]),dtype=torch.long)
        return x,m,y
    def extend(self,x,m,y): self.X+=x; self.M+=m; self.Y+=y

def masked_ce(logits,y,m):
    big_neg=torch.finfo(logits.dtype).min/2
    return F.nll_loss(F.log_softmax(torch.where(m,logits,torch.full_like(logits,big_neg)),dim=-1), y)

def run_dagger(cfg: DAggerConfig):
    os.makedirs(os.path.dirname(cfg.out_ckpt), exist_ok=True)
    enc=ActionEncoder(True)
    model=BCPolicyNet(cfg.in_planes,cfg.trunk_channels,cfg.residual_blocks,cfg.action_space_size).to(cfg.device)
    if os.path.isfile(cfg.init_ckpt): model.load_state_dict(torch.load(cfg.init_ckpt, map_location=cfg.device))
    bc=BCAgent(model, enc, device=cfg.device, temperature=1.0); heu=HeuristicAgent()
    oracle=ExpertOracle(cfg.stockfish_path,cfg.skill,cfg.depth,cfg.multipv,enc); oracle.start()
    buf=MemDS()

    for it in range(1,cfg.iters+1):
        print(f"\n==== DAgger {it}/{cfg.iters} ====")
        # rollout
        X=[]; M=[]; Y=[]; added=0
        for g in range(cfg.games_per_iter):
            b=chess.Board()
            while not b.is_game_over() and added<cfg.max_positions_per_iter:
                mv = bc.select_move(b, greedy=True) if b.turn==chess.WHITE else heu.select_move(b)
                if b.turn==chess.WHITE and added<cfg.max_positions_per_iter:
                    X.append(board_to_planes(b)); M.append(enc.valid_action_mask(b)); Y.append(oracle.best_action_id(b)); added+=1
                b.push(mv)
            if added>=cfg.max_positions_per_iter: break
        buf.extend(X,M,Y); print(f"[collect] +{added} (total {len(buf)})")
        # train few epochs
        dl=DataLoader(buf,batch_size=cfg.batch_size,shuffle=True,num_workers=2,pin_memory=True)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        for ep in range(1,cfg.epochs_per_iter+1):
            model.train(); t0=time.time(); tot=0;n=0
            for x,m,y in dl:
                x,m,y=x.to(cfg.device),m.to(cfg.device),y.to(cfg.device)
                opt.zero_grad(set_to_none=True)
                loss=masked_ce(model(x,mask=None),y,m); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
                tot+=loss.item()*x.size(0); n+=x.size(0)
            print(f"[train] ep{ep} loss={tot/max(1,n):.4f} time={time.time()-t0:.1f}s")

    torch.save(model.state_dict(), cfg.out_ckpt); oracle.stop()
    print(f"[save] {cfg.out_ckpt}")

if __name__=="__main__":
    run_dagger(DAggerConfig())
