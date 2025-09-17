# eval/evaluator.py
import os, csv, chess, torch
from typing import Dict
from config.config import EvalConfig
from utils.maths import set_global_seed, wilson_ci
from expert.action_encoder import ActionEncoder
from models.bc_policy_net import BCPolicyNet
from agents.bc_agent import BCAgent
from agents.heuristic_agent import HeuristicAgent

def play_one(white, black, max_moves=512)->str:
    b=chess.Board()
    for _ in range(max_moves):
        if b.is_game_over(): return b.result()
        pl=white if b.turn==chess.WHITE else black
        mv=pl.select_move(b)
        if mv is None or mv not in b.legal_moves:
            return "0-1" if b.turn==chess.WHITE else "1-0"
        b.push(mv)
    return "1/2-1/2"

def tuple_res(res:str):
    return (1,0,0) if res=="1-0" else (0,0,1) if res=="0-1" else (0,1,0)

def eval_ckpt(ckpt:str, cfg:EvalConfig)->Dict[str,float]:
    enc=ActionEncoder(cfg.include_promotions)
    net=BCPolicyNet(cfg.in_planes,cfg.trunk_channels,cfg.residual_blocks,cfg.action_space_size)
    if os.path.isfile(ckpt): net.load_state_dict(torch.load(ckpt, map_location=cfg.device))
    bc=BCAgent(net, enc, device=cfg.device, temperature=cfg.temperature)
    heu=HeuristicAgent()
    G=cfg.games_per_checkpoint; w=d=l=0
    for i in range(G):
        white,black=(bc,heu) if i%2==0 else (heu,bc)
        res=play_one(white,black); wi,di,li=tuple_res(res)
        if i%2==0: w+=wi; d+=di; l+=li
        else:      w+=li; d+=di; l+=wi
    p,lo,hi=wilson_ci(w,d,l)
    return {"games":G,"win":w,"draw":d,"loss":l,"winrate":p,"ci_lo":lo,"ci_hi":hi}

def run_eval(cfg:EvalConfig):
    os.makedirs(os.path.dirname(cfg.out_csv), exist_ok=True)
    with open(cfg.out_csv,"w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f)
        wr.writerow(["phase","games","win","draw","loss","winrate","ci_lo","ci_hi"])
        for ckpt in cfg.checkpoints:
            phase=os.path.splitext(os.path.basename(ckpt))[0]
            acc={"win":0,"draw":0,"loss":0,"games":0,"wr":0.0,"lo":0.0,"hi":0.0}
            for s in cfg.seeds:
                set_global_seed(s)
                st=eval_ckpt(ckpt,cfg)
                wr.writerow([f"{phase}_seed{s}",st["games"],st["win"],st["draw"],st["loss"],
                             f"{st['winrate']:.6f}",f"{st['ci_lo']:.6f}",f"{st['ci_hi']:.6f}"])
            print(f"[done] {phase}")
