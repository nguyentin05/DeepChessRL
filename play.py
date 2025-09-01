import argparse
import sys
import chess


from config import (
    ENGINE_PATH, ENGINE_THREADS, ENGINE_HASH_MB, ENGINE_LIMIT_STRENGTH,
    ENGINE_ELO, ENGINE_MOVETIME_S,
    REWARD_WIN, REWARD_LOSS, REWARD_DRAW, REWARD_INTERMEDIATE,
)
from env import ChessEnv
from utils import parse_user_move, list_legal_moves_san
from agents.stockfish_agent import StockfishAgent, StockfishConfig

HELP_TXT = """
all the commands:
help : show help table
board : print the broad
moves : list all the legal moves
fen : show FEN
resign : You have surrendered
hint : show best move
Only SAN's type (e4, Nf3, O-O, Bxe6+) or UCI's type (e2e4, g1f3).
"""

def parse_args():
    p = argparse.ArgumentParser(description="Play human vs StockfishAgent on a Gym-like ChessEnv")
    p.add_argument("--engine", default=ENGINE_PATH, help="Path to stockfish executable or in PATH")
    p.add_argument("--threads", type=int, default=ENGINE_THREADS)
    p.add_argument("--hash", type=int, default=ENGINE_HASH_MB)
    p.add_argument("--limit-strength", action="store_true" if ENGINE_LIMIT_STRENGTH else "store_false",
                                help="Enable UCI_LimitStrength to cap engine ELO")
    p.add_argument("--elo", type=int, default=ENGINE_ELO, help="Engine ELO (800–2800) when limit-strength on")
    p.add_argument("--movetime", type=float, default=ENGINE_MOVETIME_S, help="Seconds think time per move")
    p.add_argument("--human-white", action="store_true", help="Human plays White (default: human plays Black)")
    return p.parse_args()

def main():
    args = parse_args()

    agent_color = chess.BLACK if args.human_white else chess.WHITE
    env = ChessEnv(
        agent_color=agent_color,
        reward_win=REWARD_WIN,
        reward_loss=REWARD_LOSS,
        reward_draw=REWARD_DRAW,
        reward_intermediate=REWARD_INTERMEDIATE,
    )

    cfg = StockfishConfig(
        engine_path="F:\stockfish\stockfish-windows-x86-64-avx2.exe",
        threads=args.threads,
        hash_mb=args.hash,
        limit_strength=args.limit_strength,
        elo=args.elo,
        movetime_s=args.movetime,
    )
    try:
        agent = StockfishAgent(cfg)
    except FileNotFoundError:
        print("❌Stockfish not found. Use --engine to point to .exe/.bin file or add to PATH.")
        sys.exit(1)

    obs = env.reset()
    print("the game is started. You're ", "White" if args.human_white else "Black")
    print(HELP_TXT)
    env.render()

    #If human plays Black, agent (White) moves first
    try:
        if not args.human_white:
            mv = agent.select_move(env.board)
            san = env.board.san(mv)
            obs, r, done, info = env.step(mv)
            print(f"Stockfish đi: {san} (UCI: {mv.uci()})")
            env.render()
            if done:
                print(_result_line(info))
                return

        while True:
            if env.board.is_game_over():
                info = _ensure_terminal_info(env)
                print(_result_line(info))
                break

            #Human turn
            s = input("Your turn (Or 'help'): ").strip()
            if not s:
                continue
            cmd = s.lower()
            if cmd == "help":
                print(HELP_TXT)
                continue
            if cmd == "board":
                env.render()
                continue
            if cmd == "moves":
                print("legal moves:", ", ".join(list_legal_moves_san(env.board)))
                continue
            if cmd == "fen":
                print("FEN:", env.board.fen())
                continue
            if cmd == "resign":
                #Human resignation ⇒ from env.agent_color perspective, that's a win if agent != human
                winner = env.board.turn # the side to move *before* applying resignation; not essential
                #We just print a message and exit politely
                print("You surrender. Stockfish wins.")
                break
            if cmd == "hint":
                mv_hint = agent.analyse_hint(env.board, time_limit=min(1.5, args.movetime * 2))
                if mv_hint:
                    try:
                        print("Hint:", env.board.san(mv_hint), f"(UCI: {mv_hint.uci()})")
                    except Exception:
                        print("Hint UCI:", mv_hint.uci())
                else:
                    print("No hints were given.")
                continue

            mv = parse_user_move(env.board, s)
            if mv is None:
                print("Illegal move. Type 'moves' to see legal moves, or 'help' for assistance.")
                continue

            san_h = env.board.san(mv)
            obs, r, done, info = env.step(mv)
            print(f"Yours: {san_h} (UCI: {mv.uci()})")
            if done:
                print(_result_line(info))
                break

            #Agent turn
            mv_a = agent.select_move(env.board)
            san_a = env.board.san(mv_a)
            obs, r, done, info = env.step(mv_a)
            print(f"Stockfish: {san_a} (UCI: {mv_a.uci()})")
            env.render()
            if done:
                print(_result_line(info))
                break
    finally:
        agent.close()

def _ensure_terminal_info(env: ChessEnv):
    #Helper to build a terminal-like info if already game over
    oc = env.board.outcome()
    if oc is None:
        return {"done": False}
    res = env.board.result()
    term = oc.termination.name if oc.termination else "UNKNOWN"
    if oc.winner is True:
        who = "White won"
    elif oc.winner is False:
        who = "Black won"
    else:
        who = "Slalement"
    return {"done": True, "result": res, "who": who, "termination": term}

def _result_line(info):
    if not info.get("done"):
        return "The game is not over."
    return f"Result: {info.get('result')} — {info.get('who')} ({info.get('termination')})."

if __name__ == "__main__":
    main()