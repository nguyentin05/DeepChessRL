# -*- coding: utf-8 -*-
import argparse
import pygame
import sys
import time
import chess
from agents.bc_agent import BCAgent, BCConfig
from agents.factory import create_agent, available_agents
from env import ChessEnv
from agents.stockfish_agent import StockfishAgent, StockfishConfig


def _parse_weights(s: str | None) -> dict[str, float] | None:
    if not s:
        return None
    out = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            pass
    return out or None

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="heuristic", choices=available_agents(),
                   help="Chọn agent: " + ", ".join(available_agents()))
    p.add_argument("--ai-color", default="black", choices=["white", "black"],
                   help="AI đánh quân nào (mặc định: black)")
    p.add_argument("--depth", type=int, default=2, help="Độ sâu tìm kiếm (Heuristic)")
    p.add_argument("--seed", type=int, default=None, help="Seed cho random tiebreak")
    p.add_argument("--weights", type=str, default=None,
                   help='Trọng số heuristic, ví dụ: "material=1.0,mobility=0.12,center=0.05,king_safety=0.12,pawn_structure=0.05,tempo=0.02"')
    p.add_argument("--ckpt", type=str, default=None, help="Đường dẫn checkpoint .pt của BC")
    p.add_argument("--bc-in-ch", type=int, default=14)
    p.add_argument("--bc-action-dim", type=int, default=4864)
    p.add_argument("--bc-temp", type=float, default=1.0)
    p.add_argument("--bc-topk", type=int, default=None)
    return p.parse_args()
# ====== Cấu hình giao diện ======
WIDTH, HEIGHT = 520, 520              # tăng nhẹ để có viền cho overlay
BOARD_SIZE = 480
MARGIN = (WIDTH - BOARD_SIZE) // 2    # canh giữa bảng
SQ_SIZE = BOARD_SIZE // 8
FPS = 60

LIGHT = pygame.Color("#EEEED2")
DARK  = pygame.Color("#769656")
HL_SELECTED = pygame.Color(255, 215, 0, 120)   # vàng đậm
HL_MOVE_TO  = pygame.Color(0, 0, 0, 60)        # đen trong suốt
HL_LASTMOVE = pygame.Color(255, 255, 0, 80)

ANIM_MS = 180  # thời gian animation 1 nước (ms)

# ====== Load ảnh ======
def load_images():
    pieces = {}
    names = ["P","R","N","B","Q","K"]
    for n in names:
        pieces["w"+n] = pygame.image.load(f"assets/pieces/w{n}.png").convert_alpha()
        pieces["b"+n] = pygame.image.load(f"assets/pieces/b{n}.png").convert_alpha()
    # scale sẵn
    for k,v in pieces.items():
        pieces[k] = pygame.transform.smoothscale(v, (SQ_SIZE, SQ_SIZE))
    return pieces

# ====== Vẽ bàn, quân, highlight ======
def draw_board(screen, last_move=None):
    screen.fill(pygame.Color("black"))
    # last move highlight
    if last_move:
        fx, fy = chess.square_file(last_move.from_square), chess.square_rank(last_move.from_square)
        tx, ty = chess.square_file(last_move.to_square),   chess.square_rank(last_move.to_square)
        for (x, y) in [(fx, fy), (tx, ty)]:
            rx = MARGIN + x * SQ_SIZE
            ry = MARGIN + (7 - y) * SQ_SIZE
            pygame.draw.rect(screen, HL_LASTMOVE, (rx, ry, SQ_SIZE, SQ_SIZE))

    for r in range(8):
        for c in range(8):
            color = LIGHT if (r + c) % 2 == 0 else DARK
            rx = MARGIN + c * SQ_SIZE
            ry = MARGIN + r * SQ_SIZE
            pygame.draw.rect(screen, color, (rx, ry, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board, images, skip_square=None):
    for r in range(8):
        for c in range(8):
            sq = chess.square(c, 7 - r)
            if skip_square is not None and sq == skip_square:
                continue
            piece = board.piece_at(sq)
            if piece:
                key = ("w" if piece.color == chess.WHITE else "b") + piece.symbol().upper()
                screen.blit(images[key], (MARGIN + c * SQ_SIZE, MARGIN + r * SQ_SIZE))

def draw_selection_highlight(screen, selected_sq, legal_targets):
    if selected_sq is not None:
        sx, sy = chess.square_file(selected_sq), chess.square_rank(selected_sq)
        rx = MARGIN + sx * SQ_SIZE
        ry = MARGIN + (7 - sy) * SQ_SIZE
        srf = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        srf.fill(HL_SELECTED)
        screen.blit(srf, (rx, ry))
    if legal_targets:
        dot = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(dot, HL_MOVE_TO, (SQ_SIZE//2, SQ_SIZE//2), SQ_SIZE//6)
        for t in legal_targets:
            tx, ty = chess.square_file(t), chess.square_rank(t)
            screen.blit(dot, (MARGIN + tx*SQ_SIZE, MARGIN + (7-ty)*SQ_SIZE))

# ====== Animation di chuyển ======
def animate_move(screen, board_before, move, board_after, images, clock):
    start = time.time()
    ms = ANIM_MS / 1000.0
    fx, fy = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    tx, ty = chess.square_file(move.to_square),   chess.square_rank(move.to_square)

    piece = board_before.piece_at(move.from_square)
    if not piece:
        return  # không có gì để animate

    key = ("w" if piece.color == chess.WHITE else "b") + piece.symbol().upper()
    sprite = images[key]

    # Render nền: vẽ board_after nhưng bỏ quân đích để thấy “bay”
    while True:
        t = time.time() - start
        alpha = min(1.0, t / ms)

        draw_board(screen, last_move=move)
        # vẽ nền là board_before trừ from, và vẽ các quân còn lại từ board_before + move ăn
        # cách đơn giản: vẽ board_after rồi “skip” ô đến (để quân đang bay che)
        draw_pieces(screen, board_after, images, skip_square=move.to_square)

        cx = (1 - alpha) * (MARGIN + fx*SQ_SIZE) + alpha * (MARGIN + tx*SQ_SIZE)
        cy = (1 - alpha) * (MARGIN + (7 - fy)*SQ_SIZE) + alpha * (MARGIN + (7 - ty)*SQ_SIZE)
        screen.blit(sprite, (cx, cy))

        pygame.display.flip()
        clock.tick(FPS)
        if alpha >= 1.0:
            break

# ====== Hộp chọn phong cấp Tốt ======
PROMO_ORDER = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
PROMO_LABEL = {chess.QUEEN: "Q", chess.ROOK: "R", chess.BISHOP: "B", chess.KNIGHT: "N"}

def choose_promotion(screen, color_is_white, images):
    # tạo overlay đơn giản 4 ô
    box_w, box_h = SQ_SIZE*4 + 12, SQ_SIZE + 12
    box_x = (WIDTH - box_w) // 2
    box_y = (HEIGHT - box_h) // 2
    panel = pygame.Surface((box_w, box_h))
    panel.fill(pygame.Color(30, 30, 30))

    for i, p in enumerate(PROMO_ORDER):
        key = ("w" if color_is_white else "b") + {chess.QUEEN:"Q", chess.ROOK:"R", chess.BISHOP:"B", chess.KNIGHT:"N"}[p]
        img = images[key]
        panel.blit(img, (6 + i*SQ_SIZE, 6))

    font = pygame.font.SysFont(None, 20)
    txt = font.render("Chọn phong cấp (Q/R/B/N)", True, pygame.Color("white"))
    panel.blit(txt, (6, SQ_SIZE + 6) if False else (6, 6))  # (đã hiển thị bằng quân, text chỉ minh hoạ)

    # loop chờ click
    while True:
        # vẽ mờ nền
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 128))
        screen.blit(dim, (0, 0))
        screen.blit(panel, (box_x, box_y))
        pygame.display.flip()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            if e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = e.pos
                if box_x <= mx <= box_x + box_w and box_y <= my <= box_y + box_h:
                    # vị trí trong box
                    rel_x = mx - box_x
                    choice = int((rel_x - 6) // SQ_SIZE)
                    if 0 <= choice < 4:
                        return PROMO_ORDER[choice]

# ====== Hỗ trợ xác định các đích hợp lệ từ một ô (kể cả promotion) ======
def legal_targets_for(board, from_sq):
    targets = set()
    for mv in board.legal_moves:
        if mv.from_square == from_sq:
            targets.add(mv.to_square)
    return targets

def is_pawn_promotion(board, move):
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN:
        return False
    to_rank = chess.square_rank(move.to_square)
    return (piece.color == chess.WHITE and to_rank == 7) or (piece.color == chess.BLACK and to_rank == 0)

# ================== MAIN ==================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DeepChessRL - PyGame GUI")
    clock = pygame.time.Clock()
    images = load_images()

    # === ADD: đọc tham số CLI ===
    args = _parse_args()
    weights = _parse_weights(args.weights)
    ai_color = chess.WHITE if args.ai_color == "white" else chess.BLACK
    human_color = chess.BLACK if ai_color == chess.WHITE else chess.WHITE

    # === MODIFY: tạo env theo màu AI ===
    env = ChessEnv(agent_color=ai_color)  # AI color theo tham số

    # === MODIFY: khởi tạo agent (stockfish hoặc factory) ===
    if args.agent == "stockfish":
        cfg = StockfishConfig(
            engine_path=r"F:\Engines\stockfish\stockfish-windows-x86-64-avx2.exe",
            threads=2, hash_mb=256, limit_strength=True, elo=1200, movetime_s=0.7
        )
        agent = StockfishAgent(cfg)

    elif args.agent == "heuristic":
        agent = create_agent("heuristic", depth=args.depth, seed=args.seed, weights=weights)

    elif args.agent == "bc":
        # === CREATE FROM CHECKPOINT ===
        agent = create_agent(
            "bc",
            in_channels=args.bc_in_ch,
            action_dim=args.bc_action_dim,
            device="cpu",  # đổi "cuda" nếu bạn muốn & có GPU
            temperature=args.bc_temp,
            topk=args.bc_topk,
            checkpoint=args.ckpt,  # <— CHỖ TRUYỀN CHECKPOINT
        )

    else:
        agent = create_agent(args.agent)

    selected_sq = None
    legal_targets = set()
    last_move = None
    ai_thinking = False  # chặn AI đi 2 lần trong 1 frame

    running = True
    while running:
        # --- RENDER ---
        draw_board(screen, last_move=last_move)
        draw_pieces(screen, env.board, images)
        draw_selection_highlight(screen, selected_sq, legal_targets)
        pygame.display.flip()

        # --- EVENTS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                # bỏ click ngoài bàn
                if not (MARGIN <= x < MARGIN + BOARD_SIZE and MARGIN <= y < MARGIN + BOARD_SIZE):
                    selected_sq = None
                    legal_targets = set()
                    continue

                col = (x - MARGIN) // SQ_SIZE
                row = (y - MARGIN) // SQ_SIZE
                square = chess.square(col, 7 - row)

                # chọn quân (CẬP NHẬT: người chơi = màu đối nghịch với AI)
                if selected_sq is None:
                    piece = env.board.piece_at(square)
                    if piece and piece.color == human_color:
                        selected_sq = square
                        legal_targets = legal_targets_for(env.board, selected_sq)
                else:
                    # cố gắng đi
                    mv = chess.Move(selected_sq, square)

                    if square in legal_targets:
                        # promotion?
                        promo = None
                        if is_pawn_promotion(env.board, mv):
                            promo = choose_promotion(
                                screen,
                                color_is_white=(human_color == chess.WHITE),
                                images=images
                            )
                            mv = chess.Move(selected_sq, square, promotion=promo)

                        # Xác thực nước hợp lệ (bao gồm promotion)
                        if mv not in env.board.legal_moves:
                            # nếu user chọn đích đúng nhưng promotion chưa khớp,
                            # cố match với nước hợp lệ có cùng from/to
                            for lm in env.board.legal_moves:
                                if lm.from_square == selected_sq and lm.to_square == square:
                                    mv = lm
                                    break

                        if mv in env.board.legal_moves:
                            prev = env.board.copy()
                            env.board.push(mv)  # dùng board trực tiếp để animate mượt
                            after = env.board.copy()
                            animate_move(screen, prev, mv, after, images, clock)
                            last_move = mv

                            selected_sq = None
                            legal_targets = set()
                        else:
                            # đích hợp lệ nhưng thất bại (hiếm) -> reset chọn
                            selected_sq = None
                            legal_targets = set()
                    else:
                        # click vào quân khác cùng màu -> đổi selection
                        piece = env.board.piece_at(square)
                        if piece and piece.color == human_color:
                            selected_sq = square
                            legal_targets = legal_targets_for(env.board, selected_sq)
                        else:
                            selected_sq = None
                            legal_targets = set()

        # --- TURN OF AI (tự động) ---
        if not env.board.is_game_over():
            if env.board.turn == ai_color and not ai_thinking:
                ai_thinking = True
                try:
                    mv_a = agent.select_move(env.board)
                    if mv_a not in env.board.legal_moves:
                        raise ValueError("Agent returned an illegal move.")
                    prev = env.board.copy()
                    env.board.push(mv_a)
                    after = env.board.copy()
                    animate_move(screen, prev, mv_a, after, images, clock)
                    last_move = mv_a
                except Exception as e:
                    print(f"[AI ERROR] {e}")
                finally:
                    ai_thinking = False
        else:
            print(env.board.outcome())

        clock.tick(FPS)

    pygame.quit()
    # an toàn: chỉ close nếu agent có phương thức này (Stockfish)
    if hasattr(agent, "close"):
        try:
            agent.close()
        except Exception:
            pass
    sys.exit()


if __name__ == "__main__":
    main()
