# -*- coding: utf-8 -*-
import pygame
import sys
import time
import chess

from env import ChessEnv
from agents.stockfish_agent import StockfishAgent, StockfishConfig

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

    env = ChessEnv(agent_color=chess.BLACK)  # người (White), agent (Black)
    cfg = StockfishConfig(
        engine_path="stockfish",  # đổi sang path tuyệt đối nếu chưa thêm PATH
        threads=2, hash_mb=256, limit_strength=True, elo=1200, movetime_s=0.7
    )
    agent = StockfishAgent(cfg)

    selected_sq = None
    legal_targets = set()
    last_move = None

    running = True
    while running:
        draw_board(screen, last_move=last_move)
        draw_pieces(screen, env.board, images)
        draw_selection_highlight(screen, selected_sq, legal_targets)
        pygame.display.flip()

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

                # chọn quân
                if selected_sq is None:
                    piece = env.board.piece_at(square)
                    if piece and piece.color == chess.WHITE:
                        selected_sq = square
                        legal_targets = legal_targets_for(env.board, selected_sq)
                else:
                    # cố gắng đi
                    mv = chess.Move(selected_sq, square)

                    if square in legal_targets:
                        # promotion?
                        if is_pawn_promotion(env.board, mv):
                            promo = choose_promotion(screen, color_is_white=True, images=images)
                            mv = chess.Move(selected_sq, square, promotion=promo)

                        # Xác thực nước nằm trong legal_moves (bao gồm promotion)
                        if mv not in env.board.legal_moves:
                            # nếu user chọn đích đúng nhưng promotion chưa khớp,
                            # cố match với nước hợp lệ có cùng from/to
                            for lm in env.board.legal_moves:
                                if lm.from_square == selected_sq and lm.to_square == square:
                                    mv = lm; break

                        if mv in env.board.legal_moves:
                            prev = env.board.copy()
                            env.board.push(mv)  # dùng board trực tiếp để animate mượt
                            after = env.board.copy()
                            animate_move(screen, prev, mv, after, images, clock)
                            last_move = mv

                            # xong lượt người → kiểm tra kết thúc
                            if env.board.is_game_over():
                                print(env.board.outcome()); selected_sq=None; legal_targets=set(); continue

                            # lượt agent
                            mv_a = agent.select_move(env.board)
                            prev = env.board.copy()
                            env.board.push(mv_a)
                            after = env.board.copy()
                            animate_move(screen, prev, mv_a, after, images, clock)
                            last_move = mv_a

                            selected_sq = None
                            legal_targets = set()

                            if env.board.is_game_over():
                                print(env.board.outcome())
                        else:
                            # đích hợp lệ nhưng thất bại (hiếm) -> reset chọn
                            selected_sq = None
                            legal_targets = set()
                    else:
                        # click vào quân khác cùng màu -> đổi selection
                        piece = env.board.piece_at(square)
                        if piece and piece.color == chess.WHITE:
                            selected_sq = square
                            legal_targets = legal_targets_for(env.board, selected_sq)
                        else:
                            selected_sq = None
                            legal_targets = set()

        clock.tick(FPS)

    pygame.quit()
    agent.close()
    sys.exit()

if __name__ == "__main__":
    main()
