# -*- coding: utf-8 -*-
import pygame
import sys
import chess

from env import ChessEnv
from agents.stockfish_agent import StockfishAgent, StockfishConfig
from utils import parse_user_move

# Kích thước bàn cờ
WIDTH, HEIGHT = 480, 480
SQ_SIZE = WIDTH // 8

# Load ảnh quân cờ
def load_images():
    pieces = {}
    names = ["P", "R", "N", "B", "Q", "K"]
    for n in names:
        pieces["w"+n] = pygame.image.load(f"assets/pieces/w{n}.png")
        pieces["b"+n] = pygame.image.load(f"assets/pieces/b{n}.png")
    return pieces

def draw_board(screen):
    colors = [pygame.Color("#EEEED2"), pygame.Color("#769656")]
    for r in range(8):
        for c in range(8):
            color = colors[(r+c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board, images):
    for r in range(8):
        for c in range(8):
            sq = chess.square(c, 7-r)  # cột, hàng -> square index
            piece = board.piece_at(sq)
            if piece:
                key = ("w" if piece.color == chess.WHITE else "b") + piece.symbol().upper()
                img = pygame.transform.scale(images[key], (SQ_SIZE, SQ_SIZE))
                screen.blit(img, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DeepChessRL - PyGame GUI")
    clock = pygame.time.Clock()
    images = load_images()

    # Env + Agent
    env = ChessEnv(agent_color=chess.BLACK)  # human đi Trắng

    cfg = StockfishConfig(
        engine_path="F:\stockfish\stockfish-windows-x86-64-avx2.exe",
        # 👈 thay đường dẫn thật trên máy bạn
        threads=2,
        hash_mb=256,
        limit_strength=True,
        elo=1200,
        movetime_s=0.7,
    )
    agent = StockfishAgent(cfg)

    selected_square = None
    running = True
    while running:
        draw_board(screen)
        draw_pieces(screen, env.board, images)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = x // SQ_SIZE
                row = y // SQ_SIZE
                square = chess.square(col, 7-row)

                if selected_square is None:
                    piece = env.board.piece_at(square)
                    if piece and piece.color == chess.WHITE:  # human đi Trắng
                        selected_square = square
                else:
                    move = chess.Move(selected_square, square)
                    if move in env.board.legal_moves:
                        env.step(move)
                        selected_square = None

                        # Kiểm tra game over
                        if env.board.is_game_over():
                            print(env.board.outcome())
                            continue

                        # Agent đi
                        mv_a = agent.select_move(env.board)
                        env.step(mv_a)

                        if env.board.is_game_over():
                            print(env.board.outcome())
                    else:
                        selected_square = None

        clock.tick(30)

    pygame.quit()
    agent.close()
    sys.exit()

if __name__ == "__main__":
    main()
