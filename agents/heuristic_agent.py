from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import random
import chess
from .base_agent import BaseAgent

@dataclass
class HeuristicWeights:
    material: float = 1.0
    mobility: float = 0.10
    center: float = 0.05
    pawn_structure: float = 0.05
    king_safety: float = 0.10
    tempo: float = 0.02  # thưởng nhẹ cho bên tới lượt


class HeuristicAgent(BaseAgent):
    """
    Agent chọn nước dựa trên heuristic + minimax alpha-beta (tuỳ depth).
    - If depth == 1: greedy 1-ply (đánh giá sau một nước).
    - If depth >= 2: minimax alpha-beta từ quan điểm người chơi ở root.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        depth: int = 2,
        random_tiebreak: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.weights = self._build_weights(weights)
        self.depth = max(1, int(depth))
        self.random_tiebreak = random_tiebreak
        if seed is not None:
            random.seed(seed)

        # MVV-LVA xấp xỉ cho move ordering (giá trị quân cơ bản)
        self._piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000,
        }

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def select_move(self, board: chess.Board) -> chess.Move:
        """Chọn nước đi tốt nhất theo heuristic/minimax, từ POV của bên đang tới lượt."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            # Không còn nước đi (checkmate hoặc stalemate) — để GUI xử lý kết thúc ván.
            raise RuntimeError("No legal moves available; the game is over.")

        root_color = board.turn

        # Độ sâu 1: greedy — duyệt các nước, đánh giá một ply
        if self.depth == 1:
            best_score = -math.inf
            best_moves: List[chess.Move] = []
            for mv in self._ordered_moves(board, legal_moves):
                board.push(mv)
                score = self._evaluate_for_root(board, root_color)
                board.pop()
                if score > best_score:
                    best_score = score
                    best_moves = [mv]
                elif score == best_score:
                    best_moves.append(mv)
            return random.choice(best_moves) if self.random_tiebreak else best_moves[0]

        # depth >= 2: minimax alpha-beta
        alpha, beta = -math.inf, math.inf
        best_score = -math.inf
        best_moves = []
        for mv in self._ordered_moves(board, legal_moves):
            board.push(mv)
            score = self._minimax(board, self.depth - 1, alpha, beta, maximizing=False, root_color=root_color)
            board.pop()
            if score > best_score:
                best_score = score
                best_moves = [mv]
            elif score == best_score:
                best_moves.append(mv)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break

        return random.choice(best_moves) if (self.random_tiebreak and best_moves) else best_moves[0]

    def eval_mode(self) -> None:
        pass

    def train_mode(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    # --------------------------------------------------------------------- #
    # Core search
    # --------------------------------------------------------------------- #

    def _minimax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        root_color: chess.Color,
    ) -> float:
        # kết thúc: hết depth hoặc ván kết thúc
        if depth == 0 or board.is_game_over():
            return self._evaluate_for_root(board, root_color)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self._evaluate_for_root(board, root_color)

        if maximizing:
            value = -math.inf
            for mv in self._ordered_moves(board, legal_moves):
                board.push(mv)
                value = max(value, self._minimax(board, depth - 1, alpha, beta, False, root_color))
                board.pop()
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = math.inf
            for mv in self._ordered_moves(board, legal_moves):
                board.push(mv)
                value = min(value, self._minimax(board, depth - 1, alpha, beta, True, root_color))
                board.pop()
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def _ordered_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """Ưu tiên capture/check/promotion (move ordering) để tăng hiệu quả alpha-beta."""
        def score_move(mv: chess.Move) -> int:
            s = 0
            if board.is_capture(mv):
                victim = self._captured_piece_type(board, mv)
                attacker = board.piece_type_at(mv.from_square)
                s += 1000
                if victim:
                    s += 10 * self._piece_values.get(victim, 0)
                if attacker:
                    s -= self._piece_values.get(attacker, 0)
            if board.gives_check(mv):
                s += 50
            if mv.promotion:
                s += self._piece_values.get(mv.promotion, 0)
            return s

        # Sắp xếp giảm dần theo "độ hứa hẹn"
        return sorted(moves, key=score_move, reverse=True)

    # --------------------------------------------------------------------- #
    # Evaluation
    # --------------------------------------------------------------------- #

    def _evaluate_for_root(self, board: chess.Board, root_color: chess.Color) -> float:
        """
        Tính điểm từ quan điểm 'root_color'.
        Dương: lợi cho root_color. Âm: bất lợi.
        """
        # Tính lợi thế (White - Black) cho từng đặc trưng:
        mat = self._material_score(board)                       # >0 nếu White hơn chất
        mob = self._mobility_score(board)                       # >0 nếu White cơ động hơn
        ctr = self._center_control_score(board)                 # >0 nếu White kiểm soát trung tâm hơn
        pst = self._pawn_structure_score(board)                 # >0 nếu cấu trúc tốt nghiêng White
        ksf = self._king_safety_score(board)                    # >0 nếu White vua an toàn hơn
        tmp = self._tempo_bonus(board)                          # >0 nếu bên tới lượt là White

        total_white_adv = (
            self.weights.material * mat
            + self.weights.mobility * mob
            + self.weights.center * ctr
            + self.weights.pawn_structure * pst
            + self.weights.king_safety * ksf
            + self.weights.tempo * tmp
        )

        # Chuyển về POV root_color
        return total_white_adv if root_color == chess.WHITE else -total_white_adv

    def _material_score(self, board: chess.Board) -> float:
        """White material − Black material (đơn vị "centipawn" quy ước)."""
        score = 0
        for sq, piece in board.piece_map().items():
            val = self._piece_values.get(piece.piece_type, 0)
            score += val if piece.color == chess.WHITE else -val
        return float(score) / 100.0  # chuẩn hoá nhẹ về 'pawn units'

    def _mobility_score(self, board: chess.Board) -> float:
        """Số nước hợp lệ (White − Black)."""
        orig_turn = board.turn

        board.turn = chess.WHITE
        white_moves = board.legal_moves.count()

        board.turn = chess.BLACK
        black_moves = board.legal_moves.count()

        board.turn = orig_turn
        return float(white_moves - black_moves) / 10.0  # scale vừa phải

    def _center_control_score(self, board: chess.Board) -> float:
        """Số quân tấn công các ô trung tâm (White − Black)."""
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        w = sum(len(board.attackers(chess.WHITE, sq)) for sq in center_squares)
        b = sum(len(board.attackers(chess.BLACK, sq)) for sq in center_squares)
        return float(w - b) / 5.0

    def _pawn_structure_score(self, board: chess.Board) -> float:
        """
        Đánh giá đơn giản:
        +1 mỗi passed pawn, -0.5 mỗi isolated, -0.5 mỗi doubled — lấy (White − Black).
        """
        wp = list(board.pieces(chess.PAWN, chess.WHITE))
        bp = list(board.pieces(chess.PAWN, chess.BLACK))

        doubled_w = self._count_doubled_pawns(wp)
        doubled_b = self._count_doubled_pawns(bp)

        isolated_w = self._count_isolated_pawns(board, wp, chess.WHITE)
        isolated_b = self._count_isolated_pawns(board, bp, chess.BLACK)

        passed_w = self._count_passed_pawns(board, wp, chess.WHITE)
        passed_b = self._count_passed_pawns(board, bp, chess.BLACK)

        w_score = passed_w - 0.5 * isolated_w - 0.5 * doubled_w
        b_score = passed_b - 0.5 * isolated_b - 0.5 * doubled_b
        return float(w_score - b_score)

    def _king_safety_score(self, board: chess.Board) -> float:
        """
        King shield: số tốt che chắn ngay phía trước vua (3 ô: file-1, file, file+1 ở rank trước mặt).
        """
        w_shield = self._king_shield(board, chess.WHITE)
        b_shield = self._king_shield(board, chess.BLACK)
        # nhiều tốt che chắn hơn → an toàn hơn
        return float(w_shield - b_shield) / 3.0

    def _tempo_bonus(self, board: chess.Board) -> float:
        """White tới lượt → +1; Black tới lượt → -1 (sau đó nhân trọng số nhỏ)."""
        return 1.0 if board.turn == chess.WHITE else -1.0

    # --------------------------------------------------------------------- #
    # Pawn helpers
    # --------------------------------------------------------------------- #

    def _count_doubled_pawns(self, pawns: List[chess.Square]) -> int:
        files = [0] * 8
        for sq in pawns:
            files[chess.square_file(sq)] += 1
        return sum(max(c - 1, 0) for c in files)

    def _count_isolated_pawns(self, board: chess.Board, pawns: List[chess.Square], color: chess.Color) -> int:
        pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
        cnt = 0
        for sq in pawns:
            f = chess.square_file(sq)
            left_has = (f - 1) in pawn_files
            right_has = (f + 1) in pawn_files
            if not left_has and not right_has:
                cnt += 1
        return cnt

    def _count_passed_pawns(self, board: chess.Board, pawns: List[chess.Square], color: chess.Color) -> int:
        opp = chess.BLACK if color == chess.WHITE else chess.WHITE
        opp_pawns = list(board.pieces(chess.PAWN, opp))
        opp_by_file = {}
        for sq in opp_pawns:
            opp_by_file.setdefault(chess.square_file(sq), []).append(sq)

        cnt = 0
        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            # Phía trước theo hướng tiến
            if color == chess.WHITE:
                ahead_ranks = range(r + 1, 8)
            else:
                ahead_ranks = range(r - 1, -1, -1)
            blocked = False
            for cf in (f - 1, f, f + 1):
                if 0 <= cf <= 7:
                    for osq in opp_by_file.get(cf, []):
                        orank = chess.square_rank(osq)
                        if (color == chess.WHITE and orank in ahead_ranks) or (color == chess.BLACK and orank in ahead_ranks):
                            blocked = True
                            break
                if blocked:
                    break
            if not blocked:
                cnt += 1
        return cnt

    def _king_shield(self, board: chess.Board, color: chess.Color) -> int:
        ksq = board.king(color)
        if ksq is None:
            # vị thế bất hợp lệ (vua bị bắt) — coi như rất tệ
            return 0
        rank = chess.square_rank(ksq)
        file = chess.square_file(ksq)
        shield_rank = rank + (1 if color == chess.WHITE else -1)
        if not (0 <= shield_rank <= 7):
            return 0
        shield_files = [f for f in (file - 1, file, file + 1) if 0 <= f <= 7]
        cnt = 0
        for f in shield_files:
            sq = chess.square(f, shield_rank)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                cnt += 1
        return cnt

    # --------------------------------------------------------------------- #
    # Utils
    # --------------------------------------------------------------------- #

    def _build_weights(self, weights: Optional[Dict[str, float]]) -> HeuristicWeights:
        if not weights:
            return HeuristicWeights()
        base = HeuristicWeights()
        for k, v in weights.items():
            if hasattr(base, k):
                setattr(base, k, float(v))
        return base

    def _captured_piece_type(self, board: chess.Board, mv: chess.Move) -> Optional[chess.PieceType]:
        """Xác định loại quân bị bắt (xấp xỉ; en passant coi như bắt tốt)."""
        if board.is_en_passant(mv):
            return chess.PAWN
        return board.piece_type_at(mv.to_square)
