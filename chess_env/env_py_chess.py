# chess_env/env_py_chess.py
import numpy as np
import chess
from chess_env.encode import encode_board, legal_mask, index_to_chess_move
from chess_env import rewards as R
from agents.heuristic_agent import HeuristicAgent

class PyChessEnv:
    """
    Gym-like wrapper:
      - reset(agent_is_white: bool) -> state, mask
      - step_agent(action_index)    -> (next_state, next_mask, reward, done, info)
    Vòng step: Agent đi 1 nước -> (nếu chưa xong) Heuristic đi 1 nước -> trả reward cho Agent.
    Đảm bảo khi trả về khỏi step(), lượt sẽ quay lại cho agent cho vòng tiếp theo (nếu chưa done).
    """
    def __init__(self, max_halfmoves: int = 300):
        self.board = chess.Board()
        self.max_halfmoves = max_halfmoves
        self.halfmove_count = 0
        self.agent_color = chess.WHITE
        self.heuristic = HeuristicAgent()

        # stats cho logging
        self.agent_checks = 0
        self.agent_mates  = 0
        self.agent_moves  = 0

    @property
    def observation_space(self):
        class Obs: shape = (832,)
        return Obs()

    @property
    def action_space(self):
        class Act: n = 64 * 64 * 5
        return Act()

    def reset(self, agent_is_white: bool = True):
        self.board.reset()
        self.board.clear_stack()
        self.halfmove_count = 0
        self.agent_color = chess.WHITE if agent_is_white else chess.BLACK
        self.agent_checks = 0
        self.agent_mates = 0
        self.agent_moves = 0

        # Nếu agent chơi Đen, để heuristic đi trước đúng 1 nước (không cộng reward ở reset)
        if self.board.turn != self.agent_color and not self.board.is_game_over():
            self._play_heuristic_once(apply_reward=False)

        return encode_board(self.board), legal_mask(self.board)

    def _is_done(self) -> bool:
        if self.board.is_game_over():
            return True
        if self.halfmove_count >= self.max_halfmoves:
            return True
        return False

    def _reward_delta_after_agent(self, before_board: chess.Board, after_board: chess.Board, gave_check: bool):
        rew = 0.0
        rew += R.MOVE_PENALTY
        rew += R.material_delta(before_board, after_board, self.agent_color)
        if gave_check:
            rew += R.CHECK_BONUS
        if after_board.is_game_over():
            outcome = after_board.outcome()
            if outcome is not None and outcome.termination in (chess.Termination.CHECKMATE,):
                if outcome.winner == self.agent_color:
                    rew += R.MATE_WIN
                    self.agent_mates += 1
                else:
                    rew += R.MATE_LOSE
            else:
                # stalemate / repetition / 50-move ... (tuỳ chỉnh thêm nếu muốn)
                pass
        return rew

    def _reward_delta_after_opp(self, before_board: chess.Board, after_board: chess.Board, opp_gave_check: bool):
        rew = 0.0
        if opp_gave_check:
            rew += R.CHECK_PENALTY
        if after_board.is_game_over():
            outcome = after_board.outcome()
            if outcome is not None and outcome.termination in (chess.Termination.CHECKMATE,):
                if outcome.winner == self.agent_color:
                    rew += R.MATE_WIN
                    self.agent_mates += 1
                else:
                    rew += R.MATE_LOSE
        return rew

    def _heuristic_select_move(self) -> chess.Move | None:
        best = None
        best_score = -1e18
        for mv in self.board.legal_moves:
            sc = self.heuristic.score_move(self.board, mv)
            if sc > best_score:
                best_score = sc
                best = mv
        return best

    def _play_heuristic_once(self, apply_reward: bool) -> float:
        """Cho đối thủ đi đúng 1 nước nếu đang là lượt của đối thủ.
        Trả về reward (nếu apply_reward=True) tương ứng với nước đi của đối thủ (thường gọi trong step())."""
        if self._is_done():
            return 0.0
        if self.board.turn == self.agent_color:
            return 0.0

        before = self.board.copy(stack=False)
        mv = self._heuristic_select_move()
        if mv is None:
            # Không còn nước đi hợp lệ
            return 0.0
        opp_gives_check = self.board.gives_check(mv)
        self.board.push(mv)
        self.halfmove_count += 1

        if not apply_reward:
            return 0.0
        return self._reward_delta_after_opp(before, self.board, opp_gives_check)

    def step_agent(self, action_index: int):
        # Nếu chưa tới lượt agent (ví dụ do ai đó gọi step sai thời điểm), tự động để đối thủ đi trước 1 nước (không thưởng/phạt ở đây).
        if self.board.turn != self.agent_color:
            self._play_heuristic_once(apply_reward=False)

        # Sau khi “sửa lượt”, nếu game đã kết thúc thì trả về ngay
        if self._is_done():
            return encode_board(self.board), legal_mask(self.board), 0.0, True, {"after": "opp_start_end"}

        # --- Agent move ---
        before = self.board.copy(stack=False)
        mv = index_to_chess_move(action_index)

        if mv not in self.board.legal_moves:
            # Illegal move -> phạt & kết thúc ván (tuỳ chính sách của bạn)
            return encode_board(self.board), legal_mask(self.board), -1.0, True, {"illegal": True}

        gave_check = self.board.gives_check(mv)
        self.board.push(mv)
        self.agent_moves += 1
        if gave_check:
            self.agent_checks += 1

        self.halfmove_count += 1
        reward = self._reward_delta_after_agent(before, self.board, gave_check)

        # Kết thúc sau nước của agent?
        done = self._is_done()
        if done:
            return encode_board(self.board), legal_mask(self.board), reward, True, {"after": "agent"}

        # --- Opponent reply (1 ply) ---
        reward += self._play_heuristic_once(apply_reward=True)
        done = self._is_done()

        obs = encode_board(self.board)
        # Nếu chưa done và tới lượt agent thì trả mask hợp lệ cho agent; ngược lại vẫn trả mask hiện tại (không hại gì)
        mask = legal_mask(self.board)
        info = {"after": "opp" if not done else "opp_end"}
        return obs, mask, reward, done, info
