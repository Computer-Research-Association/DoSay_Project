
import time

import numpy as np
from algorithm.evaluate_result import EvaluateSummary, GameResult
from algorithm.evaluator import pick_best_action
from dataclasses import dataclass
from game.board import Board


BOARD_SIZE = (HEIGHT, WIDTH) = (9, 18) # 보드의 크기 (행, 열)

@dataclass
class Simulator:
    weights: dict[str, float] # feature weight

    # 시뮬레이션 한 번을 수행하고, 결과를 GameResult 객체로 반환
    def play_game(self, board: Board) -> GameResult:
        turn = 0
        score = 0
        is_all_clear = False
        start_time = time.perf_counter()
        
        while True:
            is_over, is_all_clear = board.is_done()
            if is_over:
                break

            actions = board.get_valid_actions()
            best_action = pick_best_action(actions, board.grid, self.weights)
           
            area = self.get_area(board.grid, best_action)
            cleared = int((area != 0).sum())  # 이번에 지운 칸 수
            score += cleared

            board.do_action(best_action)
            turn += 1

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        return GameResult(
            score=score,
            is_all_clear=is_all_clear,
            turn=turn,
            time = elapsed_time,
            max_score_ratio = score / (HEIGHT * WIDTH)
        )

    def get_area(self, board, action):
        (r1, c1), (r2, c2) = action.top_left, action.bottom_right
        return board[r1:r2+1, c1:c2+1]
    

    # 시뮬레이션을 n_games회 반복하고, 결과를 요약하여 EvaluateSummary 객체로 반환
    def simulate(self, n_games: int) -> EvaluateSummary:
        scores, turns, times, ratios = [], [], [], []
        all_clear_count = 0

        for _ in range(n_games):
            result = self.play_game(Board(board_size=BOARD_SIZE))
            scores.append(result.score)
            turns.append(result.turn)
            times.append(result.time)
            ratios.append(result.max_score_ratio)
            if result.is_all_clear:
                all_clear_count += 1

        return self._evaluate_summary(scores, turns, times, ratios, all_clear_count, n_games)
    


    def _evaluate_summary(self, scores, turns, times, ratios, all_clear_count, n_games) -> EvaluateSummary:
        max_score = max(scores)
        min_score = min(scores)
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_turn = np.mean(turns)
        avg_time = np.mean(times)
        avg_max_score_ratio = np.mean(ratios)
        clear_rate = all_clear_count / n_games

        return EvaluateSummary(
            n_games=n_games,
            max_score=max_score,
            min_score=min_score,
            avg_score=avg_score,
            std_score=std_score,
            avg_turn=avg_turn,
            avg_time=avg_time,
            avg_max_score_ratio=avg_max_score_ratio,
            clear_rate=clear_rate,
            weights=self.weights
        )
    
if __name__ == "__main__":
    simulator = Simulator(weights={"feature1": 1.0, "feature2": 0.5})  # 예시 가중치
    summary = simulator.simulate(n_games=100)
    print(summary)
    