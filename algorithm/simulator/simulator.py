
import time


import random
import time
import numpy as np
from algorithm.evaluator.evaluate_result import EvaluateSummary, GameResult
from algorithm.evaluator.evaluator import pick_best_action
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
            _, cleared = board.do_action(best_action)

            score += cleared
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

    

    # 시뮬레이션을 n_games회 반복하고, 결과를 요약하여 EvaluateSummary 객체로 반환
    def simulate(self, n_games: int) -> EvaluateSummary:
        scores, turns, times, ratios = [], [], [], []
        all_clear_count = 0

        for _ in range(n_games):
            result = self.play_game(Board.from_seed(size=BOARD_SIZE, seed = None))
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
def run_single(weights: dict[str, float], n_games: int, seed: int | None = None) -> EvaluateSummary:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    simulator = Simulator(weights=weights)
    return simulator.simulate(n_games=n_games)

DEFAULT_WEIGHTS: dict[str, float] = {"remove_nine": 9.0, "remove_the_most_grouping": 3.0} #comaprison용 변수

if __name__ == "__main__":
    summary = run_single(weights={"feature1": 1.0, "feature2": 3.0}, n_games=100, seed=42) #feature 가중치 설저 , game 판 수 설정 , seed 설정
    print(summary)
    