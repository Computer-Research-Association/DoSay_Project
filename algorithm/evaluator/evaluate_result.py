from time import time

class GameResult: # 시뮬레이션 한 번의 결과 지표
    score: int # 점수(제거한 칸 수)
    turn: int # 소요된 턴 수
    time: float # 소요된 시간(초)
    max_score_ratio : float # 0 ~ 1 범위 실수
    is_all_clear: bool # 올클리어 여부

    def __init__(self, score: int, turn: int, time: float, max_score_ratio: float, is_all_clear: bool):
        self.score = score
        self.turn = turn
        self.time = time
        self.max_score_ratio = max_score_ratio
        self.is_all_clear = is_all_clear



class EvaluateSummary: # 시뮬레이션을 N회 반복한 결과 지표
    n_games: int # 시뮬레이션 횟수
    max_score: int # 최고 점수
    min_score: int # 최저 점수
    avg_score: float # 평균 점수
    std_score: float # 표준편차 점수
    avg_turn: float # 평균 턴 수
    avg_time: float # 평균 소요 시간(초)
    avg_max_score_ratio: float # 평균 max_score_ratio
    clear_rate: float # 평균 올클리어 여부(0~1 범위 실수)
    weights: dict[str, float] # feature weight

    def __init__(self, n_games: int, max_score: int, min_score: int, avg_score: float, std_score: float, avg_turn: float,
                avg_time: float, avg_max_score_ratio: float, clear_rate: float, weights: dict[str, float]):
        self.n_games = n_games
        self.max_score = max_score
        self.min_score = min_score
        self.avg_score = avg_score
        self.std_score = std_score
        self.avg_turn = avg_turn
        self.avg_time = avg_time
        self.avg_max_score_ratio = avg_max_score_ratio
        self.clear_rate = clear_rate
        self.weights = weights

    



    



        