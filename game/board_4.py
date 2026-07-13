"""
중복 수 걸러지는지 확인하기 위해 수 개수 테스트
"""

import os, sys, random, time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Board

board = Board(board_size=(9, 18), seed=0)

random.seed(0)
start = time.time()

for i in range(10):
    valid_actions = board.get_valid_actions()
    if len(valid_actions) == 0: break
    action = random.choice(list(valid_actions))
    board.do_action(action)
board.print_board()

valid_actions = board.get_valid_actions()

end = time.time()

print(f"time elapsed: {(end-start):.10f}s")
print(f"ACTION COUNT: {len(valid_actions)}")

# list -> 평균 0.00630 (46)
# set -> 평균 0.00731 (37)