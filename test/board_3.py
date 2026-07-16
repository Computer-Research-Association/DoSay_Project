import random, time
import numpy as np
from game import Board

board = Board.from_seed(size=(9, 18), seed=0)

random.seed(0)
start = time.time()

while True:
    valid_actions = board.get_valid_actions()
    if len(valid_actions) > 0:
        action = random.choice(valid_actions)
        board.do_action(action)
    else:
        board.print_board()
        break

end = time.time()

print("GAME OVER!")
print(f"time elapsed: {(end-start):.3f}s")
print(f"SCORE: {np.count_nonzero(board.grid)}")