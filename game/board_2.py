import os, sys, random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Board


apple_grid = np.array([
    [4, 2, 4, 3, 2, 5, 6, 5, 4, 9, 3, 6, 5, 5, 1, 6, 3, 3],
    [1, 3, 5, 1, 6, 4, 1, 3, 7, 6, 1, 2, 4, 3, 3, 3, 7, 7],
    [9, 6, 4, 3, 5, 5, 1, 7, 8, 4, 4, 2, 6, 4, 5, 7, 2, 3],
    [5, 2, 6, 4, 9, 2, 8, 3, 1, 2, 2, 2, 7, 5, 2, 1, 1, 9],
    [2, 2, 8, 6, 1, 5, 2, 8, 8, 4, 2, 8, 3, 2, 5, 3, 7, 3],
    [3, 2, 8, 2, 6, 5, 6, 5, 2, 1, 3, 1, 4, 2, 3, 7, 3, 1],
    [3, 8, 7, 8, 1, 9, 6, 2, 5, 7, 5, 9, 3, 2, 4, 4, 7, 6],
    [2, 3, 1, 2, 4, 6, 1, 3, 4, 1, 9, 6, 4, 6, 4, 3, 1, 6],
    [2, 1, 5, 8, 6, 2, 6, 8, 2, 2, 3, 4, 3, 6, 2, 1, 9, 6]
])


board = Board(init_board=apple_grid)

board.print_board()

result = []
for i in range(100):
    while True:
        valid_actions = board.get_valid_actions()
        if len(valid_actions) > 0:
            action = random.choice(valid_actions)
            board.do_action(action)
        else:
            board.print_board()
            print()
            break
result.append(np.count_nonzero(board.grid != 0))

print("GAME OVER!")
print(f"LEFT APPLE(avg): {(sum(result)/len(result)):.1f}")