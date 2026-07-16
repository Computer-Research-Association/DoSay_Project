import os, sys, random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Board


apple_grid = np.array([
    [1, 1, 2, 6, 4, 9, 3, 5, 2, 6, 9, 1, 8, 6, 2, 8, 7, 6],
    [5, 7, 2, 9, 3, 3, 8, 2, 9, 9, 7, 6, 8, 8, 4, 4, 9, 4],
    [3, 7, 3, 4, 1, 4, 1, 5, 5, 2, 3, 5, 4, 8, 3, 5, 9, 2],
    [9, 1, 3, 3, 9, 7, 8, 8, 4, 9, 8, 4, 5, 6, 6, 3, 3, 6],
    [9, 2, 1, 1, 9, 5, 3, 6, 6, 9, 7, 6, 7, 7, 8, 4, 2, 5],
    [1, 8, 7, 2, 2, 5, 2, 5, 4, 7, 3, 6, 9, 5, 4, 7, 3, 1],
    [7, 2, 4, 7, 3, 8, 3, 9, 7, 7, 5, 4, 9, 9, 6, 6, 3, 9],
    [8, 9, 4, 8, 4, 3, 8, 8, 4, 1, 5, 8, 9, 1, 5, 1, 6, 7],
    [2, 8, 4, 9, 1, 4, 8, 2, 3, 1, 4, 8, 7, 8, 8, 5, 2, 3],
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
remove_count = np.count_nonzero(board.grid)

print("GAME OVER!")
print(f"SCORE: {remove_count}")