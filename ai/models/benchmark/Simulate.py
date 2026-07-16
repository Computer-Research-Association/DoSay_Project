import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(ROOT)
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import gymnasium as gym
from Common import load_model, run_episode, format_model_info, quiet_logs

import ai.envs
from ai.wrappers.action_mask import wrap_with_mask


# ── 설정 ─────────────────────────────────────────────────────────────
MODEL_PATH = "ai/MaskablePPO_V5.0_10000000.zip"
ROWS, COLS = 9, 18

SHOW_TRACE  = True
DELAY_SEC   = 0.5 # (Sec, TRACE 할 경우에만)
RENDER_MODE = "ansi" # (TRACE 할 경우에만)

# INIT_BOARD or SEED 입력. (INIT_BOARD 우선)
SEED = 42
INIT_BOARD = None
# INIT_BOARD = np.array([
#     [4, 2, 4, 3, 2, 5, 6, 5, 4, 9, 3, 6, 5, 5, 1, 6, 3, 3],
#     [1, 3, 5, 1, 6, 4, 1, 3, 7, 6, 1, 2, 4, 3, 3, 3, 7, 7],
#     [9, 6, 4, 3, 5, 5, 1, 7, 8, 4, 4, 2, 6, 4, 5, 7, 2, 3],
#     [5, 2, 6, 4, 9, 2, 8, 3, 1, 2, 2, 2, 7, 5, 2, 1, 1, 9],
#     [2, 2, 8, 6, 1, 5, 2, 8, 8, 4, 2, 8, 3, 2, 5, 3, 7, 3],
#     [3, 2, 8, 2, 6, 5, 6, 5, 2, 1, 3, 1, 4, 2, 3, 7, 3, 1],
#     [3, 8, 7, 8, 1, 9, 6, 2, 5, 7, 5, 9, 3, 2, 4, 4, 7, 6],
#     [2, 3, 1, 2, 4, 6, 1, 3, 4, 1, 9, 6, 4, 6, 4, 3, 1, 6],
#     [2, 1, 5, 8, 6, 2, 6, 8, 2, 2, 3, 4, 3, 6, 2, 1, 9, 6]
# ])


def main():
    quiet_logs()

    render_mode = RENDER_MODE if SHOW_TRACE else "None"
    env = gym.make("envs/AppleGame-v0", render_mode=render_mode, rows=ROWS, cols=COLS)
    env = wrap_with_mask(env)

    model, meta = load_model(MODEL_PATH, env=env)
    print(format_model_info(model, meta))

    if INIT_BOARD is not None:
        print("\n[Simulation] init_board 로 재현\n")
        steps, score = run_episode(
            model, env, meta,
            init_board=INIT_BOARD, render=SHOW_TRACE, delay=DELAY_SEC,
        )
    else:
        print(f"\n[Simulation] seed={SEED} 로 재현\n")
        steps, score = run_episode(
            model, env, meta,
            seed=SEED, render=SHOW_TRACE, delay=DELAY_SEC,
        )

    env.close()
    print("\n── Result ──────────────────────────────────")
    print(f" moves : {steps}")
    print(f" score : {score}")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()