import logging
import warnings
import os

def ignore_logs():
    warnings.filterwarnings("ignore")
    for name in ("gymnasium", "gym", "stable_baselines3", "sb3_contrib"):
        logging.getLogger(name).setLevel(logging.ERROR)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")