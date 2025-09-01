# -*- coding: utf-8 -*-
"""Centralized config for the chess RL-style project."""


# Engine defaults
ENGINE_PATH = "stockfish" # or absolute path to stockfish(.exe)
ENGINE_THREADS = 2
ENGINE_HASH_MB = 256
ENGINE_LIMIT_STRENGTH = True
ENGINE_ELO = 1350 # clamp to [800, 2800]
ENGINE_MOVETIME_S = 0.7


# Env defaults
REWARD_WIN = 1.0
REWARD_LOSS = -1.0
REWARD_DRAW = 0.0
REWARD_INTERMEDIATE = 0.0 # sparse reward (0 until terminal)


# Rendering
SHOW_LEGAL_MOVES_HINT = True