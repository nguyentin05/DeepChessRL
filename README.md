# Chess RL-style Agent (python-chess + Stockfish)


This project splits responsibilities into environment and agent files, following a reinforcement-learning style:


- `env.py`: Gym-like environment (`reset`, `step`, `observation`, `reward`, `done`, `info`).
- `agents/stockfish_agent.py`: wraps Stockfish as a policy/agent (`select_move`).
- `play.py`: command-line loop for human vs agent.


## Install
```bash
pip install -r requirements.txt
