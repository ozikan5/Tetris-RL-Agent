# Tetris RL Project

Reinforcement learning agents for Tetris: tabular TD(0) value learning and a neural value network (DQN-style) that learn from a 4-feature state representation.

## Overview

- **Environment**: Custom 20×10 Tetris engine with standard tetrominoes (I, O, T, S, Z, J, L). Each step: choose placement (rotation + column), get reward, next piece.
- **State representation**: Board is summarized into 4 features (see `src/tetris_rl/features.py`): aggregate height, holes, bumpiness, max height.
- **Agents**:
  - **Tabular**: Discretized state → value table; TD(0) updates and ε-greedy action selection (pick action that maximizes reward + γ·V(s′)).
  - **DQN**: Value network V(s) in PyTorch; same 4-feature input, scalar output (for use with TD targets or one-step lookahead like the tabular agent).

## Project Structure

```
Tetris_RL_Project/
├── src/
│   └── tetris_rl/
│       ├── __init__.py
│       ├── environment.py      # TetrisEngine: game logic, pieces, rewards
│       ├── features.py         # Feature extraction: heights, holes, bumpiness
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── tabular.py       # TabularAgent: discretized value learning
│       │   └── dqn.py          # DQNAgent: neural value network + replay buffer
│       └── models/
│           ├── __init__.py
│           └── dqn.py          # DQNModel: PyTorch neural network
├── scripts/
│   ├── train_tabular.py       # Training script for tabular agent
│   └── train_dqn.py           # Training script for DQN agent
├── requirements.txt
├── setup.py
└── README.md
```

## Requirements

- Python 3.8+
- NumPy
- PyTorch (for DQN agent only)

**Option 1: Install as package (recommended for development)**
```bash
pip install -e .
```
This installs the package in editable mode, so imports work everywhere and VSCode can resolve them.

**Option 2: Just install dependencies**
```bash
pip install -r requirements.txt
```
Then run scripts from project root (they handle imports automatically).

## Quick Start

**Train the tabular agent** (no PyTorch needed):

```bash
# From project root
python scripts/train_tabular.py
```

The script automatically adds `src/` to the Python path, so imports work correctly.

Defaults: 10,000 episodes, ε-greedy with decay, TD(0) with learning-rate decay. Progress prints every 100 episodes (score, 100-episode average, epsilon, LR, number of states).

**Use the Tetris engine and features** (e.g. in a script or notebook):

```python
from tetris_rl.environment import TetrisEngine
from tetris_rl.features import get_features
from tetris_rl.agents.tabular import TabularAgent
from tetris_rl.agents.dqn import DQNAgent

env = TetrisEngine()
board = env.reset()
features = get_features(board)  # shape (4,)
next_states = env.get_next_states()  # dict: (rot, x) -> (board, reward, game_over)
```

## Reward Scheme (Engine)

- +1 per piece placed.
- +10 × (lines_cleared)² for line clears.
- −25 on game over (piece overlaps top row).
- Illegal move: −10 and game over.

## License

Use as you like; no license file included.
