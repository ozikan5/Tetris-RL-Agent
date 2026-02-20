# Tetris RL Project

Reinforcement learning agents for Tetris: tabular TD(0) value learning and a neural value network (DQN-style) that learn from a 4-feature state representation.

## Overview

- **Environment**: Custom 20×10 Tetris engine with standard tetrominoes (I, O, T, S, Z, J, L). Each step: choose placement (rotation + column), get reward, next piece.
- **State representation**: Board is summarized into 4 features (see `tetris_features.py`): aggregate height, holes, bumpiness, max height.
- **Agents**:
  - **Tabular**: Discretized state → value table; TD(0) updates and ε-greedy action selection (pick action that maximizes reward + γ·V(s′)).
  - **DQN**: Value network V(s) in PyTorch; same 4-feature input, scalar output (for use with TD targets or one-step lookahead like the tabular agent).

## Project structure

| File | Description |
|------|-------------|
| `tetris_engine.py` | Tetris game logic: board, pieces, `get_next_states()`, `step()`, line clearing, rewards. |
| `tetris_features.py` | Feature extraction: column heights, holes, bumpiness, max height → 4-D vector. |
| `tabular_agent.py` | Tabular value agent: discretize state, ε-greedy select action, TD(0) update. |
| `train_tabular.py` | Training loop for the tabular agent (episodes, epsilon/LR decay, logging). |
| `dqn_model.py` | PyTorch `DQNModel`: 4 → hidden → hidden → hidden → 1 (value V(s)). |

## Requirements

- Python 3.8+
- NumPy
- PyTorch (for `dqn_model.py` only)

```bash
pip install numpy torch
```

## Quick start

**Train the tabular agent** (no PyTorch needed):

```bash
python train_tabular.py
```

Defaults: 10,000 episodes, ε-greedy with decay, TD(0) with learning-rate decay. Progress prints every 100 episodes (score, 100-episode average, epsilon, LR, number of states).

**Use the Tetris engine and features** (e.g. in a script or notebook):

```python
from tetris_engine import TetrisEngine
from tetris_features import get_features

env = TetrisEngine()
board = env.reset()
features = get_features(board)  # shape (4,)
next_states = env.get_next_states()  # dict: (rot, x) -> (board, reward, game_over)
```

## Reward scheme (engine)

- +1 per piece placed.
- +10 × (lines_cleared)² for line clears.
- −25 on game over (piece overlaps top row).
- Illegal move: −10 and game over.

## License

Use as you like; no license file included.
