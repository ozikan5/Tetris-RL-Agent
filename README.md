# Tetris RL Project

Reinforcement learning agents for Tetris: tabular TD(0) value learning and a DQN-style neural value network. Supports both a Python environment and a faster C++/OpenMP environment.

## Overview

- **Environment**: Custom 20×10 Tetris engine with standard tetrominoes (I, O, T, S, Z, J, L). Each step: choose placement (rotation + column), get reward, next piece.
- **State representation**: Board is summarized into 4 features (see `src/tetris_rl/features.py`): aggregate height, holes, bumpiness, max height.
- **Agents**:
  - **Tabular**: Discretized state → value table; TD(0) updates and ε-greedy action selection.
  - **DQN**: Value network V(s) in PyTorch with replay buffer; same 4-feature input, scalar output.
- **Environments**: Python implementation (`environment.py`) and C++/OpenMP implementation (`test_env.cpp`) exposed via Pybind11 for faster `get_next_states()`.

## Project Structure

```
Tetris_RL_Project/
├── src/
│   └── tetris_rl/
│       ├── __init__.py
│       ├── environment.py      # Python TetrisEngine
│       ├── features.py         # Feature extraction: heights, holes, bumpiness
│       ├── test_env.cpp        # C++ TetrisEngine with OpenMP (pybind11)
│       ├── Makefile            # Builds tetris_engine.so
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── tabular.py      # TabularAgent
│       │   └── dqn.py          # DQNAgent (replay buffer, TD learning)
│       └── models/
│           ├── __init__.py
│           └── dqn.py          # DQNModel (PyTorch)
├── scripts/
│   ├── train_tabular.py        # Tabular agent (Python env)
│   ├── train_dqn_py.py         # DQN agent with Python env
│   └── train_dqn_cpp.py        # DQN agent with C++ env (faster)
├── requirements.txt
├── setup.py
└── README.md
```

## Requirements

- Python 3.8+
- NumPy
- PyTorch (for DQN)
- Pybind11 (for C++ env)
- C++ compiler with OpenMP support
- libomp (macOS: `brew install libomp`)

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Optional: Install as package (recommended for development)**
```bash
pip install -e .
```

## Quick Start

**Train the tabular agent** (Python env only):
```bash
python scripts/train_tabular.py
```

**Train the DQN agent with Python env:**
```bash
python scripts/train_dqn_py.py
```

**Train the DQN agent with C++ env** (faster environment stepping):
```bash
cd src/tetris_rl && make && cd ../..
python scripts/train_dqn_cpp.py
```

The C++ environment must be built before running `train_dqn_cpp.py`. The compiled `tetris_engine.*.so` is placed in `src/tetris_rl/`.

## Building the C++ Environment

From the project root:
```bash
cd src/tetris_rl
make
```

On macOS with Homebrew, ensure `libomp` is installed: `brew install libomp`.  
If libomp paths differ on your system, edit `Makefile` (`INCLUDES`, `LDFLAGS`) to match.

## Usage Example

```python
# Python environment
from tetris_rl.environment import TetrisEngine
from tetris_rl.features import get_features
from tetris_rl.agents.tabular import TabularAgent
from tetris_rl.agents.dqn import DQNAgent

env = TetrisEngine()
board = env.reset()
features = get_features(board)  # shape (4,)
next_states = env.get_next_states()  # dict: (rot, x) -> (board, reward, game_over)

# C++ environment (after building)
import tetris_rl.tetris_engine as cpp_env
env = cpp_env.TetrisEngine()
env.reset()
cpp_moves = env.get_next_states()  # list of NextState objects
# Convert to same format as Python for agent.act()
```

## Reward Scheme

- +1 per piece placed
- +10 × (lines_cleared)² for line clears
- −25 on game over (piece overlaps top row)
- −10 and game over on illegal move

## License

Use as you like; no license file included.
