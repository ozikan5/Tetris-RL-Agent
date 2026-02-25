import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

#from tetris_rl.environment import TetrisEngine
from tetris_rl.agents import DQNAgent
from tetris_rl.features import get_features
import tetris_rl.tetris_engine as tetris_engine
import numpy as np

MAX_PIECES_IN_GAME = 5000

# main script to train the DQN agent and output the results
def train_dqn(batch_size=64, queue_len=100000, hidden_layer_size=64, episodes=10000):
    # initialize our environment and agent
    env = tetris_engine.TetrisEngine()
    agent = DQNAgent(batch_size, queue_len, hidden_layer_size)

    print("Starting to train the DQN agent...")
    print(f"{episodes} episodes.")
    # rolling window to see learning trend despite variance
    score_window = []

    for episode in range(episodes):
        env.reset()
        game_over = False
        pieces = 0

        while not game_over:
            # get the state before taking action
            current_board = np.array(env.get_board(), dtype=np.float32).reshape(20, 10)
            state_before = get_features(current_board)

            # get all possible moves
            cpp_moves = env.get_next_states()
            possible_moves = {}

            # iterate throuhg next state objects
            # we have to translate the objects into the possible moves dict like we did before
            for state in cpp_moves:
                board_2d = np.array(state.board, dtype=np.float32).reshape(20, 10)
                possible_moves[(state.rotation, state.x)] = (board_2d, state.reward, state.game_over)

            if not possible_moves:
                break

            # select action using epsilon-greedy policy
            best_action = agent.act(possible_moves)
            step_res = env.step(best_action[0], best_action[1])

            reward = step_res.reward
            game_over = step_res.game_over

            next_board = np.array(env.get_board(), dtype=np.float32).reshape(20, 10)

            # get the new state after action
            state_after = get_features(next_board)

            # save experience to replay buffer: (state, reward, next_state, done)
            agent.buffer.save(state_before, reward, state_after, game_over)

            # learn from replay buffer (samples batch, computes TD targets, updates model)
            agent.learn()

            pieces += 1
            # to avoid infinitely going game problem, set a limit to max pieces
            if pieces > MAX_PIECES_IN_GAME:
                game_over = True

        # track scores for logging
        score_window.append(env.score)
        if len(score_window) > 100:
            score_window.pop(0)

        # decay epsilon over time (explore less as we learn)
        agent.update_epsilon()

        # print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg = sum(score_window) / len(score_window)
            buffer_size = agent.buffer.size()
            print(f"Episode: {episode + 1} | Score: {env.score} | Avg100: {avg:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | Buffer: {buffer_size}")

if __name__ == "__main__":
    train_dqn()
