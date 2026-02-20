import sys
from pathlib import Path

# Add src directory to path so we can import tetris_rl
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tetris_rl.environment import TetrisEngine
from tetris_rl.agents.tabular import TabularAgent
from tetris_rl.features import get_features

def train(episodes=10000):
    env = TetrisEngine()
    agent = TabularAgent()

    print("Starting to train the tabular agent...")
    print(f"{episodes} episodes.")
    # rolling window to see learning trend despite variance
    score_window = []

    for episode in range(episodes):
        board = env.reset()
        game_over = False
        current_features = get_features(board)

        while not game_over:
            # state *before* we take the action (needed for correct TD update)
            state_before = get_features(env.board)

            # get all possible moves
            possible_moves = env.get_next_states()

            # if there are no possible moves game is over
            if not possible_moves:
                break

            best_action = agent.select_action(possible_moves)
            reward, game_over = env.step(best_action)

            # state we landed in (for TD bootstrap)
            state_after = get_features(env.board)
            agent.update(state_before, reward, state_after, game_over)
            current_features = state_after

            

        score_window.append(env.score)
        if len(score_window) > 100:
            score_window.pop(0)

        # decay exploration and learning rate over time
        agent.epsilon = max(0.02, agent.epsilon * 0.9997)
        agent.learning_rate = max(0.02, agent.learning_rate * 0.99995)

        if (episode + 1) % 100 == 0:
            avg = sum(score_window) / len(score_window)
            print(f"Episode: {episode + 1} | Score: {env.score} | Avg100: {avg:.1f} | Epsilon: {agent.epsilon:.3f} | LR: {agent.learning_rate:.4f} | States: {len(agent.q_table)}")

if __name__ == "__main__":
    train()
