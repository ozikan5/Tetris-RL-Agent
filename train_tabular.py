from tetris_engine import TetrisEngine
from tabular_agent import TabularAgent
from tetris_features import get_features

def train(episodes=2000):
    # get the environment and the agent 
    env = TetrisEngine()
    agent = TabularAgent()

    print("Starting to train the tabular agent...")
    print(f"{episodes} iterations.")
    
    # we want epiosde iterations
    # every episode goes on until the game is over
    for episode in range(episodes):
        # we have to reset the board environment
        board = env.reset()
        game_over = False

        current_features = get_features(board)

        while not game_over:
            # get all possible moves
            possible_moves = env.get_next_states()

            # if there are no possible moves game is over
            if not possible_moves: break

            best_action = agent.select_action(possible_moves)

            reward, game_over = env.step(best_action)

            current_features = get_features(env.board)

            if game_over:
                next_possible_moves = {}
            else:
                next_possible_moves = env.get_next_states()

            agent.update(current_features, reward, next_possible_moves, game_over)

            

        # Print progress every 100 episodes so we can watch it learn
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1} | Score: {env.score} | Epsilon: {agent.epsilon:.3f} | Q-Table Size: {len(agent.q_table)}")
            
            # we will use epsilon decay, we decrement epsilon every 100 steps as we become
            # more confident on our choices
            agent.epsilon = max(0.01, agent.epsilon * 0.99)

if __name__ == "__main__":
    train()
