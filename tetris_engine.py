import numpy as np
import random

# Our board dimensions will be 20x10
BOARD_HEIGHT = 20
BOARD_WIDTH = 10

# We are going to define the tetrominos as coordinate matrices
# Where coordinates are (y, x) and the origin (0, 0) is the center
TETROMINOS = {
    'I': [
        [(0, -1), (0, 0), (0, 1), (0, 2)],  # 0: Horizontal
        [(-1, 0), (0, 0), (1, 0), (2, 0)],  # 1: Vertical
    ],
    'O': [
        [(0, 0), (0, 1), (1, 0), (1, 1)]    # 0: No rotation
    ],
    'T': [
        [(0, -1), (0, 0), (0, 1), (-1, 0)], # 0: Up
        [(-1, 0), (0, 0), (1, 0), (0, 0)],  # 1: Right
        [(0, -1), (0, 0), (0, 1), (1, 0)],  # 2: Down
        [(0, 0), (-1, 0), (0, 1), (1, 0)]   # 3: Left
    ],
    'S': [
        [(0, -1), (0, 0), (-1, 0), (-1, 1)],
        [(-1, 0), (0, 0), (0, 1), (1, 1)]
    ],
    'Z': [
        [(0, 1), (0, 0), (-1, 0), (-1, -1)],
        [(-1, 1), (0, 1), (0, 0), (1, 0)]
    ],
    'J': [
        [(0, -1), (0, 0), (0, 1), (-1, -1)],
        [(-1, 0), (0, 0), (1, 0), (-1, 1)],
        [(0, -1), (0, 0), (0, 1), (1, 1)],
        [(1, -1), (-1, 0), (0, 0), (1, 0)]
    ],
    'L': [
        [(0, -1), (0, 0), (0, 1), (-1, 1)],
        [(-1, 0), (0, 0), (1, 0), (1, 1)],
        [(0, -1), (0, 0), (0, 1), (1, -1)],
        [(-1, -1), (-1, 0), (0, 0), (1, 0)]
    ]
}

class TetrisEngine:
    def __init__(self):
        self.reset()

    # reset the game state
    def reset(self):
        # create a new game matrix
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        # reset score
        self.score = 0
        self.game_over = False
        # assign a new first piece
        self.current_piece = self.get_new_piece()
        
        # return our board matrix
        return self.board
    
    # function to get a new piece randomly
    def get_new_piece(self):
        # select a new shape randomly
        shape_name = random.choice(list(TETROMINOS.keys()))

        # return a dict with name and possible rotations
        return {
            'name': shape_name,
            'rotations': TETROMINOS[shape_name],
        }
    
    # checks if a piece is valid at offset_y and offset_x on the board
    # returns false if it hits walls or other pieces
    def is_valid_position(self, board, piece_coords, offset_y, offset_x):
        for (y, x) in piece_coords:
            r, c = offset_y + y, offset_x + x

            # check if we hit a wall
            if c < 0 or c >= BOARD_WIDTH or r >= BOARD_HEIGHT:
                return False
            
            # check if we hit another piece
            if r >= 0 and board[r, c] == 1:
                return False
            
        return True
    
    # generate possible next states (with rotations and position) and returns them as a dictionary
    # mapping to state, reward, value pairs
    def get_next_states(self):
        states = {}
        piece_rotations = self.current_piece['rotations']

        for rot_idx, shape_coords in enumerate(piece_rotations):
            # we will scan all possible column positions
            # offset of 2 added to catch edge cases with wide pieces
            for x in range(-2, BOARD_WIDTH + 2):
                
                # we can check if the piece can exist at the top
                # if not we can skip this position
                if not self.is_valid_position(self.board, shape_coords, 0, x):
                    continue

                y = 0
                while self.is_valid_position(self.board, shape_coords, y + 1, x):
                    y += 1

                next_board = self.board.copy()
                valid = True

                for (py, px) in shape_coords:
                    r, c = y + py, x + px
                    if 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                        next_board[r, c] = 1
                    else:
                        valid = False

                if not valid:
                    continue

                # clear lines and calculate reward
                cleared_board, lines = self.clear_lines(next_board)

                # reward: small survival bonus + strong bonus for clearing lines
                reward = 1.0 + (lines ** 2) * 10

                # check if the game is over (if there is a piece in the top row after placing)
                is_game_over = np.any(cleared_board[0, :] == 1)
                if is_game_over:
                    # negative reward for losing, but not so large that one loss dominates learning
                    # was previously -100 and the agent wasn't learning much
                    reward -= 25

                states[(rot_idx, x)] = (cleared_board, reward, is_game_over)

        return states
    
    # executes and action given by the player where the action is a tuple
    # (rotation idx, x_position)
    def step(self, action):
        rot_idx, x = action

        possible_states = self.get_next_states()

        if (rot_idx, x) in possible_states:
            self.board, reward, self.game_over = possible_states[(rot_idx, x)]
            self.score += reward
            self.current_piece = self.get_new_piece()
            return reward, self.game_over
        else:
            # if an illegal move is attempted, end the game with negative reward
            return -10, True
        
    # function to clear lines
    def clear_lines(self, board):
        # identify all full rows
        full_rows = np.all(board == 1, axis=1)
        num_cleared = np.sum(full_rows)

        if num_cleared > 0:
            # keep only rows that are not full
            board = board[~full_rows]
            # add new rows to the top
            new_rows = np.zeros((num_cleared, BOARD_WIDTH), dtype=int)
            # stack rows together
            board = np.vstack((new_rows, board))

        return board, num_cleared
    
if __name__ == "__main__":
    env = TetrisEngine()
    print("Tetris Engine Initialized.")
    print(f"Initial Board Shape: {env.board.shape}")
    print(f"First Piece: {env.current_piece['name']}")
    
    # Get all possible moves for this piece
    next_states = env.get_next_states()
    print(f"Found {len(next_states)} possible moves for this piece.")
    
    # Simulate one random move
    if next_states:
        action = list(next_states.keys())[0] # Just pick the first one
        print(f"Executing Action: Rotation {action[0]} at Column {action[1]}")
        reward, done = env.step(action)
        print(f"Step Result -> Reward: {reward}, Game Over: {done}")
        print("Board State (Top 5 rows):")
        print(env.board[:5])

