import numpy as np

# Input: 20x10 numpy array
# Output: List of 10 integers representing the height of each column
def get_column_heights(board):
    heights = []

    for i in range(board.shape[1]):
        if np.any(board[:, i]):
            # find the block index
            top = np.argmax(board[:, i])
            height = 20 - top
        else:
            height = 0
        heights.append(height)

    return heights

def get_holes(board):
    total_holes = 0

    # for every column
    for c in range(board.shape[1]):
        col = board[:, c]

        # see if the board is not empty
        if np.any(col):
            # find the argmax
            argmax = np.argmax(col)

            # now we are going to look downwards from the argmax
            down = col[argmax:]

            total_holes += np.count_nonzero(down == 0)

    return total_holes

def get_bumpiness(board):
    bumpiness = 0

    # get the heights array
    heights = get_column_heights(board)

    for height in range(board.shape[1] - 1):
        # add to the bumpiness

        bumpiness += abs(heights[height] - heights[height + 1])

    return bumpiness

# Returns a numpy array of features: [agg_height, holes, bumpiness, max_height]
def get_features(board):
    col_heights = get_column_heights(board)
    
    # get the aggregate height
    agg_height = sum(col_heights)

    # get holes 
    holes = get_holes(board)

    # get bumpiness
    bumpiness = get_bumpiness(board)

    # get max_height
    max_height = max(col_heights)

    return np.array([agg_height, holes, bumpiness, max_height])

if __name__ == "__main__":
    
    test_board = np.zeros((20, 10), dtype=int)
    
    
    test_board[19, 0] = 1
    test_board[18, 0] = 1
    
    
    test_board[19, 1] = 1
    test_board[17, 1] = 1 
    
    features = get_features(test_board)
    print(f"Features [AggHeight, Holes, Bumpiness, MaxHeight]:")
    print(features)
    
    assert features[0] == 5
    assert features[1] == 1
    assert features[2] == 4
    assert features[3] == 3
