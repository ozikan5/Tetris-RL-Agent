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

    