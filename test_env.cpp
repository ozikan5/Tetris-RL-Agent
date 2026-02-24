#include <vector>
#include <unordered_map>
#include <cstring>
#include <random>

constexpr int BOARD_HEIGHT = 20;
constexpr int BOARD_WIDTH = 10;

// although the Python environment works perfectly
// the DQN training can be a lot faster with cpp 
// and parallel programming

// this is an test environment to test the
// env implementation with c++ and OpenMP

struct Point {
    int x;
    int y;
};

enum PieceType {
    I,
    O,
    T,
    S,
    Z,
    J,
    L
};

constexpr int PIECE_ROTATIONS[7] = {
    2, // I (Only needs Horizontal/Vertical)
    4, // J
    4, // L
    1, // O (Only needs 1 check)
    2, // S
    4, // T
    2  // Z
};

struct NextState {
    int rotation;
    int x;
    std::vector<int> board;
    float reward;
    bool game_over;
};

constexpr Point TETROMINOES[7][4][4] = {
    
    // 0: I Piece (Uses 2, Pads 2)
    {
        {{-1, 0}, {0, 0}, {1, 0}, {2, 0}},  // Rot 0
        {{0, -1}, {0, 0}, {0, 1}, {0, 2}},  // Rot 1
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}},   // --- PADDING (Unused) ---
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}}    // --- PADDING (Unused) ---
    },

    // 1: J Piece (Uses all 4)
    {
        {{-1, -1}, {-1, 0}, {0, 0}, {1, 0}}, // Rot 0
        {{0, -1}, {1, -1}, {0, 0}, {0, 1}},  // Rot 1
        {{-1, 0}, {0, 0}, {1, 0}, {1, 1}},   // Rot 2
        {{0, -1}, {0, 0}, {-1, 1}, {0, 1}}   // Rot 3
    },

    // 2: L Piece (Uses all 4)
    {
        {{1, -1}, {-1, 0}, {0, 0}, {1, 0}},  // Rot 0
        {{0, -1}, {0, 0}, {0, 1}, {1, 1}},   // Rot 1
        {{-1, 0}, {0, 0}, {1, 0}, {-1, 1}},  // Rot 2
        {{-1, -1}, {0, -1}, {0, 0}, {0, 1}}  // Rot 3
    },

    // 3: O Piece (Uses 1, Pads 3)
    {
        {{0, 0}, {1, 0}, {0, 1}, {1, 1}},    // Rot 0
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}},    // --- PADDING (Unused) ---
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}},    // --- PADDING (Unused) ---
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}}     // --- PADDING (Unused) ---
    },

    // 4: S Piece (Uses 2, Pads 2)
    {
        {{0, 0}, {1, 0}, {-1, 1}, {0, 1}},   // Rot 0
        {{0, -1}, {0, 0}, {1, 0}, {1, 1}},   // Rot 1
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}},    // --- PADDING (Unused) ---
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}}     // --- PADDING (Unused) ---
    },

    // 5: T Piece (Uses all 4)
    {
        {{0, -1}, {-1, 0}, {0, 0}, {1, 0}},  // Rot 0
        {{0, -1}, {0, 0}, {1, 0}, {0, 1}},   // Rot 1
        {{-1, 0}, {0, 0}, {1, 0}, {0, 1}},   // Rot 2
        {{0, -1}, {-1, 0}, {0, 0}, {0, 1}}   // Rot 3
    },

    // 6: Z Piece (Uses 2, Pads 2)
    {
        {{-1, 0}, {0, 0}, {0, 1}, {1, 1}},   // Rot 0
        {{1, -1}, {0, 0}, {1, 0}, {0, 1}},   // Rot 1
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}},    // --- PADDING (Unused) ---
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}}     // --- PADDING (Unused) ---
    }
};

class TetrisEngine {
private:
    // for changing random logic in each training
    std::mt19937 rng;
    std::uniform_int_distribution<int> piece_dist;
public:
    int board[BOARD_HEIGHT * BOARD_WIDTH];
    int score;
    bool game_over;
    PieceType current_piece;

    TetrisEngine() : rng(std::random_device{}()), piece_dist(0, 6) {
        this->reset();
    }

    int* reset() {
        // set the board to 0
        memset(board, 0, BOARD_HEIGHT * BOARD_WIDTH * sizeof(int));

        this->score = 0;
        this->game_over = false;
        this->current_piece = this->get_new_piece();

        return this->board;
    }

    PieceType get_new_piece() {
        return static_cast<PieceType>(piece_dist(rng));
    }

    bool is_valid_position(PieceType piece, int rotation, int x, int y) {
        // we get the relevant points for the piece and rotation
        const auto& blocks = TETROMINOES[piece][rotation];

        for (int i = 0; i < 4; i++) {
            int check_x = x + blocks[i].x;
            int check_y = y + blocks[i].y;

            // if horizontally out of bounds return false
            if (check_x < 0 || check_x >= BOARD_WIDTH) {
                return false;
            }

            // if we are vertically out of bounds also return false
            // this checks the ground dim
            if (check_y >= BOARD_HEIGHT) {
                return false;
            }

            if (check_y >= 0) {
                // if there is a piece there already, return false
                if (board[(check_y * BOARD_WIDTH) + check_x] != 0) {
                    return false; 
                }
            }

            return true;
        }
    }

    std::vector<NextState> get_next_states() {
        // initialize the vector we'll return
        std::vector<NextState> states;

        // we can have at most 40 next states
        // (10 cols * 4 rotations each)
        states.reserve(40);

        // get num of piece rotations
        int piece_rotations = PIECE_ROTATIONS[this->current_piece];

        // for each piece rotation
        for (int rot = 0; rot < piece_rotations; rot++) {
            // now we want the range of width to try
            for (int x = -2; x < BOARD_WIDTH + 2; x++) {
                int y = 0;
                if (!this->is_valid_position(this->current_piece, rot, x, y)) {
                    continue;
                }

                while (this->is_valid_position(this->current_piece, rot, x, y + 1)) {
                    y++;
                }

                NextState future;
                future.rotation = rot;
                future.x = x;
                future.game_over = false;
                future.board = std::vector<int>(this->board, this->board + (BOARD_HEIGHT * BOARD_WIDTH));

                bool valid = true;

                const auto& blocks = TETROMINOES[this->current_piece][rot];
                for (int i = 0; i < 4; i++) {
                    int final_x = x + blocks[i].x;
                    int final_y = y + blocks[i].y;

                    if (final_y >= 0 && final_y < BOARD_HEIGHT && final_x >= 0 && final_x < BOARD_WIDTH) {
                        future.board[(final_y * BOARD_WIDTH) + final_x] = 1;
                    } 
                    else {
                        valid = false;
                    }


                }

                // if this move is not possible, not valid so continue
                if (!valid) {
                    continue; 
                }

                int cleared_lines = 0;

                // iterate through every row to see lines cleared
                for (int row = BOARD_HEIGHT - 1; row >= 0; row--) {
                    bool all_clear = true;

                    // look at all cols to see if the line is clear
                    for (int col = 0; col < BOARD_WIDTH; col++) {
                        if (future.board[(row * BOARD_WIDTH) + col] == 0) {
                            all_clear = false;
                            break;
                        }
                    }

                    if (all_clear) {
                        cleared_lines++;

                        // we have to move every col one down
                        for (int pull_row = row; pull_row > 0; pull_row--) {
                            for (int col = 0; col < BOARD_WIDTH; col++) {
                                // move data from upper row to the lower row
                                future.board[(pull_row * BOARD_WIDTH) + col] = 
                                    future.board[((pull_row - 1) * BOARD_WIDTH) + col];
                            }
                        }

                        // zero out the very top row
                        for (int col = 0; col < BOARD_WIDTH; col++) {
                            future.board[col] = 0;
                        }

                        // since we moved all rows down, we have to check the same
                        // row again because there is a different line there
                        row++;
                    }
                }

                for (int col = 0; col < BOARD_WIDTH; col++) {
                    if (future.board[col] != 0) {
                        future.game_over = true;
                        break;
                    }
                }

                future.reward = 1.0f + (cleared_lines * cleared_lines) * 10.0f;

                if (future.game_over) future.reward -= 25.0f;

                states.push_back(future);

            }
        }

        return states;
    }




};


