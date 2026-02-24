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

        for (int i = 0; i < 4, i++) {
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

    


};


