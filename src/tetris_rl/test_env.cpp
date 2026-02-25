#include <vector>
#include <unordered_map>
#include <cstring>
#include <random>
#include <algorithm>
#include <omp.h>

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

// CRITICAL: Order MUST match Python's list(TETROMINOS.keys()) = ['I','O','T','S','Z','J','L']
enum PieceType {
    I,   // 0
    O,   // 1
    T,   // 2
    S,   // 3
    Z,   // 4
    J,   // 5
    L    // 6
};

constexpr int PIECE_ROTATIONS[7] = {
    2,  // 0: I
    1,  // 1: O
    4,  // 2: T
    2,  // 3: S
    2,  // 4: Z
    4,  // 5: J
    4   // 6: L
};

struct NextState {
    int rotation;
    int x;
    std::vector<int> board;
    float reward;
    bool game_over;
};

struct StepResult {
    float reward;
    bool game_over;
};

// Python (y,x) -> C++ Point{x,y} since .x=col, .y=row
// Array index MUST match Python list(TETROMINOS.keys()) = ['I','O','T','S','Z','J','L']
constexpr Point TETROMINOES[7][4][4] = {
    // 0: I
    {
        {{-1, 0}, {0, 0}, {1, 0}, {2, 0}},
        {{0, -1}, {0, 0}, {0, 1}, {0, 2}},
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}}
    },
    // 1: O
    {
        {{0, 0}, {1, 0}, {0, 1}, {1, 1}},
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}}
    },
    // 2: T
    {
        {{-1, 0}, {0, 0}, {1, 0}, {0, -1}},
        {{0, -1}, {0, 0}, {0, 1}, {1, 0}},
        {{-1, 0}, {0, 0}, {1, 0}, {0, 1}},
        {{0, -1}, {-1, 0}, {0, 0}, {0, 1}}
    },
    // 3: S
    {
        {{-1, 0}, {0, 0}, {0, -1}, {1, -1}},
        {{0, -1}, {0, 0}, {1, 0}, {1, 1}},
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}}
    },
    // 4: Z
    {
        {{1, 0}, {0, 0}, {0, -1}, {-1, -1}},
        {{1, -1}, {1, 0}, {0, 0}, {0, 1}},
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}}
    },
    // 5: J
    {
        {{-1, 0}, {0, 0}, {1, 0}, {-1, -1}},
        {{0, -1}, {0, 0}, {0, 1}, {1, -1}},
        {{-1, 0}, {0, 0}, {1, 0}, {1, 1}},
        {{-1, 1}, {-1, 0}, {0, 0}, {0, 1}}
    },
    // 6: L
    {
        {{-1, 0}, {0, 0}, {1, 0}, {1, -1}},
        {{0, -1}, {0, 0}, {0, 1}, {1, 1}},
        {{-1, 0}, {0, 0}, {1, 0}, {-1, 1}},
        {{-1, -1}, {-1, 0}, {0, 0}, {0, 1}}
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
        }

        return true;
    }

    std::vector<NextState> get_next_states() {
        // initialize the vector we'll return
        std::vector<NextState> global_states;

        // we can have at most 40 next states
        // (10 cols * 4 rotations each)
        global_states.reserve(40);

        // get num of piece rotations
        int piece_rotations = PIECE_ROTATIONS[this->current_piece];

        // create a parallel region for all threads
        #pragma omp parallel
        {

            // we can crate a local vector for each of the threads
            std::vector<NextState> local_states;
            local_states.reserve(40);

            // parallelize the for loops and collapse both loops into flat lists 
            #pragma omp for collapse(2) nowait
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

                    local_states.push_back(future);

                }
            }

            // now that the thread finished its state calculations, we use a critical region so that
            // no multiple threads try to modify the global states at the same time -> race condition
            #pragma omp critical
            {
                // insert to the end of the states vector
                global_states.insert(global_states.end(), local_states.begin(), local_states.end());
            }
        }

        return global_states;
    }

    StepResult step(int rot, int x_pos) {
        std::vector<NextState> next_states = this->get_next_states();

        bool found = false;
        float reward = 0.0f;

        for (const auto& state : next_states) {
            if (state.rotation == rot && state.x == x_pos) {
                found = true;
                
                // copy the vector into the board
                std::copy(state.board.begin(), state.board.end(), this->board);

                this->game_over = state.game_over;

                reward = state.reward;

                this->score += reward;

                
                if (!this->game_over) {
                    this->current_piece = this->get_new_piece();
                }

                break;

            }
        }

        StepResult res;

        if (!found) {
            res.reward = -10;
            res.game_over = true;
            this->game_over = true;
        }
        else {
            res.reward = reward;
            res.game_over = this->game_over;
        }

        return res;
    }

    // helper function to get the board for pybind
    std::vector<int> get_board() {
        return std::vector<int>(this->board, this->board + (BOARD_HEIGHT * BOARD_WIDTH));
    }
};

#include <pybind11/pybind11.h>
// for converting vectors into python lists
#include <pybind11/stl.h> 

namespace py = pybind11;

// we have the same class structure as our python script
PYBIND11_MODULE(tetris_engine, m) {
    
    // create classes for stepresult and nextstate structs
    py::class_<StepResult>(m, "StepResult")
        .def_readonly("reward", &StepResult::reward)
        .def_readonly("game_over", &StepResult::game_over);

    py::class_<NextState>(m, "NextState")
        .def_readonly("rotation", &NextState::rotation)
        .def_readonly("x", &NextState::x)
        .def_readonly("board", &NextState::board)
        .def_readonly("reward", &NextState::reward)
        .def_readonly("game_over", &NextState::game_over);

    // bind the main tetrisengine class with its functions
    py::class_<TetrisEngine>(m, "TetrisEngine")
        .def(py::init<>()) // Expose the constructor
        .def("reset", &TetrisEngine::reset)
        .def("step", &TetrisEngine::step)
        .def("get_next_states", &TetrisEngine::get_next_states)
        .def("get_board", &TetrisEngine::get_board)
        
        .def_readwrite("score", &TetrisEngine::score)
        .def_readwrite("game_over", &TetrisEngine::game_over)
        .def_property_readonly("current_piece", [](const TetrisEngine& env) {
            return static_cast<int>(env.current_piece); // Convert enum to int for Python
        });
}


