#include <vector>
#include <unordered_map>

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

