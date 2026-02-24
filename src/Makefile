# Define the compiler
CC = c++

# Define the compiler flags (-O3 for max speed, -shared/-fPIC for Python module)
CCFLAGS = -O3 -Wall -shared -std=c++14 -fPIC -undefined dynamic_lookup

# Automatically grab the Pybind11 include paths from your Python environment
INCLUDES = $(shell python3 -m pybind11 --includes)

# Automatically grab the correct file extension for your OS (.so, .pyd, etc.)
SUFFIX = $(shell python3-config --extension-suffix)

# Define the target output name and source file
TARGET = tetris_engine$(SUFFIX)
SRC = test_env.cpp

# Default rule when you just type 'make'
all: $(TARGET)

# The compilation rule
$(TARGET): $(SRC)
	$(CC) $(CCFLAGS) $(INCLUDES) $(SRC) -o $(TARGET)

# Rule to clean up the compiled files
clean:
	rm -f *$(SUFFIX)