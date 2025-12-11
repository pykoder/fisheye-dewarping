# Makefile for Fisheye Unwarper C++ Library (ctypes integration)

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -march=native -mtune=native -fPIC -ffast-math -funroll-loops -fopt-info-vec

# Target shared library
TARGET_LIB = libunwarper_ctypes.so

# Source files
SOURCES = unwarper_lib.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET_LIB)

# Build the shared library
$(TARGET_LIB): $(OBJECTS)
	$(CXX) -shared -o $(TARGET_LIB) $(OBJECTS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET_LIB)

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y g++ build-essential

# Install dependencies (macOS with Homebrew)
install-deps-macos:
	brew update
	brew install gcc

# Test the library with Python
test: $(TARGET_LIB)
	@echo "Testing the C++ library..."
	@echo "Make sure LD_LIBRARY_PATH includes the current directory:"
	@echo "export LD_LIBRARY_PATH=$(PWD):$$LD_LIBRARY_PATH"
	@echo "Then run: python3 ../unwarper_ctypes.py ../blurred.jpg"

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build the shared library"
	@echo "  clean            - Remove build artifacts"
	@echo "  install-deps     - Install dependencies on Ubuntu/Debian"
	@echo "  install-deps-macos - Install dependencies on macOS"
	@echo "  test             - Build and show test instructions"
	@echo "  help             - Show this help message"

.PHONY: all clean install-deps install-deps-macos test help