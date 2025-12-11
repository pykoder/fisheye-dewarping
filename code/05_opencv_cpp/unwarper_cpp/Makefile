# Makefile for OpenCV Fisheye Unwarper

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -march=native -mtune=native -ffast-math -funroll-loops

# OpenCV configuration
PKG_CONFIG = pkg-config
OPENCV_CFLAGS = $(shell $(PKG_CONFIG) --cflags opencv4)
OPENCV_LIBS = $(shell $(PKG_CONFIG) --libs opencv4)

# If pkg-config doesn't find opencv4, try opencv
ifeq ($(shell $(PKG_CONFIG) --exists opencv4; echo $$?), 1)
    OPENCV_CFLAGS = $(shell $(PKG_CONFIG) --cflags opencv)
    OPENCV_LIBS = $(shell $(PKG_CONFIG) --libs opencv)
endif

# Combine flags
ALL_CFLAGS = $(CXXFLAGS) $(OPENCV_CFLAGS)
ALL_LIBS = $(OPENCV_LIBS)

# Target executable
TARGET = unwarper

# Source files
SOURCES = unwarper.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(ALL_LIBS)

# Compile source files
%.o: %.cpp
	$(CXX) $(ALL_CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y libopencv-dev pkg-config build-essential

# Install dependencies (macOS with Homebrew)
install-deps-macos:
	brew update
	brew install opencv pkg-config

# Run the program with test image
test: $(TARGET)
	./$(TARGET) ../blurred.jpg --output-dir ../test_cpp_output --prefix blurred

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build the unwarper executable"
	@echo "  clean            - Remove build artifacts"
	@echo "  install-deps     - Install dependencies on Ubuntu/Debian"
	@echo "  install-deps-macos - Install dependencies on macOS"
	@echo "  test             - Build and run with test image"
	@echo "  help             - Show this help message"

.PHONY: all clean install-deps install-deps-macos test help