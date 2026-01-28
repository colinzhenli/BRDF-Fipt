# Makefile for MERL BRDF Reference Implementation

CXX = g++
CXXFLAGS = -O3 -Wall -std=c++11
TARGET = brdf_read

all: $(TARGET)

$(TARGET): BRDFRead.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) BRDFRead.cpp -lm

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@echo "Usage: ./$(TARGET) <path_to_brdf.binary>"
	@echo "Example: ./$(TARGET) /path/to/alum-bronze.binary"

.PHONY: all clean test


