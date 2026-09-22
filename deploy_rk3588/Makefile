CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fPIC -Wall -Wextra

all: libedgefall_yolo_postprocess.so

libedgefall_yolo_postprocess.so: yolo_postprocess.cpp yolo_postprocess.h
	$(CXX) $(CXXFLAGS) -shared yolo_postprocess.cpp -o $@

clean:
	rm -f libedgefall_yolo_postprocess.so

.PHONY: all clean
