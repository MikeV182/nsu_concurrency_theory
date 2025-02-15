CC = g++
CFLAGS = -O2 -Wall -fopenmp -lm # -lm Compiles against the shared library libm (basic math library, mostly C only)

all: task1

task1: task1.cpp pbPlots.cpp supportLib.cpp
	$(CC) $(CFLAGS) task1.cpp pbPlots.cpp supportLib.cpp

clean:
	rm -f a.out