CC = g++
CFLAGS = -O2 -Wall -fopenmp -lm # -lm Compiles against the shared library libm (basic math library, mostly C only)

all: task2

task2: task2.cpp pbPlots.cpp supportLib.cpp
	$(CC) $(CFLAGS) task2.cpp pbPlots.cpp supportLib.cpp

clean:
	rm -f a.out