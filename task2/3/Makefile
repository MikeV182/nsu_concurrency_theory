CC = g++
CFLAGS = -O2 -Wall
DEBAGFLAG = -g
CPARALLELFLAGS = -O2 -Wall -fopenmp

all: main 

main: main.cpp
	$(CC) $(CFLAGS) main.cpp

task31: task31.cpp
	$(CC) $(CPARALLELFLAGS) task31.cpp

task32: task32.cpp
	$(CC) $(CPARALLELFLAGS) task32.cpp

debug_main:
	$(CC) $(DEBAGFLAG) main.cpp
	gdb ./a.out

clean:
	rm -f a.out