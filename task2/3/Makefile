CC = g++
CFLAGS = -O2 -Wall -fopenmp
DEBAGFLAG = -g

all: main 

main: main.cpp
	$(CC) $(CFLAGS) main.cpp

task31: task31.cpp
	$(CC) $(CFLAGS) task31.cpp

task32: task32.cpp
	$(CC) $(CFLAGS) task32.cpp

task33: task33.cpp
	$(CC) $(CFLAGS) task33.cpp

debug_main:
	$(CC) $(DEBAGFLAG) main.cpp
	gdb ./a.out

clean:
	rm -f a.out