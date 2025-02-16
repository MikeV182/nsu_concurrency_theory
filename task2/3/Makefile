CC = g++
CFLAGS = -O2 -Wall -fopenmp
DEBAGFLAG = -g

all: main 

main: main.cpp
	$(CC) $(CFLAGS) main.cpp

debug:
	$(CC) $(DEBAGFLAG) main.cpp
	gdb ./a.out

clean:
	rm -f a.out