CC = g++
CFLAGS = -O2 -Wall

ifdef USE_FLOAT
    CFLAGS += -DFLOAT=1
else
    CFLAGS += -DFLOAT=0
endif

all: task1

task1: task1.cpp
	$(CC) $(CFLAGS) task1.cpp -o task1

clean:
	rm -f task1
