CC=gcc -I./include 
CFLAGS=-Wall -Wextra -Werror -pedantic -std=c99 -g
LDFLAGS=-lraylib -lGL -lm -lpthread -ldl -lrt -lX11

INC=./include
SRC=./src
OBJ=./obj
BIN=./bin

vpath %.c $(SRC)
vpath %.h $(INC)

EXEC=neural_network

all: mkpaths $(EXEC)

mkpaths:
	mkdir -p $(OBJ) $(BIN)

neural_network.o: neural_network.c config.h
	$(CC) $(CFLAGS) -c $< -o $(OBJ)/$@

run: neural_network
	./bin/neural_network

$(EXEC): neural_network.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(BIN)/$@ $(OBJ)/$^


clean:
	/bin/rm -rf $(OBJ) $(EXEC) $(BIN)

.PHONY: clean run
