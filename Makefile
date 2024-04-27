CC=gcc -I./include
CFLAGS=-Wall -Wextra -Werror -pedantic -std=c99 -g
LDFLAGS=-lm -lraylib

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

$(EXEC): neural_network.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(BIN)/$@ $(OBJ)/$^


.PHONY: clean
clean:
	/bin/rm -rf $(OBJ) $(EXEC) $(BIN)

