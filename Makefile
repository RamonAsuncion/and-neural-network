# Neural Network
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

# just make neural_network executable from the neural_network.c file in src and put it in bin and add the neural_network header file in include directory

neural_network.o: neural_network.c neural_network.h
	$(CC) $(CFLAGS) -c $< -o $(OBJ)/$@

$(EXEC): neural_network.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(BIN)/$@ $(OBJ)/$^ -lm


.PHONY: clean 
clean:
	/bin/rm -rf $(OBJ) $(EXEC) $(BIN)

