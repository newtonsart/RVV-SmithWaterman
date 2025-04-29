# clang-17 -o vector_add.clang vector_add.c -O2 -march=rv64imafdcv -no-integrated-as
CC = clang-17
CFLAGS = -O2 -march=rv64imafdcv -no-integrated-as
TARGET = RVVSW
SRC = RVVSW.c fasta_parser.c
OBJ = $(SRC:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.c fasta_parser.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJ) $(TARGET)
