CC = clang-17
CFLAGS = -O2 -march=rv64imafdcv -no-integrated-as -MMD
TARGET = RVVSW
SRC = RVVSW.c fasta_parser.c blosum.c matrix_operations.c rvv_operations.c
OBJ = $(SRC:.c=.o)
DEP = $(SRC:.c=.d)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

-include $(DEP)

clean:
	$(RM) $(OBJ) $(TARGET) $(DEP)
