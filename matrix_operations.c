#include "matrix_operations.h"
#include <stdlib.h>
#include <stdio.h>

int *create_matrix(int rows, int cols) {
    return (int *)calloc(rows * cols, sizeof(int));
}

void print_matrix(const int *H, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", H[i * cols + j]);
        }
        printf("\n");
    }
}