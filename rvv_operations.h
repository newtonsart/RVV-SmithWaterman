#ifndef RVV_OPERATIONS_H
#define RVV_OPERATIONS_H

#include <riscv_vector.h>
#include <stdint.h>

int8_t *optimizeCharSeq(const char *seq, int length);
void fill_matrix(int *H, const int8_t *seq1, const int8_t *seq2, int rows, int cols, int *max_score);
void fill_matrix_lmul2(int *H, const int *seq1, const int *seq2, int rows, int cols, int *max_score);
void fill_matrix_lmul4(int *H, const int *seq1, const int *seq2, int rows, int cols, int *max_score);
void fill_matrix_lmul8(int *H, const int *seq1, const int *seq2, int rows, int cols, int *max_score);

#endif