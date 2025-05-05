#include "rvv_operations.h"
#include "blosum.h"
#include <riscv_vector.h>
#include <stdlib.h>
#include <stdint.h>

/*
* Function to convert a character sequence to an integer sequence
* where 'A' = 0, 'B' = 1, 'C' = 2, 'D' = 3.
* The function uses vectorized operations for efficiency.
*/
int8_t *optimizeCharSeq(const char *seq, int length) {
    int8_t *queryInt = malloc(length * sizeof(int8_t));
    const char *ptr = seq;
    int8_t *out_ptr = queryInt;
    size_t vl;
    size_t avl = length;

    while(avl > 0) {
        vl = __riscv_vsetvl_e8m2(avl);  // Usamos LMUL=2 para más elementos
        
        // Cargar y procesar caracteres simultaneamente
        vint8m2_t v_seq = __riscv_vle8_v_i8m2((const int8_t*)ptr, vl);
        vint8m2_t v_sub = __riscv_vsub_vx_i8m2(v_seq, 'A', vl);
        
        // Almacenar directamente en 8 bits
        __riscv_vse8_v_i8m2(out_ptr, v_sub, vl);
        
        ptr += vl;
        out_ptr += vl;
        avl -= vl;
    }

    return queryInt;
}

/*
* Function to fill the scoring matrix H using vectorized operations.
* The function computes the maximum score in the matrix and stores it in max_score.
* The function uses the BLOSUM62 scoring matrix for scoring.
*/
void fill_matrix(int *H, const int8_t *seq1, const int8_t *seq2, int rows, int cols, int *max_score) {
    *max_score = 0;
    vint32m1_t current_max_vec = __riscv_vmv_v_x_i32m1(0, 1);

    for (int d = 2; d <= rows + cols - 2; d++) {
        int i_start = (d <= cols) ? 1 : (d - cols + 1);
        int i_end = (d <= rows) ? (d - 1) : (rows - 1);
        int len = i_end - i_start + 1;

        // Buffers para valores convertidos a 32 bits
        int temp_diag[32], temp_up[32], temp_left[32], temp_result[32], temp_blosum[32];

        for (int offset = 0; offset < len; ) {
            size_t vl = __riscv_vsetvl_e32m1(len - offset);

            // Llenar buffers con extensión manual de signo
            for (int k = 0; k < vl; k++) {
                int i = i_start + offset + k;
                int j = d - i;
                
                temp_diag[k] = H[(i-1)*cols + (j-1)];
                temp_up[k] = H[(i-1)*cols + j];
                temp_left[k] = H[i*cols + (j-1)];
                temp_blosum[k] = (int)iBlosum62[seq1[i - 1] * 26 + seq2[j - 1]]; // Extensión manual
            }

            // Cargar vectores de 32 bits directamente
            vint32m1_t vec_diag = __riscv_vle32_v_i32m1(temp_diag, vl);
            vint32m1_t vec_up = __riscv_vle32_v_i32m1(temp_up, vl);
            vint32m1_t vec_left = __riscv_vle32_v_i32m1(temp_left, vl);
            vint32m1_t vec_blosum = __riscv_vle32_v_i32m1(temp_blosum, vl);

            // Operaciones vectoriales originales
            vint32m1_t vec_match = __riscv_vadd_vv_i32m1(vec_diag, vec_blosum, vl);
            vint32m1_t vec_del = __riscv_vadd_vx_i32m1(vec_up, GAP, vl);
            vint32m1_t vec_ins = __riscv_vadd_vx_i32m1(vec_left, GAP, vl);
            
            vint32m1_t vec_tmp = __riscv_vmax_vv_i32m1(
                __riscv_vmax_vv_i32m1(vec_match, vec_del, vl),
                vec_ins, vl
            );
            vec_tmp = __riscv_vmax_vx_i32m1(vec_tmp, 0, vl);

            current_max_vec = __riscv_vredmax_vs_i32m1_i32m1(vec_tmp, current_max_vec, vl);
            
            __riscv_vse32_v_i32m1(temp_result, vec_tmp, vl);
            for (int k = 0; k < vl; k++) {
                int i = i_start + offset + k;
                H[i * cols + (d - i)] = temp_result[k];
            }

            offset += vl;
        }
    }

    *max_score = __riscv_vmv_x_s_i32m1_i32(current_max_vec);
}

void fill_matrix_lmul2(int *H, const int *seq1, const int *seq2, int rows, int cols, int *max_score) {
    *max_score = 0;
    vint32m1_t current_max_vec = __riscv_vmv_v_x_i32m1(0, 1);

    for (int d = 2; d <= rows + cols - 2; d++) {
        int i_start = (d <= cols) ? 1 : (d - cols + 1);
        int i_end = (d <= rows) ? (d - 1) : (rows - 1);
        int len = i_end - i_start + 1;

        // Buffers in the stack for VLMAX=32
        int temp_diag[32], temp_up[32], temp_left[32], temp_blosum[32], temp_result[32];

        for (int offset = 0; offset < len; ) {
            size_t vl = __riscv_vsetvl_e32m2(len - offset);

            // Fill buffers and calculate BLOSUM
            for (int k = 0; k < vl; k++) {
                int i = i_start + offset + k;
                int j = d - i;
                int a = seq1[i - 1];
                int b = seq2[j - 1];
                temp_diag[k] = H[(i-1)*cols + (j-1)];
                temp_up[k] = H[(i-1)*cols + j];
                temp_left[k] = H[i*cols + (j-1)];
                temp_blosum[k] = iBlosum62[a * 26 + b];
            }

            // Load into vectors
            vint32m2_t vec_diag = __riscv_vle32_v_i32m2(temp_diag, vl);
            vint32m2_t vec_up = __riscv_vle32_v_i32m2(temp_up, vl);
            vint32m2_t vec_left = __riscv_vle32_v_i32m2(temp_left, vl);

            // Calculate BLOSUM scores for this block
            vint32m2_t vec_blosum = __riscv_vle32_v_i32m2(temp_blosum, vl);

            // Vector operations
            vint32m2_t vec_match = __riscv_vadd_vv_i32m2(vec_diag, vec_blosum, vl);
            vint32m2_t vec_del = __riscv_vadd_vx_i32m2(vec_up, GAP, vl);
            vint32m2_t vec_ins = __riscv_vadd_vx_i32m2(vec_left, GAP, vl);
            vint32m2_t vec_tmp = __riscv_vmax_vv_i32m2(vec_match, vec_del, vl);
            vec_tmp = __riscv_vmax_vv_i32m2(vec_tmp, vec_ins, vl);
            vec_tmp = __riscv_vmax_vx_i32m2(vec_tmp, 0, vl);

            // Update global maximum
            current_max_vec = __riscv_vredmax_vs_i32m2_i32m1(vec_tmp, current_max_vec, vl);

            // Store in H
            __riscv_vse32_v_i32m2(temp_result, vec_tmp, vl);
            for (int k = 0; k < vl; k++) {
                int i = i_start + offset + k;
                int j = d - i;
                H[i*cols + j] = temp_result[k];
            }

            offset += vl;
        }
    }

    *max_score = __riscv_vmv_x_s_i32m1_i32(current_max_vec); // store the maximum score
}

void fill_matrix_lmul4(int *H, const int *seq1, const int *seq2, int rows, int cols, int *max_score) {
    *max_score = 0;
    vint32m1_t current_max_vec = __riscv_vmv_v_x_i32m1(0, 1);

    for (int d = 2; d <= rows + cols - 2; d++) {
        int i_start = (d <= cols) ? 1 : (d - cols + 1);
        int i_end = (d <= rows) ? (d - 1) : (rows - 1);
        int len = i_end - i_start + 1;

        // Buffers in the stack for VLMAX=32
        int temp_diag[32], temp_up[32], temp_left[32], temp_blosum[32], temp_result[32];

        for (int offset = 0; offset < len; ) {
            size_t vl = __riscv_vsetvl_e32m1(len - offset);

            // Fill buffers and calculate BLOSUM
            for (int k = 0; k < vl; k++) {
                int i = i_start + offset + k;
                int j = d - i;
                int a = seq1[i - 1];
                int b = seq2[j - 1];
                temp_diag[k] = H[(i-1)*cols + (j-1)];
                temp_up[k] = H[(i-1)*cols + j];
                temp_left[k] = H[i*cols + (j-1)];
                temp_blosum[k] = iBlosum62[a * 26 + b];
            }

            // Load into vectors
            vint32m4_t vec_diag = __riscv_vle32_v_i32m4(temp_diag, vl);
            vint32m4_t vec_up = __riscv_vle32_v_i32m4(temp_up, vl);
            vint32m4_t vec_left = __riscv_vle32_v_i32m4(temp_left, vl);

            // Calculate BLOSUM scores for this block
            vint32m4_t vec_blosum = __riscv_vle32_v_i32m4(temp_blosum, vl);

            // Vector operations
            vint32m4_t vec_match = __riscv_vadd_vv_i32m4(vec_diag, vec_blosum, vl);
            vint32m4_t vec_del = __riscv_vadd_vx_i32m4(vec_up, GAP, vl);
            vint32m4_t vec_ins = __riscv_vadd_vx_i32m4(vec_left, GAP, vl);
            vint32m4_t vec_tmp = __riscv_vmax_vv_i32m4(vec_match, vec_del, vl);
            vec_tmp = __riscv_vmax_vv_i32m4(vec_tmp, vec_ins, vl);
            vec_tmp = __riscv_vmax_vx_i32m4(vec_tmp, 0, vl);

            // Update global maximum
            current_max_vec = __riscv_vredmax_vs_i32m4_i32m1(vec_tmp, current_max_vec, vl);

            // Store in H
            __riscv_vse32_v_i32m4(temp_result, vec_tmp, vl);
            for (int k = 0; k < vl; k++) {
                int i = i_start + offset + k;
                int j = d - i;
                H[i*cols + j] = temp_result[k];
            }

            offset += vl;
        }
    }

    *max_score = __riscv_vmv_x_s_i32m1_i32(current_max_vec); 
}

void fill_matrix_lmul8(int *H, const int *seq1, const int *seq2, int rows, int cols, int *max_score) {
    *max_score = 0;
    vint32m1_t current_max_vec = __riscv_vmv_v_x_i32m1(0, 1);

    for (int d = 2; d <= rows + cols - 2; d++) {
        int i_start = (d <= cols) ? 1 : (d - cols + 1);
        int i_end = (d <= rows) ? (d - 1) : (rows - 1);
        int len = i_end - i_start + 1;

        // Buffers in the stack for VLMAX=32
        int temp_diag[32], temp_up[32], temp_left[32], temp_blosum[32], temp_result[32];

        for (int offset = 0; offset < len; ) {
            size_t vl = __riscv_vsetvl_e32m8(len - offset);

            // Fill buffers and calculate BLOSUM
            for (int k = 0; k < vl; k++) {
                int i = i_start + offset + k;
                int j = d - i;
                int a = seq1[i - 1];
                int b = seq2[j - 1];
                temp_diag[k] = H[(i-1)*cols + (j-1)];
                temp_up[k] = H[(i-1)*cols + j];
                temp_left[k] = H[i*cols + (j-1)];
                temp_blosum[k] = iBlosum62[a * 26 + b];
            }

            // Load into vectors
            vint32m8_t vec_diag = __riscv_vle32_v_i32m8(temp_diag, vl);
            vint32m8_t vec_up = __riscv_vle32_v_i32m8(temp_up, vl);
            vint32m8_t vec_left = __riscv_vle32_v_i32m8(temp_left, vl);

            // Calculate BLOSUM scores for this block
            vint32m8_t vec_blosum = __riscv_vle32_v_i32m8(temp_blosum, vl);

            // Vector operations
            vint32m8_t vec_match = __riscv_vadd_vv_i32m8(vec_diag, vec_blosum, vl);
            vint32m8_t vec_del = __riscv_vadd_vx_i32m8(vec_up, GAP, vl);
            vint32m8_t vec_ins = __riscv_vadd_vx_i32m8(vec_left, GAP, vl);
            vint32m8_t vec_tmp = __riscv_vmax_vv_i32m8(vec_match, vec_del, vl);
            vec_tmp = __riscv_vmax_vv_i32m8(vec_tmp, vec_ins, vl);
            vec_tmp = __riscv_vmax_vx_i32m8(vec_tmp, 0, vl);

            // Update global maximum
            current_max_vec = __riscv_vredmax_vs_i32m8_i32m1(vec_tmp, current_max_vec, vl);

            // Store in H
            __riscv_vse32_v_i32m8(temp_result, vec_tmp, vl);
            for (int k = 0; k < vl; k++) {
                int i = i_start + offset + k;
                int j = d - i;
                H[i*cols + j] = temp_result[k];
            }

            offset += vl;
        }
    }

    *max_score = __riscv_vmv_x_s_i32m1_i32(current_max_vec); 
}