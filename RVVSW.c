/*
 *   Arturo Navarro Muñoz
 *   RVV implementation of the Smith-Waterman algorithm
 *
 *            ⎧  H(i−1,j−1)+score(A[i],B[j])     (diagonal)
 * H(i,j)=max |  H(i−1,j)+gap penalty            (up)
 *            ⎨  H(i,j−1)+gap penalty            (left)
 *            ⎩  0                               (no alignment)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <riscv_vector.h>
#include <limits.h>
#include <ctype.h>
#include <stdint.h>
#include "fasta_parser.h"
#include "blosum.h"
#include "matrix_operations.h"
#include "rvv_operations.h"

char QUERY[1024];
char DATABASE[1024];

int parse_arguments(int argc, char **argv, char *query, char *database, int *verbose, int *lmul_value) {
    int opt;
    int q_provided = 0, d_provided = 0;
    *verbose = 0;
    *lmul_value = 1;

    while ((opt = getopt(argc, argv, "q:d:vL:")) != -1) {  
        if (opt == -1) break;

        switch (opt) {
            case 'q':
                strcpy(query, optarg);
                q_provided = 1;
                break;
                
            case 'd':
                strcpy(database, optarg);
                d_provided = 1;
                break;
                
            case 'v': 
                *verbose = 1;
                break;
                
            case 'L':
                *lmul_value = atoi(optarg);
                if (*lmul_value != 1 && *lmul_value != 2 && *lmul_value != 4 && *lmul_value != 8) {
                    fprintf(stderr, "Error: LMUL has to be 1, 2, 4 or 8.\n");
                    return 0;
                }                
                break;
                
            default:
                fprintf(stderr, "Use: %s -q <query> -d <database> [-v] [-L 1|2|4|8]\n", argv[0]);
                return 0;
        }
    }

    if (!q_provided || !d_provided) {
        fprintf(stderr, "Uso: %s -q <query> -d <database> [-v] [-L 1|2|4|8]\n", argv[0]);
        return 0;
    }

    return 1;
}

void print_progress(int comp_counter, const char *max_db, int max_score) {
    printf("\033[3A");  // Move cursor 3 lines up
    printf("\033[2K\rCurrent best match: %s\n", max_db);
    printf("\033[2K\rCurrent max score: %d\n", max_score);
    printf("\033[2K\rSequences compared: %d\n", comp_counter);
    fflush(stdout);
}

void showResults(int max_score, const char *max_db, clock_t start_time, clock_t end_time) {
    printf("\nMax score found for this sequence: %d\n", max_score);
    printf("Best match found in database:\n %s\n\n", max_db);
    printf("Time taken for this sequence: %.2f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
}

int main(int argc, char **argv)
{
    int opt, rows, cols, verbose, lmul_value;

    if (!parse_arguments(argc, argv, QUERY, DATABASE, &verbose, &lmul_value)) 
        return 1;
    

    FASTA_Parser *query_parser = fasta_init(QUERY);
    if (!query_parser)
        return 1;

    FASTA_Entry query;
    static int comp_counter = 0;
    while (fasta_next(query_parser, &query)) {
        printf(">%s\n", query.header);
        int max_score = 0;
        char *max_db = malloc(1024);
        rows = query.sequence_length + 1;
        int8_t *queryInt = optimizeCharSeq(query.sequence, rows - 1);

        FASTA_Parser *db_parser = fasta_init(DATABASE);
        if (!db_parser)
            return 1;

        FASTA_Entry db;
        clock_t start_time = clock(); // Start Timer
        
        if(verbose)
            printf("\n\n\n");

        while (fasta_next(db_parser, &db)) {
            cols = db.sequence_length + 1;

            int *H = create_matrix(rows, cols);
            int local_max_score = 0;
            int8_t *dbInt = optimizeCharSeq(db.sequence, cols - 1);
            switch (lmul_value) {
                case 1:
                    fill_matrix(H, queryInt, dbInt, rows, cols, &local_max_score);
                    break;
                case 2:
                    fill_matrix_lmul2(H, queryInt, dbInt, rows, cols, &local_max_score);
                    break;
                case 4:
                    fill_matrix_lmul4(H, queryInt, dbInt, rows, cols, &local_max_score);
                    break;
                case 8:
                    fill_matrix_lmul8(H, queryInt, dbInt, rows, cols, &local_max_score);
                    break;
                default:
                    fprintf(stderr, "Error: LMUL has to be 1, 2, 4 or 8.\n");
                    return 0;
            }
            
            if (local_max_score > max_score) {
                max_score = local_max_score;
                strcpy(max_db, db.header);
            }

            comp_counter++;

            if(verbose)
                print_progress(comp_counter, max_db, max_score);
            
            free(H);
            free(dbInt);
        }
        clock_t end_time = clock();
        showResults(max_score, max_db, start_time, end_time);
        comp_counter = 0;

        free(queryInt);
        free(max_db);

        fasta_close(db_parser);
    }

    fasta_close(query_parser);
    return 0;
}