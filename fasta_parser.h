#ifndef FASTA_PARSER_H
#define FASTA_PARSER_H
#include <stddef.h>

typedef struct FASTA_Parser FASTA_Parser;
typedef struct FASTA_Entry {
    char *header;
    char *sequence;
    size_t sequence_length;
} FASTA_Entry;

FASTA_Parser *fasta_init(const char *filename);
int fasta_next(FASTA_Parser *parser, FASTA_Entry *entry);
void fasta_close(FASTA_Parser *parser);

#endif