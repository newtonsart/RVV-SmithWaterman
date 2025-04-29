#include "fasta_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINEA 1024

struct FASTA_Parser {
    FILE *file;
    char *next_header;
    char *current_header;
    char *current_sequence;
    size_t current_sequence_len;
};

FASTA_Parser *fasta_init(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    FASTA_Parser *parser = malloc(sizeof(FASTA_Parser));
    if (!parser) {
        fclose(file);
        perror("Memory error");
        return NULL;
    }

    parser->file = file;
    parser->next_header = NULL;
    parser->current_header = NULL;
    parser->current_sequence = NULL;
    parser->current_sequence_len = 0;

    return parser;
}

int fasta_next(FASTA_Parser *parser, FASTA_Entry *entry) {
    if (!parser || !entry) return 0;

    // Reiniciar la entrada
    entry->header = NULL;
    entry->sequence = NULL;
    entry->sequence_length = 0;

    // Manejar encabezado pendiente
    if (parser->next_header) {
        free(parser->current_header);
        parser->current_header = parser->next_header;
        parser->next_header = NULL;
    }

    // Buscar nuevo encabezado si es necesario
    if (!parser->current_header) {
        char linea[MAX_LINEA];
        while (fgets(linea, sizeof(linea), parser->file)) {
            linea[strcspn(linea, "\r\n")] = '\0';
            if (linea[0] == '>') {
                parser->current_header = strdup(linea + 1);
                break;
            }
        }
        if (!parser->current_header) return 0;
    }

    // Leer secuencia
    char linea[MAX_LINEA];
    while (fgets(linea, sizeof(linea), parser->file)) {
        linea[strcspn(linea, "\r\n")] = '\0';

        if (linea[0] == '>') {
            parser->next_header = strdup(linea + 1);
            break;
        } else {
            size_t len = strlen(linea);
            char *temp = realloc(parser->current_sequence, parser->current_sequence_len + len + 1);
            if (!temp) {
                perror("Memory error");
                return 0;
            }
            parser->current_sequence = temp;
            strcpy(parser->current_sequence + parser->current_sequence_len, linea);
            parser->current_sequence_len += len;
        }
    }

    // Construir entrada
    entry->header = strdup(parser->current_header);
    entry->sequence = strdup(parser->current_sequence ? parser->current_sequence : "");
    entry->sequence_length = parser->current_sequence_len;

    // Limpiar estado del parser
    free(parser->current_header);
    free(parser->current_sequence);
    parser->current_header = NULL;
    parser->current_sequence = NULL;
    parser->current_sequence_len = 0;

    return entry->header && entry->sequence;
}

void fasta_close(FASTA_Parser *parser) {
    if (parser) {
        fclose(parser->file);
        free(parser->next_header);
        free(parser->current_header);
        free(parser->current_sequence);
        free(parser);
    }
}