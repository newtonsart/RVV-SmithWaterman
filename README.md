# Smith-Waterman Algorithm (RVV Implementation)

This project is an optimized implementation of the **Smith-Waterman** local sequence alignment algorithm using the **RISC-V Vector Extension (RVV)**. It supports dynamic configuration of vector lengths through LMUL settings and compares a query sequence against a sequence database using a substitution matrix (BLOSUM62).

## Features

- Vectorized dynamic programming using **RVV intrinsics** for acceleration.
- Supports multiple LMUL settings (1, 2, 4, 8).
- Uses BLOSUM62 for scoring.
- Parses input sequences in FASTA format.
- Provides verbose mode with real-time progress and match reporting.

## Algorithm Overview

The Smith-Waterman algorithm computes local alignments using the recurrence:

```
       ⎧ H(i−1,j−1) + score(A[i], B[j])     (diagonal)
H(i,j)=| H(i−1,j) + gap\_penalty            (up)
       | H(i,j−1) + gap\_penalty            (left)
       ⎩ 0                                  (no alignment)
````

This implementation leverages RVV to fill the score matrix more efficiently via vector operations.

## Build Instructions

This program is intended to run on systems with RISC-V processors that support the RVV 1.0 extension.

## Usage

```bash
./smith_waterman -q <query.fasta> -d <database.fasta> [-v] [-L 1|2|4|8]
```

### Arguments

* `-q <query>`: Path to the query sequence in FASTA format.
* `-d <database>`: Path to the database file with multiple FASTA sequences.
* `-v`: (Optional) Enable verbose mode to display alignment progress.
* `-L <LMUL>`: (Optional) Set RVV LMUL value: `1`, `2`, `4`, or `8`. Default is `1`.

## Output

* Prints the best match header and alignment score.
* Shows time taken for the comparison.
* In verbose mode, continuously updates best score and progress.
