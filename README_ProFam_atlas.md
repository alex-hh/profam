The ProFam Atlas Dataset contains protein family definitions and sequences for various family definitions including:

- FoldSeek AFDB
- OpenFold OpenProteinSet (re-clustered) MSAs
- TED clustered
- TED FunFams Clustered
- UniRef90 (single sequences)

The dataset is distributed as a collection of paired `.mapping` and `.sequences` files. This structure allows for efficient random access to specific families without loading entire files into memory.

### Mapping Files (`*.mapping`)

Contains the index information for each family.

- **Format:**

  - Line 1: Family Identifier (prefixed with `>`).

  - Line 2: The name of the corresponding `.sequences` file, followed by a colon `:`, and a comma-separated list of (entry indices) for the sequences belonging to that family.

**Example:**

```text

>1.25.40.900_ted.fasta.0

train_001.sequences:0,2,4,5,6,7,8...

```

### Sequence Files (`*.sequences`)

Contains the actual protein sequences in a FASTA-like format.

- **Format:**

  - Header line starting with `>` containing the sequence identifier (and potentially metadata).

  - Sequence line containing the amino acid sequence.

**Example:**

```text

>AF-A0A1H4DYA3-F1-model_v4_TED03/0.0

EWAKAETAANAVINSGIYTLNKDLNLRLAEQYLIRAEARAQLGNLAGAVADVDSIRSKAGLPQLDNSITQPALLLAIEKERKAELFGEWGHRWFDLKRTPAVAGGGKTRAD

>AF-A0A1W1ZJU1-F1-model_v4_TED01/1.1

LLMQAEAENEVNGPTQVAYNAVNEVRHRAGLPDLTPGLAKEAFFNALVDERAHELCFEGFRKWDLIRWNMLGAKIRATQTALKAYRANFPYVAGDNF

```

.mapping files must not have a trailing new line at the end of the file.

.sequences files should have a final new line

Full details of dataset construction can be found in the preprint:
"ProFam: Open-Source Protein Family Language Modeling for Fitness Prediction and Design" Wells et. Al 2025
