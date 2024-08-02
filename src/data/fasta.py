"""
N.B. there seems to be a bug if the first line is not >?

A sequence in FASTA format begins with a single-line description, followed by lines
of sequence data. The description line is distinguished from the sequence data by a
greater-than (">") symbol in the first column. It is recommended that all lines of
text be shorter than 80 characters in length.

The description line is optionally in format >id description (id and description
                                                              separated by whitespace)

https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=BlastHelp
"""
import gzip
import os
import re
from contextlib import contextmanager

import pandas as pd


@contextmanager
def gzread(filename, encoding=None):
    if os.path.splitext(filename)[-1] == ".gz":
        f = gzip.open(filename, "rt", encoding=encoding)
        yield f
    else:
        f = open(filename, "r")
        yield f
    f.close()


def read_fasta_lines(lines, keep_gaps=True, keep_insertions=True, to_upper=False):
    """
    From esm
    Works for fasta and a2m/a3m
    """
    seq = desc = None

    def parse(s):
        if not keep_gaps:
            s = re.sub("-", "", s)
            s = re.sub("\.", "", s)
        if not keep_insertions:
            s = re.sub(r"[a-z\.]", "", s)
        return s.replace(".", "-").upper() if to_upper else s

    for line in lines:
        # Line may be empty if seq % file_line_width == 0
        if len(line) > 0 and line[0] == ">":
            if seq is not None:
                yield desc, parse(seq)
            desc = line.strip()[1:]
            seq = ""
        else:
            assert isinstance(seq, str)
            seq += line.strip()
    assert isinstance(seq, str) and isinstance(desc, str)
    yield desc, parse(seq)


def convert_sequence_with_positions(
    seq, keep_gaps=True, keep_insertions=True, to_upper=False
):
    """
    Get positions relative to sequence. For alignments position is relative to match states.
    i.e. insertions have the same position as the previous match state.

    TODO: write test
    """
    match_index = 0
    positions = []
    sequence = ""
    if not keep_gaps:
        assert keep_insertions, "If not keeping gaps should keep insertions"
    if keep_insertions:
        assert to_upper, "If keeping insertions should convert to upper case"
    for aa in seq:
        if keep_gaps or aa != "-":
            if aa == ".":
                # dont keep gaps in insert columns: we can modify later if we ever want to use
                pass
            # TODO: check for valid characters
            upper = aa.upper()
            if upper == aa or keep_insertions:
                positions.append(match_index)
                sequence += upper
                if upper == aa and aa != ".":  # includes case where aa is "-"
                    match_index += 1

    assert len(positions) == len(sequence)
    return sequence, positions


def read_fasta_lines_with_positions(
    lines, keep_gaps=True, keep_insertions=True, to_upper=False
):
    seq = desc = None

    for line in lines:
        # Line may be empty if seq % file_line_width == 0
        if len(line) > 0 and line[0] == ">":
            if seq is not None:
                seq, pos = convert_sequence_with_positions(
                    seq,
                    keep_gaps=keep_gaps,
                    keep_insertions=keep_insertions,
                    to_upper=to_upper,
                )
                yield desc, seq, pos
            desc = line.strip()[1:]
            seq = ""
        else:
            assert isinstance(seq, str)
            seq += line.strip()
    assert isinstance(seq, str) and isinstance(desc, str)
    seq, pos = convert_sequence_with_positions(
        seq,
        keep_gaps=keep_gaps,
        keep_insertions=keep_insertions,
        to_upper=to_upper,
    )
    yield desc, seq, pos


def fasta_generator(
    filepath,
    encoding=None,
    return_dict=False,
    keep_insertions=True,
    keep_gaps=True,
    to_upper=False,
):
    # if a return statement is used it closes the context manager too early
    # with gzread(filepath, encoding=encoding) as fin:
    with open(filepath, "r", encoding=encoding) as fin:
        yield from read_fasta_lines(
            fin, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
        )


def read_fasta(
    filepath,
    return_dict=False,
    encoding=None,
    keep_insertions=True,
    keep_gaps=True,
    to_upper=False,
):
    # TODO create a context manager
    gen = fasta_generator(
        filepath,
        keep_insertions=keep_insertions,
        keep_gaps=keep_gaps,
        to_upper=to_upper,
    )
    if return_dict:
        d = {n: s for n, s in gen}
        return d
    else:
        names, seqs = [], []
        for n, s in gen:
            names.append(n)
            seqs.append(s)
        return names, seqs


def first_sequence(filepath, **kwargs):
    g = fasta_generator(filepath)
    return next(g)


def filtered_fasta_sequences(
    fasta_file,
    n_seqs=None,
    max_len=None,
    min_len=20,
):
    labels, sequences = read_fasta(fasta_file)
    filtered_labels = []
    filtered_sequences = []
    for label, s in zip(labels, sequences):
        if len(s) <= (max_len or 1e8) and len(s) >= min_len and "X" not in s:
            # removing X shouldnt be necessary - unk_char in tokenisers.
            filtered_labels.append(label)
            filtered_sequences.append(s)

    n_seqs = n_seqs or len(sequences)
    return labels[:n_seqs], sequences[:n_seqs]


def output_fasta(names, seqs, filepath):
    with open(filepath, "w") as fout:
        for name, seq in zip(names, seqs):
            fout.write(">{}\n".format(name))
            fout.write(seq + "\n")


def read_msa(msa_file, msa_format):
    if msa_format == "a3m":
        return read_fasta(msa_file, keep_insertions=False, to_upper=True)
    elif msa_format == "gym":
        return read_fasta(msa_file, keep_insertions=True, to_upper=True)
    else:
        raise NotImplementedError(f"MSA format {msa_format} not supported.")
