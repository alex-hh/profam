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
            s = re.sub(r"\.", "", s)
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


def read_fasta_sequences(lines, keep_gaps=True, keep_insertions=True, to_upper=False):
    """
    From esm
    Works for fasta and a2m/a3m
    """
    iterator = read_fasta_lines(
        lines, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
    )
    for _, seq in iterator:
        yield seq


def convert_sequence_with_positions(
    seq,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
    use_msa_pos: bool = True,
):
    """
    Get positions relative to sequence.
    For alignments, if use_msa_pos is True, the positions are relative to the alignment columns
    (match states). Insertions have the same position index as the previous match state.

    If use_msa_pos is False, or the sequence is unaligned,
    positions are relative to the retained sequence - ignored insertions dont contribute

    For both raw and aligned sequences, the first non-insertions should have position 1.

    N.B. currently there is ambiguity between position encoding for a gap then insert
    and a match state. we require a binary mask to resolve.
    """
    match_index = 0  # 0 for inserts before first match state
    positions = []
    is_match = []
    sequence = ""

    # if not use_msa_pos:
    #     return seq, list(range(1, len(seq+1))), [True] * len(seq)

    if keep_insertions:
        assert to_upper, "If keeping insertions should convert to upper case"
    for aa in seq:
        if keep_gaps or aa != "-":
            if aa == ".":
                # dont keep gaps in insert columns: we can modify later if we ever want to use
                continue
            # at this point we have any amino acid character (match or insert) or a match gap
            # TODO: check for valid characters
            upper = aa.upper()
            if upper == aa or keep_insertions:
                # increment first so that insert corresponds to prev match state
                if upper == aa and aa != ".":  # includes case where aa is "-"
                    match_index += 1
                    is_match.append(True)
                else:
                    assert aa != "."
                    # insertion
                    if not use_msa_pos:
                        match_index += 1
                    is_match.append(False)
                positions.append(match_index)
                sequence += upper
            # otherwise we're not keeping insertions in which case we pass

        elif aa == "-":
            if use_msa_pos:
                match_index += 1  # keep_gaps is False so we dont add to sequence but still increment match_index

    assert len(positions) == len(
        sequence
    ), f"positions length {len(positions)} != sequence length {len(sequence)}"
    assert len(sequence) == len(
        is_match
    ), f"sequence length {len(sequence)} != is_match length {len(is_match)}"
    return sequence, positions, is_match


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
