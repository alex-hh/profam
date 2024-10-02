"""
References:
* trRosetta paper and code
* seqmodel
* evCouplings.

posssibly superseded by covar
"""
import string

import numpy as np
import torch

from src.sequence import fasta

aa_letters = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

aa_letters_wgap = ["-"] + aa_letters


def to_numeric(seqs, alphabet, unk_char="-"):
    """
    c.f. evcouplings code (alignment.py: uses vectorised getitem)
    https://github.com/debbiemarkslab/EVcouplings/blob/develop/evcouplings/align/alignment.py#L1084
    and
    https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    https://huggingface.co/transformers/_modules/transformers/tokenization_utils.html#PreTrainedTokenizer
    seqs should be a sequence of strings

    c.f. gremlin
    https://github.com/sokrypton/GREMLIN_CPP/blob/master/GREMLIN_TF_simple.ipynb

    for i, k in enumerate(alphabet):
        seqarr[seqarr==k] = i
    this is clear but wouldnt throw an exception if you had characters unaccounted for
    """
    try:
        if isinstance(seqs, str):
            seqs = [seqs]
        # N.B. astype object is critical here otherwise the automatic type
        # will not tolerate the addition of integers and will chop them up
        # an alternative is to use python to convert from |S1 (length one strings)
        # to uint8 (integer representations of the characters - i.e. Unicode ordinals)
        # as in GREMLIN implementation
        if isinstance(seqs, np.ndarray):
            assert seqs.ndim == 2
            seqarr = seqs.copy().astype(
                object
            )  # type is important (because we put ints in?)
        else:
            seqarr = np.asarray([np.array(list(s)) for s in seqs]).astype(object)
        for i, k in enumerate(alphabet):
            seqarr[seqarr == k] = i

        if unk_char is not None:
            for letter in string.ascii_uppercase:
                if letter not in alphabet:
                    seqarr[seqarr == letter] = alphabet.index(unk_char)

        seqarr = seqarr.astype(int)
    except Exception as e:
        raise e
        # print(seqs, e)
    # if a character is missing the exception is thrown at the object to int conversion stage so
    # assert never happens assert (seqarr < len(alphabet)).all(), "Some characters were not
    # converted; check alphabet"
    return seqarr


def to_one_hot(indices, n_feats):
    """
    can also use tf.one_hot
    """
    # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    # other options - convert indices into a list of [(batch, row, col)] inds
    # apply along axis a 1d indexing thing.
    # indices (n_seqs, L)
    assert isinstance(indices, np.ndarray) and indices.ndim == 2
    return np.eye(n_feats)[np.asarray(indices)]


def apc(Jij, zero_diag=True):
    """TODO check whether implementation is affected by whether J is pre-symmetrised.

    TODO check this agrees with julia...
    """
    Jij = Jij - (Jij.mean(-1, keepdims=True) * Jij.mean(-2, keepdims=True)) / Jij.mean()
    if zero_diag:
        Jij = Jij * (1 - np.eye(Jij.shape[-1]))
    return Jij


def frobenius(Wiajb, zero_diag=True):
    # zeroing diag by default following https://colab.research.google.com/github/sokrypton/seqmodels/blob/master/seqmodels.ipynb#scrollTo=D1dhVxJxY7C4  # noqa:E501
    frob = np.sqrt((Wiajb**2).sum((-3, -1)))  # L, L
    if zero_diag:
        frob = frob * (1 - np.eye(frob.shape[-2]))
    return frob


def score_contacts(Jiajb, exclude_gaps=True, gap_index=0):
    if exclude_gaps:
        assert gap_index == 0
        Jiajb = Jiajb[:, 1:, :, 1:]
    frob = frobenius(Jiajb)
    preds = apc(frob)
    return preds


def frequencies_to_covariances(fia, fiajb):
    """ia, iajb --> iajb"""
    return fiajb - fia[..., None, None] * fia[..., None, None, :, :]  # ..., L, K, L, K


def shrunk_inv_cov(Ciajb, penalty, zero_diag=True, method="cholesky"):
    """c.f. Dauparas: https://arxiv.org/pdf/1906.02598.pdf
    https://github.com/sokrypton/seqmodels/blob/master/seqmodels.ipynb
    and trRosetta (fast_dca fn.) https://github.com/gjoni/trRosetta/blob/037534cc52e6f7dc01f7d5045c2554430adcd878/network/utils.py  # noqa:E501

    penalty should be 4.5/sqrt(Meff)

    Notes on matrix inversion
    Cholesky decomposition should be more efficient way to compute inverse for a
    symmetric matrix (e.g. a covariance matrix)
     * torch has a cholesky_inverse function
     * numpy has vectorized lin alg routines if we want to batch
        https://numpy.org/doc/stable/reference/routines.linalg.html#linear-algebra-on-several-matrices-at-once

    Refs:
        https://scicomp.stackexchange.com/questions/22105/complexity-of-matrix-inversion-in-numpy
        https://math.stackexchange.com/questions/3829564/can-an-inverse-of-a-covariance-matrix-be-computed-faster-that-inverse-of-an-arbi
        https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
        https://stackoverflow.com/questions/44345340/efficiency-of-inverting-a-matrix-in-numpy-with-cholesky-decomposition
    """
    L, K, _, _ = Ciajb.shape
    Cij = Ciajb.reshape((L * K, L * K))
    cov_reg = Cij + np.eye(L * K) * penalty
    if method == "np":
        inv_cov = np.linalg.inv(cov_reg)
    elif method == "cholesky":
        inv_cov = torch.cholesky_inverse(
            torch.linalg.cholesky(torch.from_numpy(cov_reg))
        ).numpy()
    else:
        raise ValueError(f"Unsupported inv method {method}")
    if zero_diag:
        # c.f. eq 13 from dauparas, which results from using the zero diagonal form of W
        # the division by the term along the diagonal is important
        # TODO consider - shouldn't we also zero out all terms along diagonal iaib?, not just iaia?
        inv_cov = inv_cov / np.diagonal(inv_cov) - np.eye(L * K)
        inv_cov = 0.5 * (inv_cov + inv_cov.T)

    return inv_cov.reshape(L, K, L, K)


def add_single_pseudocounts(f, pseudocount_alpha=0.0):
    """Pseudocount alpha controls the mixing weight between the true
    distribution and a uniform distribution.

    n.b. that in practice large values of the pseudocounts (>0.5)
    are recommended for mfDCA/GaussDCA: c.f. Morcos, GaussDCA, Marks Supp
    Marks supp reports a totally different value to Morcos...
    PSICOV uses a small pseudocount of 1 (i.e. 1 raw COUNT in each thing)

    Pseudocounts for MI are discussed in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2672635/
    and in Bitbol pairing https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006401  # noqa:E501

    fi(a) <- (1−alpha)fi(a) + alpha/q
    fij(a,b) <- (1−alpha)fij(a,b) + alpha/q^2

    Ref Cocco, GaussDCA code.
    """
    assert 0 <= pseudocount_alpha and pseudocount_alpha <= 1
    L, K = f.shape
    return (1 - pseudocount_alpha) * f + pseudocount_alpha / K


def weighted_frequencies(X, w, pseudocount_alpha=0.0):
    """Weighted average of identity functions (I(xi=a))."""
    Meff = w.sum()
    f = (w[:, None, None] * X).sum(0) / Meff
    return f
    # return add_pseudocounts(f, pseudocount_alpha)


def weighted_pair_frequencies(X, w):
    M, L, K = X.shape
    X = np.sqrt(w[:, None] * X.reshape(M, -1))
    Meff = w.sum()
    # sum_m X_mk X_mj = X^T X
    return np.matmul(X.T, X).reshape(L, K, L, K) / Meff


def fast_cov(X, w=None):
    M, L, K = X.shape
    x = X.reshape(M, -1)

    weights = np.ones(M) if w is None else w
    # N.B. this sqrt might only be relevant if we are doing shrinkage?
    num_points = weights.sum() - np.sqrt(weights.mean())

    mean = (x * weights[:, None]).sum(axis=0, keepdims=True) / num_points
    x = (x - mean) * np.sqrt(weights[:, None])

    cov = (x.T @ x) / num_points
    return cov


def fast_dca(X, w=None, penalty=4.5, zero_diag=True):
    """Direct, trrosetta-style
    https://github.com/lucidrains/tr-rosetta-pytorch/blob/main/tr_rosetta_pytorch/utils.py
    https://www.pnas.org/doi/10.1073/pnas.1914677117#sec-3
    """
    M, L, K = X.shape
    weights = np.ones(M) if w is None else w
    cov = fast_cov(X, w=weights)
    # cov_reg = cov + np.eye(L*K) * penalty / np.sqrt(weights.sum())
    # inv_cov = np.linalg.inv(cov_reg)
    # return inv_cov.reshape((L, K, L, K))
    return shrunk_inv_cov(cov, penalty / np.sqrt(weights.sum()), zero_diag=zero_diag)


class MSANumeric:

    """N.B. in calculating weighted averages, we divide by sum of weights.

    If the weights are pre-normalised, this step is redundant.
    """

    def __init__(self, tokens, alphabet_size, gap_char):
        self.tokens = tokens  # n, L
        self.alphabet_size = alphabet_size
        self.gap_char = gap_char
        self._X = None
        self._weights = None

    def __len__(self):
        return self.tokens.shape[0]

    @property
    def X(self):
        if self._X is None:
            self._X = to_one_hot(self.tokens, self.alphabet_size)
        return self._X

    def sample(self, n):
        if n < self.M:
            indices = np.random.choice(self.M, n, replace=False)
            return MSANumeric(self.tokens[indices], self.alphabet_size, self.gap_char)
        else:
            return MSANumeric(self.tokens, self.alphabet_size, self.gap_char)

    @property
    def L(self):
        return self.tokens.shape[1]

    @property
    def M(self):
        return self.tokens.shape[0]

    def Meff(self, threshold=None):
        if self._weights is not None and threshold is None:
            w = self._weights
        else:
            kwargs = {"threshold": threshold} if threshold is not None else {}
            w = self.weights(**kwargs)
        return w.sum()

    def approx_Meff(self, N=1000, threshold=0.8):
        """Computing Meff is very expensive for large alignments.
        We could compute a MC approximation to it, by observing that
        Meff is N times the average weight.
        """
        sampled_ids = np.random.choice(self.M, min(N, self.M), replace=False)
        sampled_X = self.X[sampled_ids]
        # we compute pair identities to ALL other sequences in the alignment.
        sampled_pair_ids = np.einsum("mik,nik->mn", sampled_X, self.X, optimize=True)
        sampled_weights = 1 / ((sampled_pair_ids / self.L) > threshold).sum(-1)
        return self.M * sampled_weights.mean()

    def summary(self, calc_Meff=False):
        gap_frequencies = self.frequencies()[:, self.gap_char]
        d = {"M": self.M, "L": self.L, "below_50pc_gaps": (gap_frequencies < 0.5).sum()}
        if calc_Meff:
            d["Meff"] = self.Meff()
        return d

    def pair_ids(self):
        """N.B. optimize=True is REQUIRED to make this efficient."""
        # XmikXnik
        # TODO figure out if flattened X would be faster.
        return np.einsum("mik,nik->mn", self.X, self.X, optimize=True)

    def weights(self, threshold=0.8):
        identities = self.pair_ids()
        return 1 / ((identities / self.L) > threshold).sum(-1)

    def set_weights(self, threshold=0.8):
        w = self.weights(threshold=threshold)
        self._weights = w

    def frequencies(self, pseudocount_alpha=0.0):
        w = np.ones(self.X.shape[0]) if self._weights is None else self._weights
        fia = weighted_frequencies(self.X, w)
        return add_single_pseudocounts(fia, pseudocount_alpha=pseudocount_alpha)

    def pair_frequencies(self):
        w = np.ones(self.X.shape[0]) if self._weights is None else self._weights
        return weighted_pair_frequencies(self.X, w)

    def covariances(self):
        frequencies = self.frequencies()
        pair_frequencies = self.pair_frequencies()
        return frequencies_to_covariances(frequencies, pair_frequencies)

    def inv_cov(self, penalty=4.5, zero_diag=True, method="cholesky"):
        w = np.ones(self.X.shape[0]) if self._weights is None else self._weights
        cov = self.covariances()
        penalty = penalty / np.sqrt(w.sum())
        inv = shrunk_inv_cov(cov, penalty, zero_diag=zero_diag, method=method)
        return inv

    def contact_preds(self, penalty=4.5, zero_diag=True, exclude_gaps=True):
        """Return contact predictions using the shrunk covariance matrix inversion method."""
        inv = self.inv_cov(penalty=penalty, zero_diag=zero_diag)
        return score_contacts(inv, exclude_gaps=exclude_gaps)

    @classmethod
    def from_fasta(cls, fasta_filename, alphabet, max_seqs=None, drop_wt=False):
        _, seqs = fasta.read_fasta(fasta_filename)
        if drop_wt:
            seqs = seqs[1:]
        if max_seqs is not None:
            seqs = seqs[:max_seqs]
        tokens = to_numeric(seqs, alphabet)
        return cls(tokens, len(alphabet))

    @classmethod
    def from_a3m(cls, msa_filename, alphabet, max_seqs=None, drop_wt=False):
        _, seqs = fasta.read_fasta(msa_filename, keep_insertions=False)
        if drop_wt:
            seqs = seqs[1:]
        if max_seqs is not None:
            seqs = seqs[:max_seqs]
        tokens = to_numeric(seqs, alphabet)
        return cls(tokens, len(alphabet), alphabet.index("-"))

    @classmethod
    def from_sequences(cls, sequences, alphabet):
        assert all([len(s) == len(sequences[0]) for s in sequences])
        tokens = to_numeric(sequences, alphabet)
        return cls(tokens, len(alphabet), alphabet.index("-"))
