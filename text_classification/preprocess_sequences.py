import numpy as np
import pickle as pkl
from collections import Iterable, Counter


def main(data_dir, max_len=85):
    """
    Load, preprocess and save sequences for Train and Test using
    `scripts/preprocess_sequences.py` `shorten` function.

    Parameters
    ----------
    data_dir: str
        Directory containing `/Learn/` and `/Test/`.
    max_len: int
        Constraint on sequence length. Unimportant tokens are removed from
        sequences longer than max_len.
    """
    with open(data_dir + '/Learn/sequences.pkl', 'rb') as f:
        train_sequences = pkl.load(f)

    with open(data_dir + '/Test/sequences.pkl', 'rb') as f:
        test_sequences = pkl.load(f)

    processed_train, processed_test = shorten(train_sequences, test_sequences, max_len=max_len)

    with open(data_dir + '/Learn/processed_sequences.pkl', 'wb') as f:
        pkl.dump(processed_train, f)

    with open(data_dir + '/Test/processed_sequences.pkl', 'wb') as f:
        pkl.dump(processed_test, f)


def shorten(train_sequences, test_sequences, max_len=85):
    """
    Restrain sequences to a maximum length, by removing tokens with smallest
    tfidf values.

    Parameters
    ----------
    max_len: int
        Constraint on sequence length. Unimportant tokens are removed from
        sequences longer than max_len.
    """
    corpus = flatten(train_sequences)
    corpus_counter = Counter(corpus)
    tfidf_train_sequences = [my_tfidf(sequence, corpus_counter) for sequence in train_sequences]
    tfidf_test_sequences = [my_tfidf(sequence, corpus_counter) for sequence in test_sequences]
    processed_train = [list(np.array(sequence)[indices_small(sequence, max_len)]) for sequence, tfidf_sequence in zip(train_sequences, tfidf_train_sequences)]
    processed_test = [list(np.array(sequence)[indices_small(sequence, max_len)]) for sequence, tfidf_sequence in zip(test_sequences, tfidf_test_sequences)]
    return processed_train, processed_test


def flatten(x):
    """
    Flatten an array of sequences to create corpus.
    """
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def my_tfidf(sequence, corpus_counter):
    """
    Compute hand tfidf representation of sequence.
    If token does not belong to corpus, return 0.
    """
    sequence_counter = Counter(sequence)
    return [(sequence_counter[x] * 1.0 / corpus_counter[x]) if x in corpus_counter.keys() else 0.0 for x in sequence]


def indices_small(sequence, max_len=85):
    """
    Return indices of smallest values if size greater than max_len.

    Parameters
    ----------
    max_len: int
        Constraint on sequence length. Unimportant tokens are removed from
        sequences longer than max_len.
    """
    if len(sequence) > max_len:
        to_remove = sorted(sequence, reverse=True)[max_len + 1:]
        return [i for i, x in enumerate(sequence) if x not in to_remove]
    else:
        return range(len(sequence))
