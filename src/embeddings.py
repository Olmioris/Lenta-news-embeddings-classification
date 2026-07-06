"""
Embedding utilities for the Lenta News Embeddings Classification project.

This module provides:
- training Word2Vec embeddings,
- loading pretrained embeddings (Navec, RusVectōrēs),
- building sentence embeddings (mean pooling),
- building embedding matrices for train/val/test sets.

"""

import numpy as np
from typing import List
from gensim.models import Word2Vec, KeyedVectors
from navec import Navec

from .tokenize import simple_tokenize, tokenize_with_pos


# ============================
#   Word2Vec Training
# ============================

def train_word2vec(
    corpus: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    workers: int = 4,
    sg: int = 1,
    epochs: int = 10,
    seed: int = 42,
) -> Word2Vec:
    """
    Train a Word2Vec model on the provided corpus.

    Parameters
    ----------
    corpus : List[List[str]]
        Tokenized corpus.
    vector_size : int
        Dimensionality of word vectors.
    window : int
        Context window size.
    min_count : int
        Minimum word frequency.
    workers : int
        Number of CPU threads.
    sg : int
        1 = skip-gram, 0 = CBOW.
    epochs : int
        Number of training epochs.
    seed : int
        Random seed.

    Returns
    -------
    Word2Vec
        Trained Word2Vec model.
    """
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        seed=seed,
    )
    model.train(corpus, total_examples=len(corpus), epochs=epochs)
    return model


# ============================
#   Sentence Embeddings
# ============================

def sentence_embedding_mean(tokens: List[str], kv, dim: int) -> np.ndarray:
    """
    Compute mean-pooled sentence embedding.

    Parameters
    ----------
    tokens : List[str]
        Tokenized text.
    kv : KeyedVectors or Navec
        Embedding model.
    dim : int
        Embedding dimensionality.

    Returns
    -------
    np.ndarray
        Sentence embedding vector.
    """
    vecs = [kv[t] for t in tokens if t in kv]

    if not vecs:
        return np.zeros(dim)

    return np.mean(vecs, axis=0)


def build_embedding_matrix(texts, kv, dim: int) -> np.ndarray:
    """
    Build embedding matrix using simple tokenization.

    Parameters
    ----------
    texts : pd.Series
        Raw texts.
    kv : KeyedVectors or Navec
        Embedding model.
    dim : int
        Embedding dimensionality.

    Returns
    -------
    np.ndarray
        Matrix of shape (N, dim).
    """
    return np.vstack([
        sentence_embedding_mean(simple_tokenize(t), kv, dim)
        for t in texts
    ])


def build_embedding_matrix_rus(texts, kv, dim: int) -> np.ndarray:
    """
    Build embedding matrix for RusVectōrēs using lemma_POS tokens.

    Parameters
    ----------
    texts : pd.Series
        Raw texts.
    kv : KeyedVectors
        RusVectōrēs model.
    dim : int
        Embedding dimensionality.

    Returns
    -------
    np.ndarray
        Matrix of shape (N, dim).
    """
    return np.vstack([
        sentence_embedding_mean(tokenize_with_pos(t), kv, dim)
        for t in texts
    ])


# ============================
#   Pretrained Embeddings
# ============================

def load_rusvectores(path: str) -> KeyedVectors:
    """
    Load RusVectōrēs pretrained embeddings.

    Parameters
    ----------
    path : str
        Path to .vec.gz file.

    Returns
    -------
    KeyedVectors
        Loaded embedding model.
    """
    return KeyedVectors.load_word2vec_format(path)


def load_navec(path: str) -> Navec:
    """
    Load Navec pretrained embeddings.

    Parameters
    ----------
    path : str
        Path to .tar file.

    Returns
    -------
    Navec
        Loaded Navec model.
    """
    return Navec.load(path)

