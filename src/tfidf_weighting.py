"""
TF‑IDF weighted sentence embeddings for the Lenta News Embeddings Classification project.

This module provides:
- TF‑IDF vectorizer initialization,
- TF‑IDF‑weighted sentence embeddings,
- building embedding matrices for train/val/test.

Used to enhance classical embeddings (Word2Vec, Navec, RusVectōrēs) by weighting each word vector with its TF‑IDF score.
"""

import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer

from .tokenize import simple_tokenize


# ============================
#   TF‑IDF Vectorizer
# ============================

def build_tfidf_vectorizer(texts):
    """
    Fit TF‑IDF vectorizer on training texts.

    Parameters
    ----------
    texts : pd.Series
        Raw training texts.

    Returns
    -------
    (tfidf, feature_names)
        Fitted TF‑IDF vectorizer and array of feature names.
    """
    tfidf = TfidfVectorizer(
        tokenizer=simple_tokenize,
        lowercase=False
    )
    tfidf.fit(texts)
    feature_names = tfidf.get_feature_names_out()
    return tfidf, feature_names


# ============================
#   TF‑IDF Weighted Embeddings
# ============================

def sentence_embedding_tfidf(
    text: str,
    kv,
    dim: int,
    tfidf,
    feature_names
) -> np.ndarray:
    """
    Compute TF‑IDF weighted sentence embedding.

    Each word vector is multiplied by its TF‑IDF weight:
        embedding = sum(w_i * v_i) / sum(w_i)

    Parameters
    ----------
    text : str
        Raw text.
    kv : KeyedVectors or Navec
        Embedding model.
    dim : int
        Embedding dimensionality.
    tfidf : TfidfVectorizer
        Fitted TF‑IDF vectorizer.
    feature_names : array
        Vocabulary of TF‑IDF vectorizer.

    Returns
    -------
    np.ndarray
        TF‑IDF weighted sentence embedding.
    """
    tokens = simple_tokenize(text)

    tfidf_vec = tfidf.transform([text])
    tfidf_dict: Dict[str, float] = {
        feature_names[i]: tfidf_vec[0, i]
        for i in tfidf_vec.nonzero()[1]
    }

    vecs = []
    weights = []

    for t in tokens:
        if t in kv and t in tfidf_dict:
            w = tfidf_dict[t]
            vecs.append(kv[t] * w)
            weights.append(w)

    if not vecs:
        return np.zeros(dim)

    return np.sum(vecs, axis=0) / np.sum(weights)


def build_embedding_matrix_tfidf(
    texts,
    kv,
    dim: int,
    tfidf,
    feature_names
) -> np.ndarray:
    """
    Build TF‑IDF weighted embedding matrix.

    Parameters
    ----------
    texts : pd.Series
        Raw texts.
    kv : KeyedVectors or Navec
        Embedding model.
    dim : int
        Embedding dimensionality.
    tfidf : TfidfVectorizer
        Fitted TF‑IDF vectorizer.
    feature_names : array
        Vocabulary of TF‑IDF vectorizer.

    Returns
    -------
    np.ndarray
        Matrix of shape (N, dim).
    """
    return np.vstack([
        sentence_embedding_tfidf(t, kv, dim, tfidf, feature_names)
        for t in texts
    ])

