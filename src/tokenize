"""
Tokenization utilities for the Lenta News Embeddings Classification project.

This module provides:
- simple whitespace + regex tokenization for Word2Vec and Navec,
- morphological tokenization (lemma_POS) required for RusVectōrēs,
- corpus-building helpers.

"""

import re
from typing import List
import pymorphy3
import razdel


# --- Simple tokenization (Word2Vec, Navec) ---
def simple_tokenize(text: str) -> List[str]:
    """
    Basic tokenizer:
    - lowercases text,
    - removes non-alphanumeric characters,
    - splits by whitespace.

    Parameters
    ----------
    text : str
        Raw text.

    Returns
    -------
    List[str]
        List of tokens.
    """
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9ё ]+", " ", text)
    return text.split()


def build_corpus(texts) -> List[List[str]]:
    """
    Build corpus for Word2Vec/Navec:
    list of token lists.

    Parameters
    ----------
    texts : pd.Series
        Series of raw texts.

    Returns
    -------
    List[List[str]]
        Tokenized corpus.
    """
    return [simple_tokenize(t) for t in texts]


# --- Morphological tokenization (RusVectōrēs) ---
morph = pymorphy3.MorphAnalyzer()

def tokenize_with_pos(text: str) -> List[str]:
    """
    Morphological tokenizer for RusVectōrēs.

    Converts tokens into lemma_POS format:
    - москва → москва_PROPN
    - идти → идти_VERB

    Parameters
    ----------
    text : str
        Raw text.

    Returns
    -------
    List[str]
        List of lemma_POS tokens.
    """
    tokens = [t.text for t in razdel.tokenize(text)]
    result = []

    for t in tokens:
        # Only process alphabetic tokens
        if re.fullmatch(r"[a-zа-яё]+", t):
            p = morph.parse(t)[0]
            lemma = p.normal_form
            pos = p.tag.POS
            if pos:
                result.append(f"{lemma}_{pos}")
        else:
            # Keep punctuation or numbers as-is
            result.append(t)

    return result


def build_corpus_pos(texts) -> List[List[str]]:
    """
    Build corpus for RusVectōrēs using lemma_POS tokens.

    Parameters
    ----------
    texts : pd.Series

    Returns
    -------
    List[List[str]]
        Tokenized corpus with POS tags.
    """
    return [tokenize_with_pos(t) for t in texts]

