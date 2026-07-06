"""
Data loading and dataset preparation utilities for the Lenta News Embeddings Classification project.

This module:
- loads the Lenta.ru dataset via Corus,
- removes rare classes (<2 samples),
- performs stratified train/val/test split (60/20/20),
- provides reusable functions for notebooks and scripts.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple

from corus import load_lenta
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_lenta_df(path: str, limit: int = 100_000) -> pd.DataFrame:
    """
    Load Lenta.ru dataset using Corus iterator.

    Parameters
    ----------
    path : str
        Path to .csv.gz file.
    limit : int
        Maximum number of records to load.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: title, text, topic.
    """
    records = load_lenta(path)
    data = []

    for i, record in enumerate(records):
        if i >= limit:
            break
        data.append({
            "title": record.title,
            "text": record.text,
            "topic": record.topic,
        })

    df = pd.DataFrame(data)
    df = df.dropna(subset=["text", "topic"])
    return df


def remove_rare_classes(df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    """
    Remove classes with fewer than `min_count` samples.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    min_count : int
        Minimum number of samples per class.

    Returns
    -------
    pd.DataFrame
        Filtered dataset with rare classes removed.
    """
    topic_counts = df["topic"].value_counts()
    df = df[df["topic"].isin(topic_counts[topic_counts >= min_count].index)]
    df = df.reset_index(drop=True)
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train/validation/test (60/20/20).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    test_size : float
        Fraction of data for test split.
    val_size : float
        Fraction of data for validation split.
    random_state : int
        Random seed.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["topic"],
        random_state=random_state,
    )

    rel_val = val_size / (1 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=rel_val,
        stratify=train_val["topic"],
        random_state=random_state,
    )

    return train, val, test


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, folder: str = "."):
    """
    Save train/val/test splits into CSV files.

    Parameters
    ----------
    train_df : pd.DataFrame
    val_df : pd.DataFrame
    test_df : pd.DataFrame
    folder : str
        Directory to save files.
    """
    train_df.to_csv(os.path.join(folder, "train.csv"), index=False)
    val_df.to_csv(os.path.join(folder, "val.csv"), index=False)
    test_df.to_csv(os.path.join(folder, "test.csv"), index=False)

