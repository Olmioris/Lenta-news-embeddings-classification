"""
Model training and evaluation utilities for the Lenta News Embeddings Classification project.

This module provides:
- training Logistic Regression models,
- evaluating models on validation/test sets,
- computing accuracy and macro F1,
- unified interface for model comparison.

"""

import numpy as np
from typing import Tuple, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# ============================
#   Training
# ============================

def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """
    Train Logistic Regression classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    max_iter : int
        Maximum number of iterations.
    random_state : int
        Random seed.

    Returns
    -------
    LogisticRegression
        Trained classifier.
    """
    clf = LogisticRegression(
        max_iter=max_iter,
        n_jobs=-1,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf


# ============================
#   Evaluation
# ============================

def evaluate_model(
    clf: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[float, float]:
    """
    Evaluate model using accuracy and macro F1.

    Parameters
    ----------
    clf : LogisticRegression
        Trained classifier.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True labels.

    Returns
    -------
    (accuracy, macro_f1)
        Evaluation metrics.
    """
    pred = clf.predict(X)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    return acc, f1


# ============================
#   Prediction Helper
# ============================

def predict(clf: LogisticRegression, X: np.ndarray):
    """
    Predict labels for given feature matrix.

    Parameters
    ----------
    clf : LogisticRegression
        Trained classifier.
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    np.ndarray
        Predicted labels.
    """
    return clf.predict(X)
