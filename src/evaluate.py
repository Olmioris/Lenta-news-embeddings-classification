"""
Evaluation utilities for the Lenta News Embeddings Classification project.

This module provides:
- evaluation of the best model on the test set,
- generation of classification report.

Used in the final notebook (03_tfidf_weighting_and_evaluation.ipynb).
"""

import numpy as np
from typing import Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# ============================
#   Final Test Evaluation
# ============================

def evaluate_on_test(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[float, float, str]:
    """
    Evaluate model on the test set.

    Parameters
    ----------
    clf : LogisticRegression
        Trained classifier.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True test labels.

    Returns
    -------
    (accuracy, macro_f1, report)
        Final evaluation metrics and classification report.
    """
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    report = classification_report(y_test, pred)

    return acc, f1, report
