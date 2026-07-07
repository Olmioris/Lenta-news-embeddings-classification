"""
Final evaluation utilities for the Lenta News Embeddings Classification project.

This module provides:
- evaluation of the best model on the test set,
- generation of classification report,
- confusion matrix plotting,
- saving metrics into reports/ directory.

Used in the final notebook (03_tfidf_weighting_and_evaluation.ipynb).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
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


# ============================
#   Save Metrics
# ============================

def save_metrics(
    results: Dict[str, Tuple[float, float]],
    best_model_name: str,
    test_acc: float,
    test_f1: float,
    report: str,
    path: str = "reports/dz2_metrics.txt"
):
    """
    Save all validation and test metrics into a text file.

    Parameters
    ----------
    results : dict
        Validation metrics for all models.
    best_model_name : str
        Name of the best model.
    test_acc : float
        Test accuracy.
    test_f1 : float
        Test macro F1.
    report : str
        Classification report.
    path : str
        Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("Validation metrics:\n")
        for name, (acc, f1) in results.items():
            f.write(f"{name}: accuracy={acc:.4f}, macro_f1={f1:.4f}\n")

        f.write("\nBest model on validation: " + best_model_name + "\n\n")

        f.write("Test metrics:\n")
        f.write(f"accuracy={test_acc:.4f}, macro_f1={test_f1:.4f}\n\n")

        f.write("Classification report:\n")
        f.write(report)
