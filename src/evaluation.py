"""
Evaluation utilities for classical and neural sentiment models.

Includes:
- metric calculation for classical models
- metric calculation for neural models
- classification report
- confusion-matrix visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


def evaluate_model(
    name,
    y_true,
    y_pred,
    figure_path=None
):
    """
    Evaluate a classical machine-learning model.

    Prints:
    - Accuracy
    - F1 score
    - Precision
    - Recall
    - Classification report

    Optionally saves the confusion matrix to disk.

    Returns
    -------
    results : dict
        Dictionary containing the model name and evaluation metrics.
    """

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Print metrics
    print(f"\n{'-' * 60}")
    print(f"Model: {name}")
    print(f"{'-' * 60}")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    # Save instead of opening an interactive window
    if figure_path is not None:
        plt.savefig(
            figure_path,
            dpi=150,
            bbox_inches="tight"
        )

        print(
            f"Confusion matrix saved to: "
            f"{figure_path}"
        )

    plt.close()

    return {
        "Model": name,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }


def evaluate_neural(
    model,
    dataloader,
    device,
    name="Model",
    figure_path=None
):
    """
    Evaluate a neural sentiment classifier.

    Converts raw logits into probabilities using sigmoid,
    applies a threshold of 0.5, calculates classification metrics,
    and optionally saves the confusion matrix.

    Returns
    -------
    results : dict
        Evaluation metrics.

    preds : np.ndarray
        Predicted sentiment labels.

    labels : np.ndarray
        Ground-truth sentiment labels.
    """

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, lengths, y in dataloader:
            X = X.to(device)
            lengths = lengths.cpu()

            logits = model(X, lengths).squeeze(1)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).long()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y.numpy().astype(int))

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    print(f"\n=== {name} ===")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    if figure_path is not None:
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {figure_path}")

    plt.close()

    results = {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    return results, preds, labels