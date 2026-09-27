import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def calculate_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, float]:
    return {
        "top_1_accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
):
    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1

    return matrix