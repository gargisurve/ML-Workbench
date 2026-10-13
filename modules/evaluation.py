import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)


# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

def classification_metrics(y_test, predictions):

    return {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    }


def get_confusion_matrix(y_test, predictions):

    return confusion_matrix(
        y_test,
        predictions
    )


# --------------------------------------------------
# REGRESSION
# --------------------------------------------------

def regression_metrics(y_test, predictions):

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    return {
        "R² Score": r2_score(
            y_test,
            predictions
        ),

        "MAE": mean_absolute_error(
            y_test,
            predictions
        ),

        "MSE": mse,

        "RMSE": rmse
    }


# --------------------------------------------------
# CLUSTERING
# --------------------------------------------------

def clustering_metrics(X, labels):

    # Ignore DBSCAN noise points
    valid = labels != -1

    unique_clusters = set(labels[valid])

    if len(unique_clusters) < 2:

        return {
            "Silhouette Score": None,
            "Number of Clusters": len(unique_clusters)
        }

    score = silhouette_score(
        X[valid],
        labels[valid]
    )

    return {
        "Silhouette Score": score,
        "Number of Clusters": len(unique_clusters)
    }