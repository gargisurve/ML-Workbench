import pandas as pd
import numpy as np

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)


# ==================================================
# K-MEANS
# ==================================================

def run_kmeans(X, n_clusters):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    return model, labels


# ==================================================
# HIERARCHICAL CLUSTERING
# ==================================================

def run_hierarchical(X, n_clusters):

    model = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    labels = model.fit_predict(X)

    return model, labels


# ==================================================
# DBSCAN
# ==================================================

def run_dbscan(X, eps, min_samples):

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X)

    return model, labels


# ==================================================
# CLUSTER SUMMARY
# ==================================================

def get_cluster_summary(
    display_data,
    labels,
    x_column,
    y_column
):

    plot_df = display_data.copy()

    plot_df["Cluster"] = labels

    # Remove DBSCAN noise for centroid calculations
    valid_df = plot_df[
        plot_df["Cluster"] != -1
    ]

    summary = (
        valid_df
        .groupby("Cluster")
        .agg(
            Observations=("Cluster", "size"),
            X_Mean=(x_column, "mean"),
            Y_Mean=(y_column, "mean")
        )
        .reset_index()
    )

    summary["Cluster"] = (
        summary["Cluster"] + 1
    )

    summary = summary.rename(
        columns={
            "Cluster": "Cluster",
            "X_Mean": f"Avg {x_column}",
            "Y_Mean": f"Avg {y_column}"
        }
    )

    return summary


# ==================================================
# PREPARE CLUSTER PLOT DATA
# ==================================================

def prepare_cluster_plot_data(
    display_data,
    labels,
    x_column,
    y_column
):

    plot_df = display_data[
        [x_column, y_column]
    ].copy()

    plot_df["Cluster"] = labels

    plot_df["Cluster Label"] = (
        plot_df["Cluster"]
        .apply(
            lambda x:
            "Noise"
            if x == -1
            else f"Cluster {x + 1}"
        )
    )

    return plot_df


# ==================================================
# CALCULATE VISUAL CENTROIDS
# ==================================================

def calculate_visual_centroids(
    display_data,
    labels,
    x_column,
    y_column
):

    plot_df = display_data[
        [x_column, y_column]
    ].copy()

    plot_df["Cluster"] = labels

    # Ignore DBSCAN noise
    plot_df = plot_df[
        plot_df["Cluster"] != -1
    ]

    if plot_df.empty:

        return pd.DataFrame(
            columns=[
                x_column,
                y_column,
                "Cluster Label"
            ]
        )

    centroids = (
        plot_df
        .groupby("Cluster")[
            [x_column, y_column]
        ]
        .mean()
        .reset_index()
    )

    centroids["Cluster Label"] = (
        centroids["Cluster"]
        .apply(
            lambda x:
            f"Centroid {x + 1}"
        )
    )

    return centroids