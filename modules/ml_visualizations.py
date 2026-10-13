import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    roc_curve,
    auc
)


# ==================================================
# ROC CURVE
# ==================================================

def create_roc_curve(
    model,
    X_test,
    y_test
):

    # ROC is most straightforward for binary classification

    unique_classes = y_test.nunique()

    if unique_classes != 2:

        return None

    try:

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

        elif hasattr(
            model,
            "decision_function"
        ):

            probabilities = (
                model.decision_function(
                    X_test
                )
            )

        else:

            return None

        classes = list(
            model.classes_
        )

        positive_class = classes[1]

        y_binary = (
            y_test == positive_class
        ).astype(int)

        fpr, tpr, _ = roc_curve(
            y_binary,
            probabilities
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {roc_auc:.3f})"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(
                    dash="dash"
                )
            )
        )

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500
        )

        return fig

    except Exception:

        return None


# ==================================================
# RESIDUAL DISTRIBUTION
# ==================================================

def residual_distribution(
    y_test,
    predictions
):

    residuals = (
        np.array(y_test)
        - np.array(predictions)
    )

    df = pd.DataFrame({
        "Residual": residuals
    })

    fig = px.histogram(
        df,
        x="Residual",
        nbins=40,
        title="Residual Distribution"
    )

    fig.update_layout(
        height=500
    )

    return fig


# ==================================================
# ABSOLUTE ERROR DISTRIBUTION
# ==================================================

def error_distribution(
    y_test,
    predictions
):

    errors = np.abs(
        np.array(y_test)
        - np.array(predictions)
    )

    df = pd.DataFrame({
        "Absolute Error": errors
    })

    fig = px.histogram(
        df,
        x="Absolute Error",
        nbins=40,
        title="Prediction Error Distribution"
    )

    fig.update_layout(
        height=500
    )

    return fig


# ==================================================
# CLUSTER PROFILE
# ==================================================

def cluster_profile(
    display_data,
    labels
):

    numeric_data = (
        display_data
        .select_dtypes(
            include=np.number
        )
        .copy()
    )

    if numeric_data.empty:

        return None

    numeric_data["Cluster"] = labels

    profile = (
        numeric_data
        .groupby("Cluster")
        .mean()
    )

    # Normalize each feature for visual comparison
    profile_scaled = (
        profile - profile.min()
    ) / (
        profile.max()
        - profile.min()
        + 1e-9
    )

    profile_scaled.index = [
        (
            "Noise"
            if cluster == -1
            else f"Cluster {cluster + 1}"
        )
        for cluster in profile_scaled.index
    ]

    fig = px.imshow(
        profile_scaled.T,
        text_auto=".2f",
        aspect="auto",
        title="Cluster Feature Profile",
        labels={
            "x": "Cluster",
            "y": "Feature",
            "color": "Relative Level"
        }
    )

    fig.update_layout(
        height=max(
            450,
            len(profile.columns) * 35
        )
    )

    return fig