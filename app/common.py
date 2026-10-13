import io
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from modules.data_processing import (
    load_data,
    get_dataset_info,
    get_numeric_columns,
    get_categorical_columns,
    handle_missing_values,
    encode_features,
    scale_features,
    split_data
)

from modules.eda import (
    histogram,
    boxplot,
    density_plot,
    scatter_plot,
    correlation_heatmap,
    categorical_bar,
    pie_chart,
    missing_values_chart,
    unique_values_chart
)

from modules.classification import (
    get_classification_models,
    train_classification_model
)

from modules.regression import (
    get_regression_models,
    train_regression_model
)

from modules.clustering import (
    run_kmeans,
    run_hierarchical,
    run_dbscan,
    prepare_cluster_plot_data,
    calculate_visual_centroids,
    get_cluster_summary
)

from modules.evaluation import (
    classification_metrics,
    get_confusion_matrix,
    regression_metrics,
    clustering_metrics
)

from modules.ml_visualizations import (
    create_roc_curve,
    residual_distribution,
    error_distribution,
    cluster_profile
)

from modules.ui import (
    metric_row,
    create_progress,
    update_progress,
    finish_progress,
    require_dataset,
    require_preprocessing,
    model_success
)


# ==================================================
# GLOBAL WORKBENCH STYLE
# ==================================================

def apply_workbench_style():

    st.markdown(
        """
        <style>

        /* ==================================================
           APPLICATION
        ================================================== */

        .stApp {
            background: #F5F7FA !important;
        }

        section[data-testid="stMain"] {
            background: #F5F7FA !important;
        }

        section[data-testid="stMain"] .block-container {
            max-width: 1220px !important;
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        footer {
            visibility: hidden !important;
        }


        /* ==================================================
           MAIN TEXT
        ================================================== */

        section[data-testid="stMain"] h1,
        section[data-testid="stMain"] h2,
        section[data-testid="stMain"] h3,
        section[data-testid="stMain"] h4 {
            color: #172033 !important;
            opacity: 1 !important;
        }

        section[data-testid="stMain"] h1 {
            font-size: 2.25rem !important;
            font-weight: 750 !important;
            letter-spacing: -0.7px !important;
        }

        section[data-testid="stMain"] h2 {
            font-size: 1.45rem !important;
            font-weight: 700 !important;
        }

        section[data-testid="stMain"] h3 {
            font-size: 1.1rem !important;
            font-weight: 650 !important;
        }

        section[data-testid="stMain"] p,
        section[data-testid="stMain"] label,
        section[data-testid="stMain"] li {
            color: #26364D !important;
            opacity: 1 !important;
        }

        section[data-testid="stMain"]
        [data-testid="stCaptionContainer"],
        section[data-testid="stMain"]
        [data-testid="stCaptionContainer"] * {
            color: #718096 !important;
            opacity: 1 !important;
        }


        /* ==================================================
           SIDEBAR
        ================================================== */

        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {
            background: #FFFFFF !important;
        }

        section[data-testid="stSidebar"] {
            min-width: 285px !important;
            max-width: 285px !important;
            border-right: 1px solid #DDE3EB !important;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.45rem !important;
            padding-left: 1.15rem !important;
            padding-right: 1.05rem !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #172033 !important;
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"]
        [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"]
        [data-testid="stCaptionContainer"] * {
            color: #748197 !important;
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: #E1E6ED !important;
        }


        /* ==================================================
           SIDEBAR NAVIGATION
        ================================================== */

        section[data-testid="stSidebar"]
        div[role="radiogroup"] {
            gap: 2px !important;
        }

        section[data-testid="stSidebar"]
        div[role="radiogroup"] > label {
            background: transparent !important;
            color: #3F4D63 !important;
            border-radius: 8px !important;
            padding: 8px 10px !important;
            margin: 1px 0 !important;
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"]
        div[role="radiogroup"] > label p,
        section[data-testid="stSidebar"]
        div[role="radiogroup"] > label span {
            color: #3F4D63 !important;
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"]
        div[role="radiogroup"] > label:hover {
            background: #F2F5FA !important;
        }

        section[data-testid="stSidebar"]
        div[role="radiogroup"] > label:has(input:checked) {
            background: #E9EFFF !important;
            color: #2448C7 !important;
            font-weight: 650 !important;
        }

        section[data-testid="stSidebar"]
        div[role="radiogroup"] > label:has(input:checked) p,
        section[data-testid="stSidebar"]
        div[role="radiogroup"] > label:has(input:checked) span {
            color: #2448C7 !important;
            opacity: 1 !important;
        }


        /* ==================================================
           SIDEBAR FILE UPLOADER
        ================================================== */

        section[data-testid="stSidebar"]
        [data-testid="stFileUploader"] {
            color: #334155 !important;
            opacity: 1 !important;
        }

        /* Uploader label */
        section[data-testid="stSidebar"]
        [data-testid="stFileUploader"] label,
        section[data-testid="stSidebar"]
        [data-testid="stFileUploader"] label p {
            color: #334155 !important;
            -webkit-text-fill-color: #334155 !important;
            opacity: 1 !important;
        }

        /* Drop area */
        section[data-testid="stSidebar"]
        [data-testid="stFileUploaderDropzone"] {
            background: #FAFBFD !important;
            border: 1px dashed #C9D3E0 !important;
            border-radius: 8px !important;
            opacity: 1 !important;
        }

        /* Dropzone text */
        section[data-testid="stSidebar"]
        [data-testid="stFileUploaderDropzone"] * {
            color: #536176 !important;
            -webkit-text-fill-color: #536176 !important;
            opacity: 1 !important;
        }

        /* Browse button */
        section[data-testid="stSidebar"]
        [data-testid="stFileUploaderDropzone"] button {
            background: #FFFFFF !important;
            color: #334155 !important;
            -webkit-text-fill-color: #334155 !important;
            border: 1px solid #D4DCE7 !important;
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"]
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #F4F7FF !important;
            color: #315BEA !important;
            -webkit-text-fill-color: #315BEA !important;
            border-color: #315BEA !important;
        }

        /* Uploaded file container */
        section[data-testid="stSidebar"]
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            opacity: 1 !important;
            color: #334155 !important;
        }

        /* Uploaded filename and all nested text */
        section[data-testid="stSidebar"]
        [data-testid="stFileUploader"]
        [data-testid="stFileUploaderFile"] * {
            color: #334155 !important;
            -webkit-text-fill-color: #334155 !important;
            opacity: 1 !important;
        }

        /* File size / metadata */
        section[data-testid="stSidebar"]
        [data-testid="stFileUploader"]
        small,
        section[data-testid="stSidebar"]
        [data-testid="stFileUploader"]
        small * {
            color: #718096 !important;
            -webkit-text-fill-color: #718096 !important;
            opacity: 1 !important;
        }

        /* File icon */
        section[data-testid="stSidebar"]
        [data-testid="stFileUploader"]
        svg {
            color: #718096 !important;
            fill: #718096 !important;
            opacity: 1 !important;
        }


        /* ==================================================
           METRICS
        ================================================== */

        div[data-testid="stMetric"] {
            background: #FFFFFF !important;
            border: 1px solid #E0E6EE !important;
            border-radius: 9px !important;
            padding: 16px 18px !important;
            box-shadow: 0 2px 5px rgba(30, 45, 70, 0.025) !important;
        }

        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] * {
            color: #718096 !important;
            opacity: 1 !important;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {
            color: #172033 !important;
            opacity: 1 !important;
        }


        /* ==================================================
           BUTTONS
        ================================================== */

        .stButton > button {
            background: #FFFFFF !important;
            color: #334155 !important;
            border: 1px solid #D4DCE7 !important;
            border-radius: 7px !important;
            min-height: 40px !important;
            font-weight: 600 !important;
        }

        .stButton > button:hover {
            background: #F4F7FF !important;
            border-color: #315BEA !important;
            color: #315BEA !important;
        }

        .stButton > button[kind="primary"] {
            background: #315BEA !important;
            border-color: #315BEA !important;
            color: #FFFFFF !important;
        }

        .stButton > button[kind="primary"]:hover {
            background: #2448C7 !important;
            border-color: #2448C7 !important;
            color: #FFFFFF !important;
        }


        /* ==================================================
           DOWNLOAD BUTTON
        ================================================== */

        .stDownloadButton > button {
            background: #FFFFFF !important;
            color: #334155 !important;
            border: 1px solid #D4DCE7 !important;
            border-radius: 7px !important;
            min-height: 40px !important;
            font-weight: 600 !important;
        }

        .stDownloadButton > button:hover {
            background: #F4F7FF !important;
            border-color: #315BEA !important;
            color: #315BEA !important;
        }


        /* ==================================================
           SELECTBOXES
        ================================================== */

        div[data-baseweb="select"] {
            color: #172033 !important;
        }

        div[data-baseweb="select"] > div {
            background: #FFFFFF !important;
            border: 1px solid #D4DCE7 !important;
            border-radius: 7px !important;
            color: #172033 !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] [role="combobox"] {
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] input::placeholder {
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] svg {
            fill: #536176 !important;
            color: #536176 !important;
        }

        /* Dropdown */
        div[data-baseweb="popover"] {
            background: #FFFFFF !important;
        }

        div[data-baseweb="popover"] li {
            color: #172033 !important;
            background: #FFFFFF !important;
        }

        div[data-baseweb="popover"] li:hover {
            background: #F1F5FA !important;
            color: #315BEA !important;
        }


        /* ==================================================
           TEXT INPUTS
        ================================================== */

        input,
        textarea {
            background: #FFFFFF !important;
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            border-color: #D4DCE7 !important;
            border-radius: 7px !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #718096 !important;
            -webkit-text-fill-color: #718096 !important;
            opacity: 1 !important;
        }

        input:focus,
        textarea:focus {
            border-color: #315BEA !important;
            box-shadow: 0 0 0 1px #315BEA !important;
        }


        /* ==================================================
           DATAFRAME / EXPANDERS
        ================================================== */

        div[data-testid="stDataFrame"] {
            background: #FFFFFF !important;
            border: 1px solid #E0E6EE !important;
            border-radius: 8px !important;
            overflow: hidden;
        }

        details {
            background: #FFFFFF !important;
            border: 1px solid #E0E6EE !important;
            border-radius: 8px !important;
        }


        /* ==================================================
           ALERTS
        ================================================== */

        div[data-testid="stAlert"] {
            border-radius: 8px !important;
        }

        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span {
            opacity: 1 !important;
        }


        /* ==================================================
           TABS / DIVIDERS / PROGRESS
        ================================================== */

        button[data-baseweb="tab"] {
            color: #718096 !important;
            font-weight: 600 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #315BEA !important;
        }

        hr {
            border: none !important;
            border-top: 1px solid #E1E6ED !important;
            margin: 22px 0 !important;
        }

        div[data-testid="stProgress"] > div > div {
            background: #315BEA !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# OVERFITTING ALERT
# ==================================================

def show_overfitting_alert(model_type):

    train_metrics = st.session_state.get("train_metrics")
    test_metrics = st.session_state.get("metrics")

    if not train_metrics or not test_metrics:
        return

    if model_type == "Classification":

        train_score = train_metrics.get("Accuracy")
        test_score = test_metrics.get("Accuracy")

        metric_name = "accuracy"
        threshold = 0.05

    elif model_type == "Regression":

        train_score = train_metrics.get("R² Score")
        test_score = test_metrics.get("R² Score")

        metric_name = "R² score"
        threshold = 0.10

    else:
        return

    if train_score is None or test_score is None:
        return

    gap = train_score - test_score

    if gap >= threshold:

        st.warning(
            f"Potential overfitting detected. Training {metric_name} is "
            f"{train_score:.2%} while test {metric_name} is "
            f"{test_score:.2%} (gap: {gap:.2%}). The model may be "
            "memorizing the training data instead of generalizing well."
        )

    elif model_type == "Classification" and test_score >= 0.98:

        st.info(
            "Very high test performance. Check for duplicate rows, "
            "target leakage, or features that directly reveal the target."
        )

    elif model_type == "Regression" and test_score >= 0.95:

        st.info(
            "Very high test performance. Verify that no target leakage "
            "or duplicate information is present."
        )


# ==================================================
# CONFUSION MATRIX
# ==================================================

def show_confusion_matrix(y_true, y_pred):

    labels = list(
        pd.unique(
            pd.concat(
                [
                    pd.Series(y_true),
                    pd.Series(y_pred)
                ]
            )
        )
    )

    if len(labels) == 2:

        try:
            labels = sorted(labels)

        except TypeError:

            labels = sorted(
                labels,
                key=lambda value: str(value)
            )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    if len(labels) == 2:

        annotations = [
            [
                f"TN\n{cm[0, 0]}",
                f"FP\n{cm[0, 1]}"
            ],
            [
                f"FN\n{cm[1, 0]}",
                f"TP\n{cm[1, 1]}"
            ]
        ]

        fig = px.imshow(
            cm,
            text_auto=False,
            x=[
                f"Predicted: {labels[0]}",
                f"Predicted: {labels[1]}"
            ],
            y=[
                f"Actual: {labels[0]}",
                f"Actual: {labels[1]}"
            ],
            title="Confusion Matrix",
            color_continuous_scale=[
                "#F1F5FF",
                "#DCE6FF",
                "#AFC4F5",
                "#668DE5",
                "#315BEA"
            ]
        )

        fig.update_traces(
            text=annotations,
            texttemplate="%{text}",
            hovertemplate="Count: %{z}<extra></extra>"
        )

    else:

        fig = px.imshow(
            cm,
            text_auto=True,
            x=[
                f"Predicted: {x}"
                for x in labels
            ],
            y=[
                f"Actual: {x}"
                for x in labels
            ],
            title="Confusion Matrix",
            color_continuous_scale=[
                "#F1F5FF",
                "#DCE6FF",
                "#AFC4F5",
                "#668DE5",
                "#315BEA"
            ]
        )

        fig.update_traces(
            texttemplate="%{text}"
        )

    fig.update_layout(
        height=520,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            color="#263247"
        ),
        title_font=dict(
            size=18,
            color="#172033"
        ),
        margin=dict(
            l=80,
            r=40,
            t=70,
            b=80
        ),
        coloraxis_colorbar=dict(
            title="Count"
        )
    )

    return fig