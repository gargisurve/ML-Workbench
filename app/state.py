import streamlit as st


DEFAULT_STATE = {
    "df": None,
    "cleaned_df": None,

    # ML data
    "processed_X": None,
    "processed_y": None,

    # Selected features
    "selected_features": None,
    "date_features": [],

    # Model
    "trained_model": None,
    "predictions": None,
    "metrics": None,
    "train_metrics": None,

    "model_type": None,
    "y_test": None,
    "X_test": None,
    "model_name": None,

    # Clustering
    "cluster_labels": None,
    "display_X": None,

    # Comparison
    "comparison_results": None,

    # Preprocessing
    "target_column": None,
    "scaler": None,
    "test_size": 0.2,
    "preprocessing_done": False,

    # Upload
    "file_id": None,
    "file_name": None,

    # Single prediction
    "latest_single_prediction": None,
}


def initialize_state():

    for key, value in DEFAULT_STATE.items():

        if key not in st.session_state:
            st.session_state[key] = value


def reset_ml_state():

    keys_to_reset = [
        "cleaned_df",
        "processed_X",
        "processed_y",
        "selected_features",
        "date_features",

        "trained_model",
        "predictions",
        "metrics",
        "train_metrics",

        "model_type",
        "y_test",
        "X_test",
        "model_name",

        "cluster_labels",
        "display_X",

        "comparison_results",

        "target_column",
        "scaler",

        "latest_single_prediction",
    ]

    for key in keys_to_reset:
        st.session_state[key] = None

    st.session_state["date_features"] = []

    st.session_state["test_size"] = 0.2
    st.session_state["preprocessing_done"] = False