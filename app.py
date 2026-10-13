import streamlit as st

# Page configuration is kept here so no styling from the old
# setup_page() function can override the application theme.
st.set_page_config(
    page_title="ML Workbench",
    page_icon="ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

from modules.data_processing import load_data

from app.common import apply_workbench_style
from app.navigation import render_navigation
from app.state import initialize_state, reset_ml_state

from app.dataset_page import render_dataset
from app.explore_page import render_explore
from app.preprocessing_page import render_preprocessing
from app.ml_page import render_machine_learning
from app.prediction_page import render_prediction
from app.evaluation_page import render_evaluation
from app.comparison_page import render_model_comparison
from app.export_page import render_export


apply_workbench_style()
initialize_state()

st.title("ML Workbench")
st.caption(
    "An interactive environment for data preparation, "
    "machine learning and model evaluation."
)

page, uploaded_file = render_navigation()

if uploaded_file is not None:
    try:
        file_id = (uploaded_file.name, uploaded_file.size)

        if st.session_state.get("file_id") != file_id:
            st.session_state["file_id"] = file_id
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["df"] = load_data(uploaded_file)

            reset_ml_state()

            st.session_state["file_id"] = file_id
            st.session_state["file_name"] = uploaded_file.name

    except Exception as e:
        st.error(f"Unable to read dataset: {e}")


if page == "Dataset":
    render_dataset()
elif page == "Explore Data":
    render_explore()
elif page == "Preprocessing":
    render_preprocessing()
elif page == "Machine Learning":
    render_machine_learning()
elif page == "New Data & Prediction":
    render_prediction()
elif page == "Evaluation":
    render_evaluation()
elif page == "Model Comparison":
    render_model_comparison()
elif page == "Export":
    render_export()
