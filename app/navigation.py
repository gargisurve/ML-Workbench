import streamlit as st


PAGES = [
    "Dataset",
    "Explore Data",
    "Preprocessing",
    "Machine Learning",
    "New Data & Prediction",
    "Evaluation",
    "Model Comparison",
    "Export",
]


def render_navigation():

    st.sidebar.title("ML Workbench")
    st.sidebar.caption("Machine Learning Environment")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Workspace",
        PAGES
    )

    st.sidebar.divider()
    st.sidebar.subheader("Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV dataset",
        type=["csv"]
    )

    if uploaded_file is not None:
        st.sidebar.success(
            f"Dataset loaded\n\n{uploaded_file.name}"
        )
    else:
        st.sidebar.info(
            "Upload a CSV dataset to begin."
        )

    return page, uploaded_file
