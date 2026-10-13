import streamlit as st


# ==================================================
# METRIC ROW
# ==================================================

def metric_row(metrics):

    columns = st.columns(len(metrics))

    for column, metric in zip(columns, metrics):

        column.metric(
            metric["label"],
            metric["value"]
        )


# ==================================================
# PROGRESS BAR
# ==================================================

def create_progress():

    progress_bar = st.progress(0)

    status = st.empty()

    return progress_bar, status


def update_progress(
    progress_bar,
    status,
    percentage,
    message
):

    progress_bar.progress(
        percentage
    )

    status.markdown(
        f"**{message} — {percentage}%**"
    )


def finish_progress(
    progress_bar,
    status,
    message="Complete!"
):

    progress_bar.progress(100)

    status.success(
        message
    )


# ==================================================
# REQUIRE DATASET
# ==================================================

def require_dataset(df):

    if df is None:

        st.warning(
            "Please upload a dataset first."
        )

        return False

    return True


# ==================================================
# REQUIRE PREPROCESSING
# ==================================================

def require_preprocessing(X):

    if X is None:

        st.warning(
            "Please complete preprocessing first."
        )

        st.info(
            "Go to **Preprocessing**, select your target "
            "if required, and click **Apply Preprocessing**."
        )

        return False

    return True


# ==================================================
# MODEL STATUS
# ==================================================

def model_success(model_name):

    st.success(
        f"**{model_name}** trained successfully."
    )


# ==================================================
# EMPTY STATE
# ==================================================

def empty_state(
    title,
    message
):

    st.subheader(title)

    st.info(message)