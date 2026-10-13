from .common import *


def is_date_like(column, series):
    """
    Detect columns that are likely to contain dates/timestamps.
    """

    name = str(column).lower()

    date_keywords = [
        "date",
        "datetime",
        "timestamp",
        "time"
    ]

    if any(word in name for word in date_keywords):
        return True

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if series.dtype == "object":

        sample = series.dropna().astype(str).head(100)

        if len(sample) > 0:

            parsed = pd.to_datetime(
                sample,
                errors="coerce"
            )

            return parsed.notna().mean() >= 0.8

    return False


def is_id_like(column, series):

    name = str(column).lower()

    id_keywords = [
        "id",
        "index",
        "row_number",
        "record_number"
    ]

    if any(
        name == keyword or name.endswith("_" + keyword)
        for keyword in id_keywords
    ):
        return True

    # Very high-cardinality columns are often identifiers.
    if len(series) > 0:

        unique_ratio = (
            series.nunique(dropna=True)
            / len(series)
        )

        if unique_ratio > 0.98:
            return True

    return False


def expand_selected_dates(X):

    """
    Convert selected date columns into useful numerical
    date components instead of treating every date as
    a separate categorical value.
    """

    X = X.copy()
    date_columns = []

    for column in list(X.columns):

        if is_date_like(column, X[column]):

            parsed = pd.to_datetime(
                X[column],
                errors="coerce"
            )

            if parsed.notna().sum() == 0:
                continue

            X[f"{column}__year"] = parsed.dt.year
            X[f"{column}__month"] = parsed.dt.month
            X[f"{column}__day"] = parsed.dt.day
            X[f"{column}__dayofweek"] = parsed.dt.dayofweek

            X = X.drop(
                columns=[column]
            )

            date_columns.append(column)

    return X, date_columns


def render_preprocessing():

    st.header("Data Preprocessing")

    df = st.session_state["df"]

    if not require_dataset(df):
        return

    df = df.copy()

    st.caption(
        "Prepare your dataset and select exactly which "
        "features should be used by the machine-learning model."
    )

    st.divider()

    # ==================================================
    # DUPLICATES
    # ==================================================

    st.subheader("🧹 Duplicate Rows")

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count > 0:

        st.warning(
            f"This dataset contains "
            f"**{duplicate_count:,}** duplicate rows."
        )

        remove_duplicate_rows = st.checkbox(
            "Remove duplicate rows",
            value=True,
            key="remove_duplicates"
        )

    else:

        st.success(
            "No duplicate rows found."
        )

        remove_duplicate_rows = False

    st.divider()

    # ==================================================
    # TARGET
    # ==================================================

    st.subheader("Target Variable")

    target_column = st.selectbox(
        "Select target column for supervised learning",
        ["None"] + df.columns.tolist(),
        key="target_selector"
    )

    target = (
        None
        if target_column == "None"
        else target_column
    )

    st.session_state["target_column"] = target

    st.divider()

    # ==================================================
    # FEATURE SELECTION
    # ==================================================

    st.subheader("Feature Selection")

    if target is not None:

        available_features = [
            column
            for column in df.columns
            if column != target
        ]

    else:

        available_features = list(
            df.columns
        )

    date_columns = [
        column
        for column in available_features
        if is_date_like(
            column,
            df[column]
        )
    ]

    id_columns = [
        column
        for column in available_features
        if is_id_like(
            column,
            df[column]
        )
    ]

    default_features = [
        column
        for column in available_features
        if column not in date_columns
        and column not in id_columns
    ]

    # If automatic detection removes everything,
    # give the user all available features.
    if not default_features:
        default_features = available_features.copy()

    selected_features = st.multiselect(
        "Select features to use for training",
        options=available_features,
        default=default_features,
        key="feature_selector"
    )

    if id_columns:

        st.caption(
            "ID/index-like columns are excluded by default "
            "because they usually do not provide useful predictive information."
        )

    if not selected_features:

        st.warning(
            "Please select at least one feature."
        )

    else:

        st.success(
            f"**{len(selected_features)} feature(s)** selected for training."
        )

    st.session_state[
        "selected_features"
    ] = selected_features

    st.divider()

    # ==================================================
    # MISSING VALUES
    # ==================================================

    st.subheader(
        "Missing Value Treatment"
    )

    numeric_strategy = st.selectbox(
        "Numerical columns",
        [
            "median",
            "mean",
            "zero"
        ],
        key="numeric_missing"
    )

    categorical_strategy = st.selectbox(
        "Categorical columns",
        [
            "mode",
            "unknown"
        ],
        key="categorical_missing"
    )

    st.divider()

    # ==================================================
    # ENCODING
    # ==================================================

    st.subheader(
        "Categorical Encoding"
    )

    encoding_enabled = st.checkbox(
        "Apply One-Hot Encoding",
        value=True,
        key="encoding_enabled"
    )

    st.caption(
        "Categorical features are converted into numerical features."
    )

    st.divider()

    # ==================================================
    # SCALING
    # ==================================================

    st.subheader(
        "Feature Scaling"
    )

    scaling_enabled = st.checkbox(
        "Apply Feature Scaling",
        value=True,
        key="scaling_enabled"
    )

    scaling_method = st.selectbox(
        "Scaling method",
        [
            "StandardScaler",
            "MinMaxScaler"
        ],
        disabled=not scaling_enabled,
        key="scaling_method"
    )

    st.divider()

    # ==================================================
    # TRAIN TEST SPLIT
    # ==================================================

    st.subheader(
        "Train-Test Split"
    )

    test_size = st.slider(
        "Test set size",
        0.1,
        0.4,
        0.2,
        0.05,
        key="test_size_slider"
    )

    st.caption(
        f"Training: **{int((1-test_size)*100)}%**   |   "
        f"Testing: **{int(test_size*100)}%**"
    )

    st.divider()

    # ==================================================
    # APPLY
    # ==================================================

    if st.button(
        "Apply Preprocessing",
        type="primary",
        use_container_width=True
    ):

        if not selected_features:

            st.error(
                "Please select at least one feature."
            )

            st.stop()

        if target is not None and target in selected_features:

            st.error(
                "The target variable cannot also be used as a feature."
            )

            st.stop()

        progress, status = create_progress()

        update_progress(
            progress,
            status,
            10,
            "Preparing dataset"
        )

        processed_df = df.copy()

        # --------------------------------------------------
        # DUPLICATES
        # --------------------------------------------------

        if remove_duplicate_rows:

            processed_df = (
                processed_df
                .drop_duplicates()
                .reset_index(drop=True)
            )

        update_progress(
            progress,
            status,
            30,
            "Handling missing values"
        )

        # Missing values are handled on the selected
        # dataset before feature transformation.
        columns_to_process = selected_features.copy()

        if target is not None:
            columns_to_process.append(target)

        columns_to_process = list(
            dict.fromkeys(columns_to_process)
        )

        temp_df = processed_df[
            columns_to_process
        ].copy()

        temp_df = handle_missing_values(
            temp_df,
            numeric_strategy,
            categorical_strategy
        )

        # Preserve target separately.
        if target is not None:

            y = temp_df[target]

            X = temp_df[
                selected_features
            ].copy()

        else:

            y = None

            X = temp_df[
                selected_features
            ].copy()

        # --------------------------------------------------
        # CLEANED DATASET
        # --------------------------------------------------

        cleaned_for_export = processed_df.copy()

        # Replace only selected columns with their
        # missing-value-treated versions.
        for column in columns_to_process:

            if column in temp_df.columns:
                cleaned_for_export[column] = temp_df[column]

        st.session_state[
            "cleaned_df"
        ] = cleaned_for_export.copy()

        # --------------------------------------------------
        # DATE FEATURES
        # --------------------------------------------------

        update_progress(
            progress,
            status,
            45,
            "Processing selected date features"
        )

        X, date_features = expand_selected_dates(
            X
        )

        st.session_state[
            "date_features"
        ] = date_features

        # --------------------------------------------------
        # DISPLAY DATA FOR CLUSTERING
        # --------------------------------------------------

        display_X = X.select_dtypes(
            include=np.number
        ).copy()

        st.session_state[
            "display_X"
        ] = display_X

        # --------------------------------------------------
        # ENCODING
        # --------------------------------------------------

        update_progress(
            progress,
            status,
            60,
            "Encoding categorical features"
        )

        if encoding_enabled:

            X = encode_features(
                X
            )

        # Save the exact feature structure used by the model.
        # New prediction data MUST reproduce this structure.
        st.session_state["training_feature_columns"] = X.columns.tolist()
        st.session_state["encoding_enabled_for_model"] = encoding_enabled

        # --------------------------------------------------
        # SCALING
        # --------------------------------------------------

        update_progress(
            progress,
            status,
            75,
            "Scaling numerical features"
        )

        scaler = None

        if scaling_enabled:

            X, scaler = scale_features(
                X,
                scaling_method
            )

        st.session_state["scaler"] = scaler

        # --------------------------------------------------
        # SAVE
        # --------------------------------------------------

        update_progress(
            progress,
            status,
            90,
            "Saving processed dataset"
        )

        st.session_state[
            "processed_X"
        ] = X

        st.session_state[
            "processed_y"
        ] = y

        st.session_state[
            "scaler"
        ] = scaler

        st.session_state[
            "test_size"
        ] = test_size

        st.session_state[
            "preprocessing_done"
        ] = True

        # New preprocessing means old models
        # are no longer valid.
        st.session_state[
            "trained_model"
        ] = None

        st.session_state[
            "predictions"
        ] = None

        st.session_state[
            "metrics"
        ] = None

        st.session_state[
            "train_metrics"
        ] = None

        st.session_state[
            "model_type"
        ] = None

        st.session_state[
            "model_name"
        ] = None

        st.session_state[
            "y_test"
        ] = None

        st.session_state[
            "X_test"
        ] = None

        st.session_state[
            "cluster_labels"
        ] = None

        st.session_state[
            "latest_single_prediction"
        ] = None

        finish_progress(
            progress,
            status,
            "Preprocessing completed successfully!"
        )

    # ==================================================
    # RESULTS
    # ==================================================

    if st.session_state.get(
        "preprocessing_done",
        False
    ):

        st.divider()

        st.subheader(
            "Processed Dataset"
        )

        X_processed = st.session_state[
            "processed_X"
        ]

        metric_row([
            {
                "label": "Selected Features",
                "value": len(
                    st.session_state.get(
                        "selected_features",
                        []
                    )
                )
            },
            {
                "label": "Processed Features",
                "value": X_processed.shape[1]
            },
            {
                "label": "Processed Rows",
                "value": f"{X_processed.shape[0]:,}"
            }
        ])

        st.dataframe(
            X_processed.head(10),
            use_container_width=True
        )