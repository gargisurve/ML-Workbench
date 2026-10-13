from .common import *


def create_input_form(feature_df, date_features):

    row = {}

    with st.form("single_prediction_form"):

        st.subheader("Enter New Data")

        st.caption(
            "Enter values for the features used by the trained model."
        )

        columns = st.columns(2)

        for i, column in enumerate(feature_df.columns):

            series = feature_df[column]

            with columns[i % 2]:

                # ------------------------------------------
                # DATE
                # ------------------------------------------

                if column in date_features:

                    default_date = pd.Timestamp(
                        "2015-01-01"
                    ).date()

                    parsed = pd.to_datetime(
                        series,
                        errors="coerce"
                    ).dropna()

                    if len(parsed) > 0:
                        default_date = parsed.iloc[0].date()

                    row[column] = st.date_input(
                        column,
                        value=default_date,
                        key=f"new_pred_{column}"
                    )

                # ------------------------------------------
                # NUMERIC
                # ------------------------------------------

                elif pd.api.types.is_numeric_dtype(series):

                    clean = pd.to_numeric(
                        series,
                        errors="coerce"
                    ).dropna()

                    default = (
                        float(clean.median())
                        if len(clean) > 0
                        else 0.0
                    )

                    if pd.api.types.is_integer_dtype(series):

                        row[column] = st.number_input(
                            column,
                            value=int(round(default)),
                            step=1,
                            key=f"new_pred_{column}"
                        )

                    else:

                        row[column] = st.number_input(
                            column,
                            value=default,
                            format="%.4f",
                            key=f"new_pred_{column}"
                        )

                # ------------------------------------------
                # CATEGORICAL
                # ------------------------------------------

                else:

                    values = (
                        series
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if 0 < len(values) <= 100:

                        row[column] = st.selectbox(
                            column,
                            values,
                            key=f"new_pred_{column}"
                        )

                    else:

                        row[column] = st.text_input(
                            column,
                            value=(
                                values[0]
                                if values
                                else ""
                            ),
                            key=f"new_pred_{column}"
                        )

        submitted = st.form_submit_button(
            "Predict",
            type="primary",
            use_container_width=True
        )

    return submitted, row


def expand_prediction_dates(
    data,
    date_features
):

    data = data.copy()

    for column in date_features:

        if column not in data.columns:
            continue

        parsed = pd.to_datetime(
            data[column],
            errors="coerce"
        )

        data[f"{column}__year"] = parsed.dt.year
        data[f"{column}__month"] = parsed.dt.month
        data[f"{column}__day"] = parsed.dt.day
        data[f"{column}__dayofweek"] = parsed.dt.dayofweek

        data.drop(
            columns=[column],
            inplace=True
        )

    return data


def render_prediction():

    st.header("New Data & Prediction")

    model = st.session_state.get(
        "trained_model"
    )

    processed_X = st.session_state.get(
        "processed_X"
    )

    cleaned_df = st.session_state.get(
        "cleaned_df"
    )

    selected_features = st.session_state.get(
        "selected_features"
    )

    scaler = st.session_state.get(
        "scaler"
    )

    date_features = st.session_state.get(
        "date_features",
        []
    )

    model_type = st.session_state.get(
        "model_type"
    )

    # ==================================================
    # CHECK MODEL
    # ==================================================

    if model is None:

        st.info(
            "Train a Classification or Regression model first."
        )

        return

    if model_type not in [
        "Classification",
        "Regression"
    ]:

        st.info(
            "New-data prediction is available "
            "for Classification and Regression models."
        )

        return

    if (
        processed_X is None
        or cleaned_df is None
        or not selected_features
    ):

        st.warning(
            "Please complete preprocessing and "
            "feature selection first."
        )

        return

    # ==================================================
    # GET THE ACTUAL MODEL FEATURE NAMES
    # ==================================================

    if hasattr(model, "feature_names_in_"):

        model_features = list(
            model.feature_names_in_
        )

    else:

        model_features = list(
            processed_X.columns
        )

    st.success(
        f"Ready to predict using "
        f"**{st.session_state.get('model_name', 'Trained Model')}**"
    )

    st.caption(
        f"Model trained using "
        f"**{len(selected_features)} selected feature(s)**."
    )

    # ==================================================
    # INPUT FORM
    # ==================================================

    feature_df = cleaned_df[
        selected_features
    ].copy()

    submitted, row = create_input_form(
        feature_df,
        date_features
    )

    # ==================================================
    # PREDICTION
    # ==================================================

    if submitted:

        try:

            raw_row = pd.DataFrame(
                [row]
            )

            # ------------------------------------------
            # DATE TRANSFORMATION
            # ------------------------------------------

            raw_row = expand_prediction_dates(
                raw_row,
                date_features
            )

            # ------------------------------------------
            # CATEGORICAL ENCODING
            # ------------------------------------------

            raw_row = pd.get_dummies(
                raw_row,
                drop_first=False
            )

            # ------------------------------------------
            # FORCE EXACT MODEL FEATURES
            # ------------------------------------------

            raw_row = raw_row.reindex(
                columns=model_features,
                fill_value=0
            )

            # ------------------------------------------
            # SCALING
            # ------------------------------------------

            if scaler is not None:

                # Use the scaler's actual feature schema
                # whenever available.

                if hasattr(
                    scaler,
                    "feature_names_in_"
                ):

                    scaler_features = list(
                        scaler.feature_names_in_
                    )

                    raw_row = raw_row.reindex(
                        columns=scaler_features,
                        fill_value=0
                    )

                raw_row = pd.DataFrame(
                    scaler.transform(
                        raw_row
                    ),
                    columns=raw_row.columns,
                    index=raw_row.index
                )

            # ------------------------------------------
            # FINAL SAFETY CHECK
            # ------------------------------------------

            if hasattr(
                model,
                "feature_names_in_"
            ):

                expected = list(
                    model.feature_names_in_
                )

                actual = list(
                    raw_row.columns
                )

                if actual != expected:

                    raw_row = raw_row.reindex(
                        columns=expected,
                        fill_value=0
                    )

            # ------------------------------------------
            # PREDICT
            # ------------------------------------------

            prediction = model.predict(
                raw_row
            )[0]

            st.divider()

            st.subheader(
                "Prediction Result"
            )

            if model_type == "Classification":

                st.metric(
                    "Predicted Class",
                    str(prediction)
                )

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            raw_row
                        )[0]
                    )

                    confidence = float(
                        np.max(
                            probabilities
                        )
                    )

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2%}"
                    )

            else:

                st.metric(
                    "Predicted Value",
                    f"{float(prediction):,.4f}"
                )

            result = row.copy()

            result["Prediction"] = prediction

            st.session_state[
                "latest_single_prediction"
            ] = result

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

    # ==================================================
    # DOWNLOAD
    # ==================================================

    saved_prediction = st.session_state.get(
        "latest_single_prediction"
    )

    if saved_prediction is not None:

        st.divider()

        result_df = pd.DataFrame(
            [saved_prediction]
        )

        st.download_button(
            "⬇️ Download The Prediction",
            data=result_df.to_csv(
                index=False
            ),
            file_name="prediction.csv",
            mime="text/csv",
            use_container_width=True
        )