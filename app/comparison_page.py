from .common import *

def render_model_comparison():

    st.header(
        "Model Comparison"
    )

    X = st.session_state[
        "processed_X"
    ]

    if not require_preprocessing(X):

        st.stop()

    y = st.session_state[
        "processed_y"
    ]

    test_size = st.session_state.get(
        "test_size",
        0.2
    )

    comparison_task = st.selectbox(
        "Select task for comparison",
        [
            "Classification",
            "Regression"
        ]
    )

    st.divider()

    # ==================================================
    # CLASSIFICATION COMPARISON
    # ==================================================

    if comparison_task == "Classification":

        if y is None:

            st.error(
                "Please select a target variable."
            )

        else:

            models = get_classification_models()

            selected_models = st.multiselect(
                "Select models to compare",
                list(models.keys()),
                default=list(models.keys()),
                key="comparison_classification"
            )

            if st.button(
                "Compare Classification Models",
                type="primary",
                use_container_width=True
            ):

                if not selected_models:

                    st.warning(
                        "Select at least one model."
                    )

                else:

                    results = []

                    progress, status = (
                        create_progress()
                    )

                    valid_rows = y.notna()

                    X_clean = X.loc[
                        valid_rows
                    ]

                    y_clean = y.loc[
                        valid_rows
                    ]

                    (
                        X_train,
                        X_test,
                        y_train,
                        y_test
                    ) = split_data(
                        X_clean,
                        y_clean,
                        test_size
                    )

                    total = len(
                        selected_models
                    )

                    for i, model_name in enumerate(
                        selected_models
                    ):

                        percentage = int(
                            (i / total) * 100
                        )

                        update_progress(
                            progress,
                            status,
                            percentage,
                            f"Training {model_name}"
                        )

                        try:

                            start = time.time()

                            model, predictions = (
                                train_classification_model(
                                    model_name,
                                    X_train,
                                    X_test,
                                    y_train,
                                    y_test
                                )
                            )

                            elapsed = (
                                time.time() - start
                            )

                            metrics = (
                                classification_metrics(
                                    y_test,
                                    predictions
                                )
                            )

                            results.append({
                                "Model":
                                    model_name,
                                "Accuracy":
                                    metrics[
                                        "Accuracy"
                                    ],
                                "Precision":
                                    metrics[
                                        "Precision"
                                    ],
                                "Recall":
                                    metrics[
                                        "Recall"
                                    ],
                                "F1 Score":
                                    metrics[
                                        "F1 Score"
                                    ],
                                "Training Time (s)":
                                    elapsed
                            })

                        except Exception:

                            results.append({
                                "Model":
                                    model_name,
                                "Accuracy":
                                    np.nan,
                                "Precision":
                                    np.nan,
                                "Recall":
                                    np.nan,
                                "F1 Score":
                                    np.nan,
                                "Training Time (s)":
                                    np.nan
                            })

                    finish_progress(
                        progress,
                        status,
                        "Model comparison completed!"
                    )

                    results_df = pd.DataFrame(
                        results
                    )

                    st.session_state[
                        "comparison_results"
                    ] = results_df

                    st.subheader(
                        "Comparison Results"
                    )

                    display_df = (
                        results_df.copy()
                    )

                    for column in [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 Score"
                    ]:

                        display_df[column] = (
                            display_df[column]
                            .map(
                                lambda x:
                                f"{x:.2%}"
                                if pd.notna(x)
                                else "Failed"
                            )
                        )

                    display_df[
                        "Training Time (s)"
                    ] = (
                        display_df[
                            "Training Time (s)"
                        ].map(
                            lambda x:
                            f"{x:.2f}"
                            if pd.notna(x)
                            else "Failed"
                        )
                    )

                    st.dataframe(
                        display_df,
                        use_container_width=True
                    )

                    valid_results = (
                        results_df
                        .dropna(
                            subset=["F1 Score"]
                        )
                    )

                    if not valid_results.empty:

                        best = valid_results.loc[
                            valid_results[
                                "F1 Score"
                            ].idxmax()
                        ]

                        st.success(
                            f"Best Model: "
                            f"**{best['Model']}** "
                            f"with F1 Score "
                            f"**{best['F1 Score']:.2%}**"
                        )

    # ==================================================
    # REGRESSION COMPARISON
    # ==================================================

    else:

        if y is None:

            st.error(
                "Please select a target variable."
            )

        else:

            models = get_regression_models()

            selected_models = st.multiselect(
                "Select models to compare",
                list(models.keys()),
                default=list(models.keys()),
                key="comparison_regression"
            )

            if st.button(
                "Compare Regression Models",
                type="primary",
                use_container_width=True
            ):

                if not selected_models:

                    st.warning(
                        "Select at least one model."
                    )

                else:

                    results = []

                    progress, status = (
                        create_progress()
                    )

                    valid_rows = y.notna()

                    X_clean = X.loc[
                        valid_rows
                    ]

                    y_clean = pd.to_numeric(
                        y.loc[valid_rows],
                        errors="coerce"
                    )

                    valid = y_clean.notna()

                    X_clean = X_clean.loc[
                        valid
                    ]

                    y_clean = y_clean.loc[
                        valid
                    ]

                    (
                        X_train,
                        X_test,
                        y_train,
                        y_test
                    ) = split_data(
                        X_clean,
                        y_clean,
                        test_size
                    )

                    total = len(
                        selected_models
                    )

                    for i, model_name in enumerate(
                        selected_models
                    ):

                        percentage = int(
                            (i / total) * 100
                        )

                        update_progress(
                            progress,
                            status,
                            percentage,
                            f"Training {model_name}"
                        )

                        try:

                            start = time.time()

                            model, predictions = (
                                train_regression_model(
                                    model_name,
                                    X_train,
                                    X_test,
                                    y_train,
                                    y_test
                                )
                            )

                            elapsed = (
                                time.time() - start
                            )

                            metrics = (
                                regression_metrics(
                                    y_test,
                                    predictions
                                )
                            )

                            results.append({
                                "Model":
                                    model_name,
                                "R² Score":
                                    metrics[
                                        "R² Score"
                                    ],
                                "MAE":
                                    metrics[
                                        "MAE"
                                    ],
                                "RMSE":
                                    metrics[
                                        "RMSE"
                                    ],
                                "Training Time (s)":
                                    elapsed
                            })

                        except Exception:

                            results.append({
                                "Model":
                                    model_name,
                                "R² Score":
                                    np.nan,
                                "MAE":
                                    np.nan,
                                "RMSE":
                                    np.nan,
                                "Training Time (s)":
                                    np.nan
                            })

                    finish_progress(
                        progress,
                        status,
                        "Model comparison completed!"
                    )

                    results_df = pd.DataFrame(
                        results
                    )

                    st.session_state[
                        "comparison_results"
                    ] = results_df

                    st.subheader(
                        "Comparison Results"
                    )

                    display_df = (
                        results_df.copy()
                    )

                    for column in [
                        "R² Score",
                        "MAE",
                        "RMSE"
                    ]:

                        display_df[column] = (
                            display_df[column]
                            .map(
                                lambda x:
                                f"{x:.4f}"
                                if pd.notna(x)
                                else "Failed"
                            )
                        )

                    display_df[
                        "Training Time (s)"
                    ] = (
                        display_df[
                            "Training Time (s)"
                        ].map(
                            lambda x:
                            f"{x:.2f}"
                            if pd.notna(x)
                            else "Failed"
                        )
                    )

                    st.dataframe(
                        display_df,
                        use_container_width=True
                    )

                    valid_results = (
                        results_df
                        .dropna(
                            subset=["R² Score"]
                        )
                    )

                    if not valid_results.empty:

                        best = valid_results.loc[
                            valid_results[
                                "R² Score"
                            ].idxmax()
                        ]

                        st.success(
                            f"Best Model: "
                            f"**{best['Model']}** "
                            f"with R² Score "
                            f"**{best['R² Score']:.4f}**"
                        )


