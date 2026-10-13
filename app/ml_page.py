from .common import *

def render_machine_learning():

    st.header("Machine Learning")

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

    task = st.selectbox(
        "Select Machine Learning Task",
        [
            "Classification",
            "Regression",
            "Clustering"
        ]
    )

    st.divider()

    # ==================================================
    # CLASSIFICATION
    # ==================================================

    if task == "Classification":

        st.subheader(
            "Classification"
        )

        if y is None:

            st.error(
                "Please select a target variable "
                "in Preprocessing."
            )

        else:

            models = get_classification_models()

            model_name = st.selectbox(
                "Select Classification Model",
                list(models.keys()),
                key="classification_model"
            )

            descriptions = {
                "Logistic Regression":
                    "Linear classification model",

                "K-Nearest Neighbors":
                    "Uses nearby observations",

                "Decision Tree":
                    "Tree-based classification",

                "Random Forest":
                    "Ensemble of decision trees",

                "Support Vector Machine":
                    "Finds an optimal separating boundary"
            }

            st.info(
                f"**{model_name}:** "
                f"{descriptions[model_name]}"
            )

            if st.button(
                "Train Classification Model",
                type="primary",
                use_container_width=True
            ):

                try:

                    progress, status = (
                        create_progress()
                    )

                    update_progress(
                        progress,
                        status,
                        10,
                        "Preparing data"
                    )

                    valid_rows = y.notna()

                    X_clean = X.loc[
                        valid_rows
                    ]

                    y_clean = y.loc[
                        valid_rows
                    ]

                    update_progress(
                        progress,
                        status,
                        25,
                        "Splitting dataset"
                    )

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

                    update_progress(
                        progress,
                        status,
                        35,
                        f"Training {model_name}"
                    )

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

                    training_time = (
                        time.time() - start
                    )

                    update_progress(
                        progress,
                        status,
                        85,
                        "Evaluating model"
                    )

                    metrics = classification_metrics(
                        y_test,
                        predictions
                    )

                    train_predictions = model.predict(X_train)
                    train_metrics = classification_metrics(
                        y_train,
                        train_predictions
                    )

                    st.session_state[
                        "trained_model"
                    ] = model

                    st.session_state[
                        "predictions"
                    ] = predictions

                    st.session_state[
                        "metrics"
                    ] = metrics

                    st.session_state[
                        "train_metrics"
                    ] = train_metrics

                    st.session_state[
                        "model_type"
                    ] = "Classification"

                    st.session_state[
                        "model_name"
                    ] = model_name

                    st.session_state[
                        "y_test"
                    ] = y_test

                    st.session_state[
                        "X_test"
                    ] = X_test

                    st.session_state[
                        "training_time"
                    ] = training_time

                    finish_progress(
                        progress,
                        status,
                        "Model training completed!"
                    )

                    st.divider()

                    st.subheader(
                        "Model Performance"
                    )

                    metric_row([
                        {
                            "label": "Accuracy",
                            "value":
                                f"{metrics['Accuracy']:.2%}"
                        },
                        {
                            "label": "Precision",
                            "value":
                                f"{metrics['Precision']:.2%}"
                        },
                        {
                            "label": "Recall",
                            "value":
                                f"{metrics['Recall']:.2%}"
                        },
                        {
                            "label": "F1 Score",
                            "value":
                                f"{metrics['F1 Score']:.2%}"
                        }
                    ])

                    st.subheader(
                        "Confusion Matrix"
                    )

                    st.plotly_chart(
                        show_confusion_matrix(y_test, predictions),
                        use_container_width=True
                    )

                    show_overfitting_alert("Classification")

                except Exception as e:

                    st.error(
                        f"Model training failed: {e}"
                    )

    # ==================================================
    # REGRESSION
    # ==================================================

    elif task == "Regression":

        st.subheader(
            "Regression"
        )

        if y is None:

            st.error(
                "Please select a target variable "
                "in Preprocessing."
            )

        else:

            models = get_regression_models()

            model_name = st.selectbox(
                "Select Regression Model",
                list(models.keys()),
                key="regression_model"
            )

            if st.button(
                "Train Regression Model",
                type="primary",
                use_container_width=True
            ):

                try:

                    progress, status = (
                        create_progress()
                    )

                    update_progress(
                        progress,
                        status,
                        10,
                        "Preparing data"
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

                    update_progress(
                        progress,
                        status,
                        25,
                        "Splitting dataset"
                    )

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

                    update_progress(
                        progress,
                        status,
                        35,
                        f"Training {model_name}"
                    )

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

                    training_time = (
                        time.time() - start
                    )

                    update_progress(
                        progress,
                        status,
                        85,
                        "Evaluating model"
                    )

                    metrics = regression_metrics(
                        y_test,
                        predictions
                    )

                    train_predictions = model.predict(X_train)
                    train_metrics = regression_metrics(
                        y_train,
                        train_predictions
                    )

                    st.session_state[
                        "trained_model"
                    ] = model

                    st.session_state[
                        "predictions"
                    ] = predictions

                    st.session_state[
                        "metrics"
                    ] = metrics

                    st.session_state[
                        "train_metrics"
                    ] = train_metrics

                    st.session_state[
                        "model_type"
                    ] = "Regression"

                    st.session_state[
                        "model_name"
                    ] = model_name

                    st.session_state[
                        "y_test"
                    ] = y_test

                    st.session_state[
                        "X_test"
                    ] = X_test

                    st.session_state[
                        "training_time"
                    ] = training_time

                    finish_progress(
                        progress,
                        status,
                        "Model training completed!"
                    )

                    st.divider()

                    st.subheader(
                        "Model Performance"
                    )

                    metric_row([
                        {
                            "label": "R² Score",
                            "value":
                                f"{metrics['R² Score']:.4f}"
                        },
                        {
                            "label": "MAE",
                            "value":
                                f"{metrics['MAE']:.4f}"
                        },
                        {
                            "label": "MSE",
                            "value":
                                f"{metrics['MSE']:.4f}"
                        },
                        {
                            "label": "RMSE",
                            "value":
                                f"{metrics['RMSE']:.4f}"
                        }
                    ])

                    comparison_df = pd.DataFrame({
                        "Actual":
                            y_test.values,
                        "Predicted":
                            predictions
                    })

                    st.subheader(
                        "Actual vs Predicted"
                    )

                    fig = px.scatter(
                        comparison_df,
                        x="Actual",
                        y="Predicted",
                        title="Actual vs Predicted",
                        opacity=0.6
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    show_overfitting_alert("Regression")

                except Exception as e:

                    st.error(
                        f"Model training failed: {e}"
                    )

    # ==================================================
    # CLUSTERING
    # ==================================================

    else:

        st.subheader(
            "Clustering"
        )

        st.caption(
            "Discover natural groups in your dataset "
            "without using a target variable."
        )

        clustering_model = st.selectbox(
            "Select Clustering Algorithm",
            [
                "K-Means",
                "Hierarchical Clustering",
                "DBSCAN"
            ],
            key="clustering_model"
        )

        display_X = st.session_state.get(
            "display_X"
        )

        # --------------------------------------------------
        # AVAILABLE VISUAL FEATURES
        # --------------------------------------------------

        if display_X is None or display_X.shape[1] < 2:

            st.warning(
                "At least two numerical features "
                "are required for cluster visualization."
            )

        else:

            visual_columns = (
                display_X.columns.tolist()
            )

            # ==================================================
            # K-MEANS
            # ==================================================

            if clustering_model == "K-Means":

                n_clusters = st.slider(
                    "Number of clusters",
                    min_value=2,
                    max_value=10,
                    value=5,
                    key="kmeans_clusters"
                )

                if st.button(
                    "Run K-Means",
                    type="primary",
                    use_container_width=True
                ):

                    try:

                        progress, status = (
                            create_progress()
                        )

                        update_progress(
                            progress,
                            status,
                            10,
                            "Preparing clustering data"
                        )

                        update_progress(
                            progress,
                            status,
                            25,
                            "Initializing K-Means"
                        )

                        start = time.time()

                        model, labels = run_kmeans(
                            X,
                            n_clusters
                        )

                        training_time = (
                            time.time() - start
                        )

                        update_progress(
                            progress,
                            status,
                            80,
                            "Calculating cluster metrics"
                        )

                        metrics = clustering_metrics(
                            X.values,
                            labels
                        )

                        st.session_state[
                            "trained_model"
                        ] = model

                        st.session_state[
                            "cluster_labels"
                        ] = labels

                        st.session_state[
                            "metrics"
                        ] = metrics

                        st.session_state[
                            "model_type"
                        ] = "Clustering"

                        st.session_state[
                            "model_name"
                        ] = "K-Means"

                        st.session_state[
                            "training_time"
                        ] = training_time

                        finish_progress(
                            progress,
                            status,
                            "K-Means completed!"
                        )

                        st.divider()

                        # ==================================================
                        # RESULTS
                        # ==================================================

                        st.subheader(
                            "Clustering Results"
                        )

                        silhouette = (
                            metrics[
                                "Silhouette Score"
                            ]
                        )

                        metric_row([
                            {
                                "label": "Clusters",
                                "value":
                                    metrics[
                                        "Number of Clusters"
                                    ]
                            },
                            {
                                "label": "Silhouette Score",
                                "value":
                                    (
                                        f"{silhouette:.4f}"
                                        if silhouette is not None
                                        else "N/A"
                                    )
                            },
                            {
                                "label": "Training Time",
                                "value":
                                    f"{training_time:.2f}s"
                            }
                        ])

                        st.divider()

                        # ==================================================
                        # VISUALIZATION CONTROLS
                        # ==================================================

                        st.subheader(
                            "Cluster Visualization"
                        )

                        col1, col2 = st.columns(2)

                        x_column = col1.selectbox(
                            "X-axis",
                            visual_columns,
                            key="kmeans_x_axis"
                        )

                        y_options = [
                            col
                            for col in visual_columns
                            if col != x_column
                        ]

                        y_column = col2.selectbox(
                            "Y-axis",
                            y_options,
                            key="kmeans_y_axis"
                        )

                        # ==================================================
                        # PLOT DATA
                        # ==================================================

                        plot_df = prepare_cluster_plot_data(
                            display_X,
                            labels,
                            x_column,
                            y_column
                        )

                        centroids = calculate_visual_centroids(
                            display_X,
                            labels,
                            x_column,
                            y_column
                        )

                        # --------------------------------------------------
                        # MAIN SCATTER
                        # --------------------------------------------------

                        fig = px.scatter(
                            plot_df,
                            x=x_column,
                            y=y_column,
                            color="Cluster Label",
                            title=(
                                f"K-Means Clustering — "
                                f"{x_column} vs {y_column}"
                            ),
                            hover_data=[
                                x_column,
                                y_column,
                                "Cluster Label"
                            ]
                        )

                        # --------------------------------------------------
                        # ADD CENTROIDS
                        # --------------------------------------------------

                        if not centroids.empty:

                            fig.add_scatter(
                                x=centroids[
                                    x_column
                                ],
                                y=centroids[
                                    y_column
                                ],
                                mode="markers",
                                name="Centroid",
                                marker=dict(
                                    size=20,
                                    symbol="circle",
                                    line=dict(
                                        width=2
                                    )
                                ),
                                hovertemplate=(
                                    "<b>%{text}</b><br>"
                                    + f"{x_column}: "
                                    + "%{x:.2f}<br>"
                                    + f"{y_column}: "
                                    + "%{y:.2f}"
                                    + "<extra></extra>"
                                ),
                                text=centroids[
                                    "Cluster Label"
                                ]
                            )

                        fig.update_layout(
                            height=650,
                            legend_title_text="Clusters",
                            title_x=0.5
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        # ==================================================
                        # CLUSTER SUMMARY
                        # ==================================================

                        st.subheader(
                            "Cluster Summary"
                        )

                        summary = get_cluster_summary(
                            display_X,
                            labels,
                            x_column,
                            y_column
                        )

                        st.dataframe(
                            summary.round(2),
                            use_container_width=True
                        )

                        # ==================================================
                        # DISTRIBUTION
                        # ==================================================

                        st.subheader(
                            "Cluster Distribution"
                        )

                        counts = (
                            pd.Series(labels)
                            .value_counts()
                            .sort_index()
                            .reset_index()
                        )

                        counts.columns = [
                            "Cluster",
                            "Observations"
                        ]

                        counts["Cluster"] = (
                            counts["Cluster"]
                            .apply(
                                lambda x:
                                (
                                    "Noise"
                                    if x == -1
                                    else f"Cluster {x + 1}"
                                )
                            )
                        )

                        distribution_fig = px.bar(
                            counts,
                            x="Cluster",
                            y="Observations",
                            title="Observations per Cluster"
                        )

                        st.plotly_chart(
                            distribution_fig,
                            use_container_width=True
                        )

                    except Exception as e:

                        st.error(
                            f"Clustering failed: {e}"
                        )

            # ==================================================
            # HIERARCHICAL
            # ==================================================

            elif clustering_model == "Hierarchical Clustering":

                n_clusters = st.slider(
                    "Number of clusters",
                    min_value=2,
                    max_value=10,
                    value=5,
                    key="hierarchical_clusters"
                )

                st.warning(
                    "Hierarchical clustering may be slower "
                    "for large datasets."
                )

                if st.button(
                    "Run Hierarchical Clustering",
                    type="primary",
                    use_container_width=True
                ):

                    try:

                        progress, status = (
                            create_progress()
                        )

                        update_progress(
                            progress,
                            status,
                            10,
                            "Preparing clustering data"
                        )

                        update_progress(
                            progress,
                            status,
                            25,
                            "Running hierarchical clustering"
                        )

                        start = time.time()

                        model, labels = (
                            run_hierarchical(
                                X,
                                n_clusters
                            )
                        )

                        training_time = (
                            time.time() - start
                        )

                        update_progress(
                            progress,
                            status,
                            80,
                            "Calculating cluster metrics"
                        )

                        metrics = clustering_metrics(
                            X.values,
                            labels
                        )

                        st.session_state[
                            "trained_model"
                        ] = model

                        st.session_state[
                            "cluster_labels"
                        ] = labels

                        st.session_state[
                            "metrics"
                        ] = metrics

                        st.session_state[
                            "model_type"
                        ] = "Clustering"

                        st.session_state[
                            "model_name"
                        ] = "Hierarchical Clustering"

                        st.session_state[
                            "training_time"
                        ] = training_time

                        finish_progress(
                            progress,
                            status,
                            "Hierarchical clustering completed!"
                        )

                        st.divider()

                        metric_row([
                            {
                                "label": "Clusters",
                                "value":
                                    metrics[
                                        "Number of Clusters"
                                    ]
                            },
                            {
                                "label": "Silhouette Score",
                                "value":
                                    (
                                        f"{metrics['Silhouette Score']:.4f}"
                                        if metrics[
                                            "Silhouette Score"
                                        ] is not None
                                        else "N/A"
                                    )
                            },
                            {
                                "label": "Training Time",
                                "value":
                                    f"{training_time:.2f}s"
                            }
                        ])

                        # ==================================================
                        # VISUALIZATION
                        # ==================================================

                        st.subheader(
                            "Cluster Visualization"
                        )

                        col1, col2 = st.columns(2)

                        x_column = col1.selectbox(
                            "X-axis",
                            visual_columns,
                            key="hierarchical_x_axis"
                        )

                        y_options = [
                            col
                            for col in visual_columns
                            if col != x_column
                        ]

                        y_column = col2.selectbox(
                            "Y-axis",
                            y_options,
                            key="hierarchical_y_axis"
                        )

                        plot_df = prepare_cluster_plot_data(
                            display_X,
                            labels,
                            x_column,
                            y_column
                        )

                        fig = px.scatter(
                            plot_df,
                            x=x_column,
                            y=y_column,
                            color="Cluster Label",
                            title=(
                                f"Hierarchical Clustering — "
                                f"{x_column} vs {y_column}"
                            ),
                            hover_data=[
                                x_column,
                                y_column,
                                "Cluster Label"
                            ]
                        )

                        fig.update_layout(
                            height=650,
                            title_x=0.5
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        # ==================================================
                        # SUMMARY
                        # ==================================================

                        st.subheader(
                            "Cluster Summary"
                        )

                        summary = get_cluster_summary(
                            display_X,
                            labels,
                            x_column,
                            y_column
                        )

                        st.dataframe(
                            summary.round(2),
                            use_container_width=True
                        )

                    except Exception as e:

                        st.error(
                            f"Clustering failed: {e}"
                        )

            # ==================================================
            # DBSCAN
            # ==================================================

            else:

                col1, col2 = st.columns(2)

                eps = col1.slider(
                    "Epsilon",
                    min_value=0.1,
                    max_value=3.0,
                    value=0.5,
                    step=0.1,
                    key="dbscan_eps"
                )

                min_samples = col2.slider(
                    "Minimum Samples",
                    min_value=2,
                    max_value=20,
                    value=5,
                    key="dbscan_samples"
                )

                if st.button(
                    "Run DBSCAN",
                    type="primary",
                    use_container_width=True
                ):

                    try:

                        progress, status = (
                            create_progress()
                        )

                        update_progress(
                            progress,
                            status,
                            10,
                            "Preparing clustering data"
                        )

                        update_progress(
                            progress,
                            status,
                            25,
                            "Running DBSCAN"
                        )

                        start = time.time()

                        model, labels = run_dbscan(
                            X,
                            eps,
                            min_samples
                        )

                        training_time = (
                            time.time() - start
                        )

                        update_progress(
                            progress,
                            status,
                            80,
                            "Calculating cluster metrics"
                        )

                        metrics = clustering_metrics(
                            X.values,
                            labels
                        )

                        st.session_state[
                            "trained_model"
                        ] = model

                        st.session_state[
                            "cluster_labels"
                        ] = labels

                        st.session_state[
                            "metrics"
                        ] = metrics

                        st.session_state[
                            "model_type"
                        ] = "Clustering"

                        st.session_state[
                            "model_name"
                        ] = "DBSCAN"

                        st.session_state[
                            "training_time"
                        ] = training_time

                        finish_progress(
                            progress,
                            status,
                            "DBSCAN completed!"
                        )

                        noise = int(
                            np.sum(labels == -1)
                        )

                        st.divider()

                        metric_row([
                            {
                                "label": "Clusters",
                                "value":
                                    metrics[
                                        "Number of Clusters"
                                    ]
                            },
                            {
                                "label": "Noise Points",
                                "value":
                                    f"{noise:,}"
                            },
                            {
                                "label": "Silhouette Score",
                                "value":
                                    (
                                        f"{metrics['Silhouette Score']:.4f}"
                                        if metrics[
                                            "Silhouette Score"
                                        ] is not None
                                        else "N/A"
                                    )
                            }
                        ])

                        # ==================================================
                        # VISUALIZATION
                        # ==================================================

                        st.subheader(
                            "Cluster Visualization"
                        )

                        col1, col2 = st.columns(2)

                        x_column = col1.selectbox(
                            "X-axis",
                            visual_columns,
                            key="dbscan_x_axis"
                        )

                        y_options = [
                            col
                            for col in visual_columns
                            if col != x_column
                        ]

                        y_column = col2.selectbox(
                            "Y-axis",
                            y_options,
                            key="dbscan_y_axis"
                        )

                        plot_df = prepare_cluster_plot_data(
                            display_X,
                            labels,
                            x_column,
                            y_column
                        )

                        fig = px.scatter(
                            plot_df,
                            x=x_column,
                            y=y_column,
                            color="Cluster Label",
                            title=(
                                f"DBSCAN Clustering — "
                                f"{x_column} vs {y_column}"
                            ),
                            hover_data=[
                                x_column,
                                y_column,
                                "Cluster Label"
                            ]
                        )

                        fig.update_layout(
                            height=650,
                            title_x=0.5
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        # ==================================================
                        # SUMMARY
                        # ==================================================

                        st.subheader(
                            "Cluster Summary"
                        )

                        summary = get_cluster_summary(
                            display_X,
                            labels,
                            x_column,
                            y_column
                        )

                        st.dataframe(
                            summary.round(2),
                            use_container_width=True
                        )

                    except Exception as e:

                        st.error(
                            f"Clustering failed: {e}"
                        )


