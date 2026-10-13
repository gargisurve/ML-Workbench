from .common import *


def render_evaluation():

    st.header("Model Evaluation")

    model = st.session_state["trained_model"]

    if model is None:
        st.info("Train a model first from **Machine Learning**.")
        return

    model_type = st.session_state["model_type"]
    model_name = st.session_state["model_name"]
    metrics = st.session_state["metrics"]

    st.success(f"Evaluating: **{model_name}**")

    # --------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------
    if model_type == "Classification":

        y_test = st.session_state["y_test"]
        predictions = st.session_state["predictions"]

        metric_row([
            {"label": "Accuracy", "value": f"{metrics['Accuracy']:.2%}"},
            {"label": "Precision", "value": f"{metrics['Precision']:.2%}"},
            {"label": "Recall", "value": f"{metrics['Recall']:.2%}"},
            {"label": "F1 Score", "value": f"{metrics['F1 Score']:.2%}"}
        ])

        st.divider()
        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).transpose().round(4),
            use_container_width=True
        )

        st.subheader("Confusion Matrix")
        st.plotly_chart(
            show_confusion_matrix(y_test, predictions),
            use_container_width=True
        )

        show_overfitting_alert("Classification")

        st.divider()
        st.subheader("ROC Curve")

        X_test = st.session_state.get("X_test")

        if X_test is not None:
            roc_fig = create_roc_curve(model, X_test, y_test)

            if roc_fig is not None:
                st.plotly_chart(roc_fig, use_container_width=True)
            else:
                st.info(
                    "ROC Curve is available for binary classification models "
                    "that provide probabilities or decision scores."
                )
        else:
            st.info("Test data is not available for ROC analysis.")

    # --------------------------------------------------
    # REGRESSION
    # --------------------------------------------------
    elif model_type == "Regression":

        y_test = st.session_state["y_test"]
        predictions = st.session_state["predictions"]

        metric_row([
            {"label": "R² Score", "value": f"{metrics['R² Score']:.4f}"},
            {"label": "MAE", "value": f"{metrics['MAE']:.4f}"},
            {"label": "MSE", "value": f"{metrics['MSE']:.4f}"},
            {"label": "RMSE", "value": f"{metrics['RMSE']:.4f}"}
        ])

        comparison_df = pd.DataFrame({
            "Observation": np.arange(1, len(predictions) + 1),
            "Actual": np.asarray(y_test),
            "Predicted": np.asarray(predictions)
        })

        st.subheader("Actual vs Predicted")

        # Two clearly different series instead of a single same-colour cloud.
        long_df = comparison_df.melt(
            id_vars="Observation",
            value_vars=["Actual", "Predicted"],
            var_name="Series",
            value_name="Value"
        )

        comparison_fig = px.scatter(
            long_df,
            x="Observation",
            y="Value",
            color="Series",
            title="Actual vs Predicted",
            color_discrete_map={
                "Actual": "#4EA8DE",
                "Predicted": "#FF7B72"
            },
            opacity=0.75
        )

        comparison_fig.update_traces(marker=dict(size=7))
        comparison_fig.update_layout(
            height=560,
            legend_title_text="",
            xaxis_title="Observation",
            yaxis_title="Value"
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True
        )

        st.subheader("Residual Analysis")

        residual_df = pd.DataFrame({
            "Predicted": predictions,
            "Residual": np.asarray(y_test) - np.asarray(predictions)
        })

        residual_fig = px.scatter(
            residual_df,
            x="Predicted",
            y="Residual",
            title="Residuals vs Predicted",
            color_discrete_sequence=["#A78BFA"]
        )

        residual_fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#FF7B72"
        )

        st.plotly_chart(
            residual_fig,
            use_container_width=True
        )

        st.subheader("Residual Distribution")
        st.plotly_chart(
            residual_distribution(y_test, predictions),
            use_container_width=True
        )

        st.subheader("Prediction Error Distribution")
        st.plotly_chart(
            error_distribution(y_test, predictions),
            use_container_width=True
        )

        show_overfitting_alert("Regression")

    # --------------------------------------------------
    # CLUSTERING
    # --------------------------------------------------
    else:

        labels = st.session_state["cluster_labels"]

        metric_row([
            {
                "label": "Clusters",
                "value": metrics["Number of Clusters"]
            },
            {
                "label": "Silhouette Score",
                "value": (
                    f"{metrics['Silhouette Score']:.4f}"
                    if metrics["Silhouette Score"] is not None
                    else "N/A"
                )
            }
        ])

        display_X = st.session_state.get("display_X")

        if (
            display_X is not None
            and labels is not None
            and len(display_X) == len(labels)
        ):
            profile_fig = cluster_profile(display_X, labels)

            if profile_fig is not None:
                st.subheader("Cluster Feature Profile")
                st.plotly_chart(
                    profile_fig,
                    use_container_width=True
                )
