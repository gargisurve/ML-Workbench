from .common import *

def render_export():

    st.header(
        "Export Results"
    )

    cleaned_df = st.session_state.get("cleaned_df")

    if cleaned_df is not None:
        st.subheader("Cleaned Dataset")
        st.caption("After duplicate removal and missing-value treatment, before encoding/scaling.")
        st.dataframe(cleaned_df.head(20), use_container_width=True)
        st.download_button(
            "Download Cleaned Dataset (.csv)",
            data=cleaned_df.to_csv(index=False),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Apply preprocessing first to create a cleaned dataset.")

    st.divider()

    model = st.session_state[
        "trained_model"
    ]

    predictions = st.session_state[
        "predictions"
    ]

    model_name = st.session_state[
        "model_name"
    ]

    if model is not None:

        st.subheader(
            "Trained Model"
        )

        st.success(
            f"Model available: **{model_name}**"
        )

        buffer = io.BytesIO()

        joblib.dump(
            model,
            buffer
        )

        buffer.seek(0)

        st.download_button(
            "⬇Download Model (.pkl)",
            data=buffer,
            file_name=(
                f"{model_name.replace(' ', '_')}.pkl"
            ),
            mime="application/octet-stream",
            use_container_width=True
        )

    else:

        st.info(
            "Train a model first to download it."
        )

    st.divider()

    if predictions is not None:

        st.subheader(
            "Predictions"
        )

        prediction_df = pd.DataFrame({
            "Actual":
                st.session_state[
                    "y_test"
                ].values,
            "Predicted":
                predictions
        })

        st.dataframe(
            prediction_df.head(20),
            use_container_width=True
        )

        st.download_button(
            "Download Predictions (.csv)",
            data=prediction_df.to_csv(
                index=False
            ),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

    elif st.session_state[
        "cluster_labels"
    ] is not None:

        st.subheader(
            "Cluster Assignments"
        )

        cluster_df = pd.DataFrame({
            "Cluster":
                st.session_state[
                    "cluster_labels"
                ]
        })

        st.dataframe(
            cluster_df.head(20),
            use_container_width=True
        )

        st.download_button(
            "Download Cluster Results (.csv)",
            data=cluster_df.to_csv(
                index=False
            ),
            file_name="cluster_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    if st.session_state[
        "metrics"
    ] is not None:

        st.divider()

        st.subheader(
            "Model Metrics"
        )

        metrics_df = pd.DataFrame([
            st.session_state["metrics"]
        ])

        st.dataframe(
            metrics_df,
            use_container_width=True
        )

        st.download_button(
            "Download Metrics (.csv)",
            data=metrics_df.to_csv(
                index=False
            ),
            file_name="model_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

    if st.session_state[
        "comparison_results"
    ] is not None:

        st.divider()

        st.subheader(
            "Model Comparison Results"
        )

        comparison_results = (
            st.session_state[
                "comparison_results"
            ]
        )

        st.dataframe(
            comparison_results,
            use_container_width=True
        )

        st.download_button(
            "Download Comparison (.csv)",
            data=comparison_results.to_csv(
                index=False
            ),
            file_name="model_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )
