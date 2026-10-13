from .common import *

def render_dataset():

    section = "Dataset"

    st.header(section)

    df = st.session_state["df"]

    if not require_dataset(df):

        st.markdown(
            """
            ### What can you do with ML Workbench?

            Perform a complete machine-learning workflow
            through an interactive interface.

            **Workflow**

            📂 Upload Dataset  
            ↓  
            📊 Explore Data  
            ↓  
            ⚙️ Preprocess  
            ↓  
            🤖 Train ML Model  
            ↓  
            📈 Evaluate  
            ↓  
            🏆 Compare Models  
            ↓  
            💾 Export Results
            """
        )

    else:

        st.success(
            f"Dataset loaded successfully: "
            f"**{st.session_state.get('file_name', 'uploaded dataset')}**"
        )

        info = get_dataset_info(df)

        metric_row([
            {
                "label": "Rows",
                "value": f"{info['rows']:,}"
            },
            {
                "label": "Columns",
                "value": info["columns"]
            },
            {
                "label": "Missing Values",
                "value": f"{info['missing_values']:,}"
            },
            {
                "label": "Duplicates",
                "value": f"{info['duplicates']:,}"
            }
        ])

        st.divider()

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.subheader("Column Information")

        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type":
                df.dtypes.astype(str),
            "Missing Values":
                df.isnull().sum().values,
            "Unique Values":
                df.nunique().values
        })

        st.dataframe(
            column_info,
            use_container_width=True
        )

        st.subheader(
            "Statistical Summary"
        )

        st.dataframe(
            df.describe(
                include="all"
            ).T,
            use_container_width=True
        )


