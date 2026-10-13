from .common import *

def render_explore():

    st.header("Explore Data")

    df = st.session_state["df"]

    if require_dataset(df):

        numeric_columns = (
            get_numeric_columns(df)
        )

        categorical_columns = (
            get_categorical_columns(df)
        )

        st.subheader(
            "Dataset Health"
        )

        metric_row([
            {
                "label": "Missing Values",
                "value":
                    f"{df.isnull().sum().sum():,}"
            },
            {
                "label": "Duplicate Rows",
                "value":
                    f"{df.duplicated().sum():,}"
            },
            {
                "label": "Total Columns",
                "value":
                    len(df.columns)
            }
        ])

        st.divider()

        visualization = st.selectbox(
            "Choose visualization",
            [
                "Distribution", "Boxplot", "Density", "Scatter",
                "Correlation", "Categorical", "Composition",
                "Missing Values", "Unique Values"
            ],
            key="explore_visualization"
        )

        # Streamlit tabs execute all tab contents on every rerun.
        # Render only the selected visualization to keep the app responsive.
        if visualization == "Distribution":
            if numeric_columns:
                column = st.selectbox("Select numerical column", numeric_columns, key="hist_column")
                st.plotly_chart(histogram(df, column), use_container_width=True)
            else:
                st.info("No numerical columns available.")

        elif visualization == "Boxplot":
            if numeric_columns:
                column = st.selectbox("Select numerical column", numeric_columns, key="box_column")
                st.plotly_chart(boxplot(df, column), use_container_width=True)
            else:
                st.info("No numerical columns available.")

        elif visualization == "Density":
            if numeric_columns:
                column = st.selectbox("Select numerical column", numeric_columns, key="density_column")
                st.plotly_chart(density_plot(df, column), use_container_width=True)
            else:
                st.info("No numerical columns available.")

        elif visualization == "Scatter":
            if len(numeric_columns) >= 2:
                col1, col2 = st.columns(2)
                x_column = col1.selectbox("X-axis", numeric_columns, key="scatter_x")
                y_column = col2.selectbox("Y-axis", numeric_columns, index=1, key="scatter_y")
                st.plotly_chart(scatter_plot(df, x_column, y_column), use_container_width=True)
            else:
                st.info("At least two numerical columns are required.")

        elif visualization == "Correlation":
            if len(numeric_columns) >= 2:
                st.plotly_chart(correlation_heatmap(df), use_container_width=True)
            else:
                st.info("At least two numerical columns are required.")

        elif visualization == "Categorical":
            if categorical_columns:
                column = st.selectbox("Select categorical column", categorical_columns, key="cat_bar")
                st.plotly_chart(categorical_bar(df, column), use_container_width=True)
            else:
                st.info("No categorical columns available.")

        elif visualization == "Composition":
            if categorical_columns:
                column = st.selectbox("Select categorical column", categorical_columns, key="pie_column")
                st.plotly_chart(pie_chart(df, column), use_container_width=True)
            else:
                st.info("No categorical columns available.")

        elif visualization == "Missing Values":
            fig = missing_values_chart(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found!")

        elif visualization == "Unique Values":
            st.plotly_chart(unique_values_chart(df), use_container_width=True)


