import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------
# HISTOGRAM
# --------------------------------------------------

def histogram(df, column):

    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}",
        marginal="box",
        template="plotly_white"
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequency"
    )

    return fig


# --------------------------------------------------
# BOX PLOT
# --------------------------------------------------

def boxplot(df, column):

    fig = px.box(
        df,
        y=column,
        title=f"Boxplot of {column}",
        points="outliers",
        template="plotly_white"
    )

    fig.update_layout(
        yaxis_title=column
    )

    return fig


# --------------------------------------------------
# DENSITY PLOT
# --------------------------------------------------

def density_plot(df, column):

    fig = px.histogram(
        df,
        x=column,
        marginal="rug",
        histnorm="probability density",
        title=f"Density Distribution of {column}",
        template="plotly_white"
    )

    return fig


# --------------------------------------------------
# SCATTER PLOT
# --------------------------------------------------

def scatter_plot(df, x_column, y_column):

    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        title=f"{x_column} vs {y_column}",
        template="plotly_white",
        opacity=0.6
    )

    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column
    )

    return fig


# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

def correlation_heatmap(df):

    numeric_df = df.select_dtypes(include="number")

    corr = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            text=corr.round(2).values,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmid=0
        )
    )

    fig.update_layout(
        title="Correlation Heatmap",
        template="plotly_white"
    )

    return fig


# --------------------------------------------------
# CATEGORICAL BAR CHART
# --------------------------------------------------

def categorical_bar(df, column):

    counts = df[column].value_counts().reset_index()

    counts.columns = [column, "Count"]

    fig = px.bar(
        counts,
        x=column,
        y="Count",
        title=f"Distribution of {column}",
        template="plotly_white"
    )

    return fig


# --------------------------------------------------
# PIE CHART
# --------------------------------------------------

def pie_chart(df, column):

    counts = df[column].value_counts().reset_index()

    counts.columns = [column, "Count"]

    fig = px.pie(
        counts,
        names=column,
        values="Count",
        title=f"Composition of {column}",
        template="plotly_white"
    )

    return fig


# --------------------------------------------------
# MISSING VALUE CHART
# --------------------------------------------------

def missing_values_chart(df):

    missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    if len(missing) == 0:
        return None

    missing_df = missing.reset_index()

    missing_df.columns = ["Column", "Missing"]

    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing",
        title="Missing Values by Column",
        template="plotly_white"
    )

    return fig


# --------------------------------------------------
# UNIQUE VALUE CHART
# --------------------------------------------------

def unique_values_chart(df):

    unique = df.nunique().sort_values(
        ascending=False
    )

    unique_df = unique.reset_index()

    unique_df.columns = ["Column", "Unique Values"]

    fig = px.bar(
        unique_df,
        x="Column",
        y="Unique Values",
        title="Unique Values per Column",
        template="plotly_white"
    )

    return fig