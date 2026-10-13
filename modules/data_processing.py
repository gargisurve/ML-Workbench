import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data(uploaded_file):
    """Load CSV file into DataFrame."""
    return pd.read_csv(uploaded_file)


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

def get_dataset_info(df):

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }


# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

def get_missing_values(df):

    missing = df.isnull().sum()

    return missing[missing > 0].sort_values(
        ascending=False
    )


# --------------------------------------------------
# COLUMN TYPES
# --------------------------------------------------

def get_numeric_columns(df):

    return df.select_dtypes(
        include="number"
    ).columns.tolist()


def get_categorical_columns(df):

    return df.select_dtypes(
        exclude="number"
    ).columns.tolist()


# --------------------------------------------------
# BASIC CLEANING
# --------------------------------------------------

def remove_duplicates(df):

    return df.drop_duplicates().reset_index(drop=True)


# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------

def handle_missing_values(df, numeric_strategy="median",
                           categorical_strategy="mode"):

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns

    # Numerical columns
    for column in numeric_columns:

        if df[column].isnull().any():

            if numeric_strategy == "mean":
                df[column] = df[column].fillna(
                    df[column].mean()
                )

            elif numeric_strategy == "median":
                df[column] = df[column].fillna(
                    df[column].median()
                )

            elif numeric_strategy == "zero":
                df[column] = df[column].fillna(0)

    # Categorical columns
    for column in categorical_columns:

        if df[column].isnull().any():

            if categorical_strategy == "mode":

                mode = df[column].mode()

                if not mode.empty:
                    df[column] = df[column].fillna(
                        mode.iloc[0]
                    )

            elif categorical_strategy == "unknown":

                df[column] = df[column].fillna(
                    "Unknown"
                )

    return df


# --------------------------------------------------
# ENCODING
# --------------------------------------------------

def encode_features(X):

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    return X


# --------------------------------------------------
# SCALING
# --------------------------------------------------

def scale_features(X, method="StandardScaler"):

    X = X.copy()

    numeric_columns = X.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) == 0:
        return X, None

    if method == "StandardScaler":

        scaler = StandardScaler()

    else:

        scaler = MinMaxScaler()

    X[numeric_columns] = scaler.fit_transform(
        X[numeric_columns]
    )

    return X, scaler


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

def split_data(X, y, test_size=0.2, random_state=42):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )