import pandas as pd
from sklearn.impute import SimpleImputer

def impute_missing_values(df, strategy='mean', columns=None):
    """
    Impute missing values in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    - strategy (str): The imputation strategy ('mean', 'median', 'most_frequent', 'constant').
    - columns (list): List of columns to impute. If None, all columns are considered.

    Returns:
    - pd.DataFrame: DataFrame with imputed values.
    """
    imputer = SimpleImputer(strategy=strategy)
    if columns is None:
        columns = df.columns
    df[columns] = imputer.fit_transform(df[columns])
    return df

def remove_outliers(df, column, threshold=1.5):
    """
    Remove outliers from a DataFrame using the IQR method.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    - column (str): The column to check for outliers.
    - threshold (float): The IQR multiplier to define outliers.

    Returns:
    - pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    """
    Normalize a column in a DataFrame to a range of 0 to 1.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    - column (str): The column to normalize.

    Returns:
    - pd.DataFrame: DataFrame with the normalized column.
    """
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def drop_columns_with_high_null(df, threshold=0.5):
    """
    Drop columns with a high percentage of null values.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    - threshold (float): The null percentage threshold (e.g., 0.5 for 50%).

    Returns:
    - pd.DataFrame: DataFrame with columns dropped.
    """
    null_percentage = df.isnull().mean()
    columns_to_drop = null_percentage[null_percentage > threshold].index
    return df.drop(columns=columns_to_drop)

def encode_categorical_columns(df, columns):
    """
    Encode categorical columns using one-hot encoding.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    - columns (list): List of categorical columns to encode.

    Returns:
    - pd.DataFrame: DataFrame with encoded columns.
    """
    return pd.get_dummies(df, columns=columns, drop_first=True)