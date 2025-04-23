import pandas as pd
from sklearn.impute import SimpleImputer


def handle_missing_values(df, strategy="mean", fill_value=None):
    """
    Handles missing values in a DataFrame.
    """
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def impute_missing_values(df, strategy='mean', columns=None):
    """
    Impute missing values in a DataFrame.
    """
    imputer = SimpleImputer(strategy=strategy)
    if columns is None:
        columns = df.columns
    df[columns] = imputer.fit_transform(df[columns])
    return df

def remove_outliers(df, column, threshold=1.5):
    """
    Remove outliers from a DataFrame using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def drop_columns_with_high_null(df, threshold=0.5):
    """
    Drop columns with a high percentage of null values.
    """
    null_percentage = df.isnull().mean()
    columns_to_drop = null_percentage[null_percentage > threshold].index
    return df.drop(columns=columns_to_drop)

def encode_categorical_columns(df, columns):
    """
    Encode categorical columns using one-hot encoding.
    """
    return pd.get_dummies(df, columns=columns, drop_first=True)