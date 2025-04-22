import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def handle_missing_values(df, strategy="mean", fill_value=None):
    """
    Handles missing values in a DataFrame.
    """
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def scale_features(df, method="standard"):
    """
    Scales numerical features using StandardScaler or MinMaxScaler.
    """
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

def one_hot_encode(df, columns):
    """
    Performs one-hot encoding on specified categorical columns.
    """
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns))
    return pd.concat([df.drop(columns, axis=1), encoded_df], axis=1)

def create_interaction_features(df, columns):
    """
    Creates interaction features by multiplying specified columns.
    """
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_name = f"{columns[i]}_x_{columns[j]}"
            df[col_name] = df[columns[i]] * df[columns[j]]
    return df

def bin_numerical_feature(df, column, bins, labels=None):
    """
    Bins a numerical feature into discrete intervals.
    """
    df[f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=labels)
    return df

def log_transform(df, column):
    """
    Applies log transformation to a numerical column.
    """
    df[f"{column}_log"] = np.log1p(df[column])
    return df

def normalize_feature(df, column):
    """
    Normalizes a numerical feature to a range of [0, 1].
    """
    min_val = df[column].min()
    max_val = df[column].max()
    df[f"{column}_normalized"] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_feature(df, column):
    """
    Standardizes a numerical feature to have a mean of 0 and standard deviation of 1.
    """
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[f"{column}_standardized"] = (df[column] - mean_val) / std_val
    return df