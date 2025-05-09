import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import SelectFromModel

def select_k_best_features(X, y, k=10, score_func=f_classif):
    """
    Selects the top k features based on a scoring function.
    
    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        k (int): Number of top features to select.
        score_func (callable): Scoring function (e.g., f_classif, mutual_info_classif).
    
    Returns:
        pd.DataFrame or np.ndarray: Reduced feature matrix with top k features.
    """
    selector = SelectKBest(score_func=score_func, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, selected_features]
    return X_new

def apply_pca(X, n_components=2):
    """
    Applies Principal Component Analysis (PCA) for dimensionality reduction.
    
    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        n_components (int): Number of principal components to keep.
    
    Returns:
        np.ndarray: Transformed feature matrix with reduced dimensions.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

def select_features_ridge(X, y, alpha=1.0):
    """
    Selects features using Ridge regression.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        alpha (float): Regularization strength for Ridge regression.

    Returns:
        pd.DataFrame or np.ndarray: Reduced feature matrix with selected features.
    """
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)
    selector = SelectFromModel(ridge, prefit=True)
    X_new = selector.transform(X)
    selected_features = selector.get_support(indices=True)
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, selected_features]
    return X_new

def select_features_lasso(X, y, alpha=1.0):
    """
    Selects features using Lasso regression.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        alpha (float): Regularization strength for Lasso regression.

    Returns:
        pd.DataFrame or np.ndarray: Reduced feature matrix with selected features.
    """
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    selector = SelectFromModel(lasso, prefit=True)
    X_new = selector.transform(X)
    selected_features = selector.get_support(indices=True)
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, selected_features]
    return X_new
