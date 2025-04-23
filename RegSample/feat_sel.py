import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple
from sklearn.feature_selection import (
    SelectKBest, f_regression, RFE, SelectFromModel,
    mutual_info_regression
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

class FeatureSelector:
    """
    A comprehensive class for feature selection using various methods:
    - K-Best Features (based on mutual information or F-scores)
    - Lasso Regularization
    - Recursive Feature Elimination (RFE)
    - Random Forest Importance
    - Multicollinearity Analysis (VIF)
    - KNN-based Feature Selection
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize FeatureSelector with a dataset.
        
        Args:
            data (pd.DataFrame): Input dataset for feature selection
        """
        self.data = data.copy()
        self.selection_results = {}  # Store selection results for each method
        self.feature_scores = {}     # Store feature importance scores
        
    def _prepare_numeric_data(self, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare numeric data for feature selection.
        
        Args:
            target (str): Name of target variable
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variable
        """
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = numeric_cols[numeric_cols != target]
        
        X = self.data[numeric_cols]
        y = self.data[target]
        
        return X, y
    
    def select_k_best(self,
                     target: str,
                     k: int = 10,
                     method: str = 'f_regression') -> pd.DataFrame:
        """
        Select k best features using various scoring methods.
        
        Args:
            target (str): Target variable name
            k (int): Number of features to select
            method (str): Scoring method ('f_regression' or 'mutual_info')
            
        Returns:
            pd.DataFrame: Dataset with selected features
        """
        X, y = self._prepare_numeric_data(target)
        
        if method == 'f_regression':
            scorer = f_regression
        elif method == 'mutual_info':
            scorer = mutual_info_regression
        else:
            raise ValueError(f"Unknown scoring method: {method}")
            
        selector = SelectKBest(score_func=scorer, k=k)
        selector.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store scores
        self.feature_scores['k_best'] = {
            'method': method,
            'scores': dict(zip(X.columns, selector.scores_)),
            'selected_features': selected_features
        }
        
        return self.data[selected_features + [target]]
    
    def lasso_selection(self,
                       target: str,
                       alpha: float = None,
                       threshold: float = 1e-5) -> pd.DataFrame:
        """
        Select features using Lasso regularization.
        
        Args:
            target (str): Target variable name
            alpha (float): Regularization parameter (if None, uses CV)
            threshold (float): Coefficient threshold for feature selection
            
        Returns:
            pd.DataFrame: Dataset with selected features
        """
        X, y = self._prepare_numeric_data(target)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if alpha is None:
            # Use cross-validation to find optimal alpha
            lasso_cv = LassoCV(cv=5, random_state=42)
            lasso_cv.fit(X_scaled, y)
            alpha = lasso_cv.alpha_
        
        # Fit Lasso with optimal/specified alpha
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_scaled, y)
        
        # Get selected features
        selected_features = X.columns[abs(lasso.coef_) > threshold].tolist()
        
        # Store coefficients
        self.feature_scores['lasso'] = {
            'alpha': alpha,
            'coefficients': dict(zip(X.columns, lasso.coef_)),
            'selected_features': selected_features
        }
        
        return self.data[selected_features + [target]]
    
    def recursive_feature_elimination(self,
                                   target: str,
                                   n_features_to_select: int = 10,
                                   step: int = 1,
                                   estimator=None) -> pd.DataFrame:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            target (str): Target variable name
            n_features_to_select (int): Number of features to select
            step (int): Number of features to remove at each iteration
            estimator: Optional custom estimator (defaults to RandomForestRegressor)
            
        Returns:
            pd.DataFrame: Dataset with selected features
        """
        X = self.data.drop(columns=[target])
        y = self.data[target]
        
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Initialize RFE
        rfe = RFE(estimator=estimator, 
                  n_features_to_select=n_features_to_select,
                  step=step)
        
        # Fit RFE
        rfe.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[rfe.support_].tolist()
        
        # Store feature rankings
        self.feature_scores['rfe'] = {
            'rankings': dict(zip(X.columns, rfe.ranking_)),
            'selected_features': selected_features
        }
        
        # Return data with selected features and target
        return self.data[selected_features + [target]]
    
    def random_forest_selection(self,
                              target: str,
                              n_features: int = 10) -> pd.DataFrame:
        """
        Select features using Random Forest importance.
        
        Args:
            target (str): Target variable name
            n_features (int): Number of features to select
            
        Returns:
            pd.DataFrame: Dataset with selected features
        """
        X, y = self._prepare_numeric_data(target)
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance scores
        importance_scores = rf.feature_importances_
        feature_importance = pd.Series(importance_scores, index=X.columns)
        
        # Select top features
        selected_features = feature_importance.nlargest(n_features).index.tolist()
        
        # Store importance scores
        self.feature_scores['random_forest'] = {
            'importance_scores': dict(zip(X.columns, importance_scores)),
            'selected_features': selected_features
        }
        
        return self.data[selected_features + [target]]
    
    def multicollinearity_analysis(self,
                                 target: str = None,
                                 threshold: float = 5.0) -> pd.DataFrame:
        """
        Analyze and remove multicollinear features using VIF.
        
        Args:
            target (str): Target variable name (optional)
            threshold (float): VIF threshold for feature removal
            
        Returns:
            pd.DataFrame: Dataset with non-multicollinear features
        """
        X, _ = self._prepare_numeric_data(target) if target else (self.data, None)
        
        # Calculate initial VIF for all features
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(X.shape[1])]
        
        # Iteratively remove features with high VIF
        high_vif_features = []
        while True:
            max_vif = vif_data["VIF"].max()
            if max_vif <= threshold:
                break
                
            # Remove feature with highest VIF
            feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
            high_vif_features.append((feature_to_remove, max_vif))
            
            # Recalculate VIF
            X = X.drop(columns=[feature_to_remove])
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                              for i in range(X.shape[1])]
        
        # Store VIF analysis results
        self.feature_scores['vif'] = {
            'initial_vif': dict(zip(vif_data["Feature"], vif_data["VIF"])),
            'removed_features': dict(high_vif_features),
            'selected_features': X.columns.tolist()
        }
        
        return self.data[X.columns.tolist() + ([target] if target else [])]
    
    def knn_feature_selection(self,
                            target: str,
                            n_features: int = 10,
                            n_neighbors: int = 5) -> pd.DataFrame:
        """
        Select features using KNN-based importance scoring.
        
        Args:
            target (str): Target variable name
            n_features (int): Number of features to select
            n_neighbors (int): Number of neighbors for KNN
            
        Returns:
            pd.DataFrame: Dataset with selected features
        """
        X, y = self._prepare_numeric_data(target)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize scores dictionary
        feature_scores = {}
        
        # Calculate importance score for each feature
        for feature in X.columns:
            # Train KNN on single feature
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            X_feature = X_scaled[:, X.columns.get_loc(feature)].reshape(-1, 1)
            knn.fit(X_feature, y)
            
            # Calculate score (R² score)
            score = knn.score(X_feature, y)
            feature_scores[feature] = score
        
        # Select top features
        selected_features = pd.Series(feature_scores).nlargest(n_features).index.tolist()
        
        # Store scores
        self.feature_scores['knn'] = {
            'scores': feature_scores,
            'selected_features': selected_features
        }
        
        return self.data[selected_features + [target]]
    
    def get_selection_summary(self) -> Dict:
        """
        Get summary of feature selection results from all methods.
        
        Returns:
            Dict: Dictionary containing selection results for each method
        """
        return self.feature_scores