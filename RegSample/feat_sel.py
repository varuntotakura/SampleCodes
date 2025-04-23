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
    
    def select_k_best(self, target: str, k: int = 10, score_func=None) -> pd.DataFrame:
        """
        Select k best features based on scoring function.
        
        Args:
            target (str): Target variable name
            k (int): Number of features to select
            score_func: Scoring function (defaults to f_regression)
            
        Returns:
            pd.DataFrame: Dataset with selected features
        """
        X, y = self._prepare_numeric_data(target)
        
        if score_func is None:
            score_func = f_regression
            
        # Initialize selector
        selector = SelectKBest(k=k)
        
        # Fit and transform
        selected_features = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_names = X.columns[selected_mask].tolist()
        
        # Store scores
        self.feature_scores['k_best'] = {
            'scores': dict(zip(X.columns, selector.scores_)),
            'selected_features': selected_names
        }
        
        # Return selected features and target
        return self.data[selected_names + [target]]
    
    def lasso_selection(self,
                       target: str,
                       alpha: float = 0.1,
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
        
        # Ensure we have at least one feature
        if len(selected_features) == 0:
            # If no features meet the threshold, select the one with the highest coefficient
            selected_features = [X.columns[np.argmax(abs(lasso.coef_))]]
        
        # Store coefficients
        self.feature_scores['lasso'] = {
            'alpha': alpha,
            'coefficients': dict(zip(X.columns, lasso.coef_)),
            'selected_features': selected_features
        }
        
        return self.data[selected_features + [target]]
    
    def recursive_feature_elimination(self,
                                   target: str,
                                   n_features_to_select: int = 5,  # Reduced from 10
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
        X, y = self._prepare_numeric_data(target)
        
        if estimator is None:
            # Use a faster estimator with limited depth and fewer estimators
            estimator = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        
        # Initialize RFE
        try:
            # Try with RandomForestRegressor (faster) first
            rfe = RFE(estimator=estimator, 
                    n_features_to_select=n_features_to_select,
                    step=step)
            
            # Fit RFE with a timeout
            import threading
            import time
            
            class TimeoutError(Exception):
                pass
            
            def timeout_handler():
                raise TimeoutError("RFE took too long to complete")
            
            # Set a timeout for RFE (60 seconds)
            timer = threading.Timer(60, timeout_handler)
            timer.start()
            
            try:
                rfe.fit(X, y)
                timer.cancel()
            except (TimeoutError, KeyboardInterrupt):
                timer.cancel()
                # If timeout occurs, fall back to SelectKBest which is faster
                print("RFE timed out, falling back to SelectKBest")
                selector = SelectKBest(f_regression, k=n_features_to_select)
                selector.fit(X, y)
                support = np.zeros(X.shape[1], dtype=bool)
                support[np.argsort(selector.scores_)[-n_features_to_select:]] = True
                ranking = np.zeros(X.shape[1], dtype=int)
                ranking[np.argsort(selector.scores_)] = np.arange(X.shape[1]) + 1
                
                # Create a simple object with the same interface as RFE
                class SimpleSelector:
                    def __init__(self, support, ranking):
                        self.support_ = support
                        self.ranking_ = ranking
                
                rfe = SimpleSelector(support, ranking)
            
            # Get selected feature names
            selected_features = X.columns[rfe.support_].tolist()
            
            # Store feature rankings
            self.feature_scores['rfe'] = {
                'rankings': dict(zip(X.columns, rfe.ranking_)),
                'selected_features': selected_features
            }
            
            # Return data with selected features and target
            return self.data[selected_features + [target]]
        
        except Exception as e:
            print(f"Error in RFE: {str(e)}, falling back to SelectKBest")
            import logging
            logging.error(f"Error in RFE: {str(e)}")
            
            # Fall back to SelectKBest which is more robust
            return self.select_k_best(target, k=n_features_to_select)
    
    def random_forest_selection(self, target: str, n_features: int = None, **kwargs) -> pd.DataFrame:
        """
        Select features using Random Forest feature importance.
        
        Args:
            target (str): Target variable column name
            n_features (int): Number of features to select
            **kwargs: Additional parameters to pass to RandomForestRegressor
            
        Returns:
            pd.DataFrame: Dataset with selected features
        """
        # Get feature matrix and target
        X = self.data.drop(columns=[target])
        y = self.data[target]
        
        # Set number of features if not specified
        if n_features is None:
            n_features = X.shape[1] // 2  # Default to half of features
        
        # Get default parameters for RandomForestRegressor
        rf_params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'random_state': kwargs.get('random_state', 42)
        }
        
        # Remove any config parameters that aren't valid for RandomForestRegressor
        rf_params = {k: v for k, v in rf_params.items() 
                    if k not in ['enable', 'threshold', 'method']}
        
        # Create and fit random forest
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X, y)
        
        # Get feature importance scores
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        # Select top n_features
        selected_features = importance['feature'].head(n_features).tolist()
        
        # Return dataset with selected features and target
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