import numpy as np
import pandas as pd
from typing import List, Dict, Union, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression

class FeatureEngineer:
    """
    A comprehensive class for feature engineering, including:
    - Derived variables
    - Feature interactions
    - Principal Component Analysis
    - Custom feature creation
    - Polynomial features
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize FeatureEngineer with a dataset.
        
        Args:
            data (pd.DataFrame): Input dataset for feature engineering
        """
        self.data = data.copy()
        self.feature_info = {}  # Store information about created features
        
    def create_derived_features(self, 
                              operations: Dict[str, Dict[str, Union[List[str], str]]]) -> pd.DataFrame:
        """
        Create derived features based on mathematical operations.
        
        Args:
            operations (Dict[str, Dict]): Dictionary defining derived features
                Example: {
                    'sum_features': {
                        'columns': ['feat1', 'feat2'],
                        'operation': 'sum'
                    },
                    'ratio_features': {
                        'columns': ['feat1', 'feat2'],
                        'operation': 'ratio'
                    }
                }
        
        Returns:
            pd.DataFrame: Dataset with new derived features
        """
        for new_feature, params in operations.items():
            columns = params['columns']
            operation = params['operation'].lower()
            
            if isinstance(columns, str):
                columns = [columns]
                
            if all(col in self.data.columns for col in columns):
                if operation == 'sum':
                    self.data[new_feature] = self.data[columns].sum(axis=1)
                elif operation == 'mean':
                    self.data[new_feature] = self.data[columns].mean(axis=1)
                elif operation == 'ratio':
                    if len(columns) == 2 and (self.data[columns[1]] != 0).all():
                        self.data[new_feature] = self.data[columns[0]] / self.data[columns[1]]
                elif operation == 'difference':
                    if len(columns) == 2:
                        self.data[new_feature] = self.data[columns[0]] - self.data[columns[1]]
                elif operation == 'product':
                    self.data[new_feature] = self.data[columns].prod(axis=1)
                
                self.feature_info[new_feature] = {
                    'type': 'derived',
                    'operation': operation,
                    'source_columns': columns
                }
            
        return self.data
    
    def _ensure_numeric(self, columns: List[str]) -> pd.DataFrame:
        """
        Ensure columns are numeric, converting if necessary.
        
        Args:
            columns (List[str]): Columns to check/convert
            
        Returns:
            pd.DataFrame: Data with numeric columns
        """
        numeric_data = self.data[columns].copy()
        for col in columns:
            if not pd.api.types.is_numeric_dtype(numeric_data[col]):
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
        return numeric_data

    def create_interaction_features(self,
                                  feature_pairs: List[List[str]],
                                  interaction_type: str = 'multiplication') -> pd.DataFrame:
        """
        Create interaction features between specified pairs of features.
        
        Args:
            feature_pairs (List[List[str]]): List of feature pairs to interact
            interaction_type (str): Type of interaction ('multiplication', 'division', 'addition', 'subtraction')
            
        Returns:
            pd.DataFrame: Dataset with new interaction features
        """
        if not isinstance(feature_pairs, list):
            return self.data
            
        for pair in feature_pairs:
            if len(pair) != 2:
                continue
                
            feat1, feat2 = pair[0], pair[1]
            # Convert columns to list for safe membership testing
            columns_list = list(self.data.columns)
            if feat1 in columns_list and feat2 in columns_list:
                new_feature = f"{feat1}_{interaction_type}_{feat2}"
                
                # Convert to numeric if needed
                numeric_data = self._ensure_numeric([feat1, feat2])
                
                if interaction_type == 'multiplication':
                    self.data[new_feature] = numeric_data[feat1] * numeric_data[feat2]
                elif interaction_type == 'division' and (self.data[feat2] != 0).all():
                    self.data[new_feature] = numeric_data[feat1] / numeric_data[feat2]
                elif interaction_type == 'addition':
                    self.data[new_feature] = numeric_data[feat1] + numeric_data[feat2]
                elif interaction_type == 'subtraction':
                    self.data[new_feature] = numeric_data[feat1] - numeric_data[feat2]
                
                self.feature_info[new_feature] = {
                    'type': 'interaction',
                    'method': interaction_type,
                    'features': [feat1, feat2]
                }
                
        return self.data
    
    def apply_pca(self,
                  columns: List[str],
                  n_components: int = None,
                  variance_ratio: float = 0.95) -> pd.DataFrame:
        """
        Apply PCA to specified columns.
        
        Args:
            columns (List[str]): Columns to apply PCA to
            n_components (int): Number of components to keep
            variance_ratio (float): Minimum cumulative variance ratio to maintain
            
        Returns:
            pd.DataFrame: Dataset with PCA components added
        """
        if not columns:
            return self.data
            
        # Initialize PCA
        if n_components is None:
            pca = PCA(n_components=variance_ratio, svd_solver='full')
        else:
            pca = PCA(n_components=n_components)
            
        # Fit and transform
        pca_result = pca.fit_transform(self.data[columns])
        
        # Add PCA components to dataframe
        for i in range(pca_result.shape[1]):
            new_feature = f'pca_component_{i+1}'
            self.data[new_feature] = pca_result[:, i]
            
            self.feature_info[new_feature] = {
                'type': 'pca',
                'explained_variance_ratio': pca.explained_variance_ratio_[i],
                'cumulative_variance_ratio': sum(pca.explained_variance_ratio_[:i+1]),
                'source_columns': columns
            }
            
        return self.data
    
    def create_polynomial_features(self,
                                 columns: List[str],
                                 degree: int = 2,
                                 interaction_only: bool = False) -> pd.DataFrame:
        """
        Create polynomial features from specified columns.
        
        Args:
            columns (List[str]): Columns to create polynomial features from
            degree (int): Degree of polynomial features
            interaction_only (bool): Whether to only include interaction terms
            
        Returns:
            pd.DataFrame: Dataset with polynomial features added
        """
        poly = PolynomialFeatures(degree=degree, 
                                 interaction_only=interaction_only, 
                                 include_bias=False)
        
        # Generate polynomial features
        poly_features = poly.fit_transform(self.data[columns])
        feature_names = poly.get_feature_names_out(columns)
        
        # Add new features to dataframe
        for i, name in enumerate(feature_names):
            if name not in columns:  # Skip original features
                new_feature = f'poly_{name}'
                self.data[new_feature] = poly_features[:, i]
                
                self.feature_info[new_feature] = {
                    'type': 'polynomial',
                    'degree': degree,
                    'source_features': name
                }
                
        return self.data
    
    def create_custom_features(self,
                             feature_definitions: Dict[str, callable]) -> pd.DataFrame:
        """
        Create custom features using provided functions.
        
        Args:
            feature_definitions (Dict[str, callable]): Dictionary mapping new feature names
                to functions that generate them
                
        Returns:
            pd.DataFrame: Dataset with custom features added
        """
        for new_feature, func in feature_definitions.items():
            try:
                self.data[new_feature] = func(self.data)
                self.feature_info[new_feature] = {
                    'type': 'custom',
                    'function': func.__name__
                }
            except Exception as e:
                print(f"Error creating feature {new_feature}: {str(e)}")
                
        return self.data
    
    def select_best_features(self,
                           target: str,
                           k: int = 10) -> pd.DataFrame:
        """
        Select the k best features based on F-regression scores.
        
        Args:
            target (str): Target variable name
            k (int): Number of features to select
            
        Returns:
            pd.DataFrame: Dataset with selected features
        """
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = numeric_cols[numeric_cols != target]
        
        if len(numeric_cols) == 0:
            return self.data
            
        # Initialize selector
        selector = SelectKBest(score_func=f_regression, k=k)
        
        # Fit and transform
        X = self.data[numeric_cols]
        y = self.data[target]
        selected_features = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_names = numeric_cols[selected_mask].tolist()
        
        # Store feature scores
        for name, score in zip(numeric_cols, selector.scores_):
            if name in selected_names:
                self.feature_info[name] = {
                    'type': 'selected',
                    'f_score': score,
                    'p_value': selector.pvalues_[numeric_cols.get_loc(name)]
                }
        
        return self.data[selected_names + [target]]
    
    def get_feature_info(self) -> pd.DataFrame:
        """
        Get information about all created/selected features.
        
        Returns:
            pd.DataFrame: Summary of feature information
        """
        return pd.DataFrame.from_dict(self.feature_info, orient='index')