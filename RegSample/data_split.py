import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from typing import Tuple, Union, Dict, List

class DataSplitter:
    """
    A comprehensive class for data splitting with various sampling methods.
    Includes simple random split, stratified sampling, and bias-based sampling.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataSplitter with a dataset.
        
        Args:
            data (pd.DataFrame): Input dataset for splitting
        """
        self.data = data
        
    def simple_random_split(self, 
                          test_size: float = 0.3,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform simple random split of data.
        
        Args:
            test_size (float): Proportion of dataset to include in the test split
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test datasets
        """
        train_data, test_data = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state
        )
        return train_data, test_data
    
    def stratified_split(self,
                        target: str,
                        n_bins: int = 10,
                        test_size: float = 0.3,
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified sampling based on target variable.
        For continuous targets, creates bins first.
        
        Args:
            target (str): Name of target column
            n_bins (int): Number of bins for continuous target
            test_size (float): Proportion of dataset to include in the test split
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test datasets
        """
        # Create bins for continuous target
        if self.data[target].dtype in ['float64', 'float32']:
            self.data['temp_bins'] = pd.qcut(self.data[target], q=n_bins, labels=False)
            strat_column = 'temp_bins'
        else:
            strat_column = target
            
        # Perform stratified split
        train_data, test_data = train_test_split(
            self.data,
            test_size=test_size,
            stratify=self.data[strat_column],
            random_state=random_state
        )
        
        # Remove temporary binning column if created
        if 'temp_bins' in train_data.columns:
            train_data = train_data.drop('temp_bins', axis=1)
            test_data = test_data.drop('temp_bins', axis=1)
            
        return train_data, test_data
    
    def bias_based_split(self,
                        columns: List[str],
                        bias_weights: Dict[str, float],
                        test_size: float = 0.3,
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform biased sampling based on specified columns and their weights.
        
        Args:
            columns (List[str]): List of columns to consider for biasing
            bias_weights (Dict[str, float]): Dictionary of column names and their bias weights
            test_size (float): Proportion of dataset to include in the test split
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test datasets
        """
        np.random.seed(random_state)
        
        # Calculate combined bias score
        bias_score = np.zeros(len(self.data))
        for col in columns:
            if col in bias_weights:
                if self.data[col].dtype in ['float64', 'float32']:
                    # Normalize numeric columns
                    normalized_values = (self.data[col] - self.data[col].mean()) / self.data[col].std()
                    bias_score += normalized_values * bias_weights[col]
                else:
                    # For categorical columns, use dummy variables
                    dummies = pd.get_dummies(self.data[col], prefix=col)
                    for dummy_col in dummies.columns:
                        bias_score += dummies[dummy_col] * bias_weights[col]
        
        # Sort by bias score and split
        n_test = int(len(self.data) * test_size)
        sorted_indices = np.argsort(bias_score)
        test_indices = sorted_indices[:n_test]
        train_indices = sorted_indices[n_test:]
        
        return self.data.iloc[train_indices], self.data.iloc[test_indices]
    
    def time_based_split(self,
                        time_column: str,
                        test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform time-based splitting for temporal data.
        
        Args:
            time_column (str): Name of the timestamp column
            test_size (float): Proportion of dataset to include in the test split
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test datasets
        """
        # Sort by time
        sorted_data = self.data.sort_values(time_column)
        
        # Calculate split point
        split_idx = int(len(sorted_data) * (1 - test_size))
        
        return sorted_data.iloc[:split_idx], sorted_data.iloc[split_idx:]
    
    def cross_validation_split(self,
                             n_splits: int = 5,
                             target: str = None,
                             random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate cross-validation splits, optionally with stratification.
        
        Args:
            n_splits (int): Number of folds
            target (str, optional): Target column for stratified splits
            random_state (int): Random seed for reproducibility
            
        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: List of (train, test) pairs
        """
        if target is not None:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = cv.split(self.data, self.data[target])
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = cv.split(self.data)
            
        cv_splits = []
        for train_idx, test_idx in splits:
            cv_splits.append((self.data.iloc[train_idx], self.data.iloc[test_idx]))
            
        return cv_splits