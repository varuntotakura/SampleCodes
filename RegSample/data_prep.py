import numpy as np
import pandas as pd
from typing import Union, Dict, List, Any, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class DataPreprocessor:
    """
    A comprehensive class for data preprocessing, focusing on handling missing values
    with various imputation strategies and column management based on missing value thresholds.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataPreprocessor with a dataset.
        
        Args:
            data (pd.DataFrame): Input dataset for preprocessing
        """
        self.data = data.copy()
        self._original_dtypes = data.dtypes
        self.preprocessing_steps = []
        self.encoders = {}
        self.imputation_stats = {}  # Store imputation statistics for reference
        self.scaling_stats = {}
        self._scalers = {}
        self._handle_data_types()
        # Set default value for use_inf_as_null if not in config
        self.use_inf_as_null = False
        
    def _handle_infinities(self):
        """Handle infinite values in the dataset based on configuration"""
        if self.use_inf_as_null:
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
        return self.data

    def _handle_data_types(self):
        """Ensure proper data types for all columns"""
        # Identify numeric columns
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        # Convert any string columns that should be numeric
        for col in self.data.columns:
            if col not in numeric_cols:
                try:
                    self.data[col] = pd.to_numeric(self.data[col])
                except (ValueError, TypeError):
                    # Keep as non-numeric if conversion fails
                    continue

    def get_missing_info(self) -> pd.DataFrame:
        """
        Get comprehensive information about missing values in the dataset.
        
        Returns:
            pd.DataFrame: Missing value statistics for each column
        """
        missing_info = pd.DataFrame({
            'missing_count': self.data.isnull().sum(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).round(2)
        })
        missing_info['dtype'] = self.data.dtypes
        return missing_info.sort_values('missing_percentage', ascending=False)
    
    def drop_high_missing_columns(self, threshold: float = 30.0) -> pd.DataFrame:
        """
        Drop columns with missing values above the specified threshold.
        
        Args:
            threshold (float): Maximum allowed percentage of missing values (default: 30.0)
            
        Returns:
            pd.DataFrame: Dataset with high-missing columns removed
        """
        missing_info = self.get_missing_info()
        columns_to_drop = missing_info[missing_info['missing_percentage'] > threshold].index.tolist()
        
        if columns_to_drop:
            self.data = self.data.drop(columns=columns_to_drop)
            print(f"Dropped {len(columns_to_drop)} columns with >{threshold}% missing values: {columns_to_drop}")
            
        return self.data
    
    def _get_column_type(self, column: str) -> str:
        """
        Determine the type of the column for appropriate imputation.
        
        Args:
            column (str): Column name to analyze
            
        Returns:
            str: Column type ('numeric', 'categorical', or 'boolean')
        """
        if self.data[column].dtype == 'bool':
            return 'boolean'
        elif pd.api.types.is_numeric_dtype(self.data[column]):
            return 'numeric'
        else:
            return 'categorical'
    
    def impute_missing_values(self, 
                            strategy: Union[str, Dict[str, str]] = 'auto',
                            custom_values: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Impute missing values using specified strategy.
        
        Args:
            strategy (Union[str, Dict[str, str]]): Imputation strategy.
                If str: 'auto', 'mean', 'median', 'mode', or 'constant'
                If dict: Column-specific strategies
            custom_values (Dict[str, Any]): Custom values for constant imputation
            
        Returns:
            pd.DataFrame: Dataset with imputed values
        """
        if strategy == 'auto':
            return self._auto_impute()
        
        if isinstance(strategy, str):
            strategy = {col: strategy for col in self.data.columns}
            
        if custom_values is None:
            custom_values = {}
            
        for column in self.data.columns:
            if self.data[column].isnull().any():
                col_strategy = strategy.get(column, 'auto')
                if col_strategy == 'auto':
                    col_type = self._get_column_type(column)
                    if col_type == 'numeric':
                        col_strategy = 'mean'
                    else:
                        col_strategy = 'mode'
                
                # Store original statistics
                self.imputation_stats[column] = {
                    'missing_count': self.data[column].isnull().sum(),
                    'strategy': col_strategy
                }
                
                if col_strategy in ['mean', 'median']:
                    value = getattr(self.data[column], col_strategy)()
                    self.data[column].fillna(value, inplace=True)
                    self.imputation_stats[column]['imputed_value'] = value
                    
                elif col_strategy == 'mode':
                    value = self.data[column].mode()[0]
                    self.data[column].fillna(value, inplace=True)
                    self.imputation_stats[column]['imputed_value'] = value
                    
                elif col_strategy == 'constant':
                    value = custom_values.get(column)
                    if value is not None:
                        self.data[column].fillna(value, inplace=True)
                        self.imputation_stats[column]['imputed_value'] = value
                        
        return self.data
    
    def _auto_impute(self) -> pd.DataFrame:
        """
        Automatically impute missing values based on column types.
        - Numeric columns: mean for normal distribution, median for skewed
        - Categorical columns: mode
        - Boolean columns: mode
        
        Returns:
            pd.DataFrame: Dataset with automatically imputed values
        """
        for column in self.data.columns:
            if self.data[column].isnull().any():
                col_type = self._get_column_type(column)
                
                if col_type == 'numeric':
                    # Check for skewness
                    non_null_values = self.data[column].dropna()
                    skewness = non_null_values.skew()
                    
                    if abs(skewness) > 1:  # Skewed distribution
                        strategy = 'median'
                    else:  # Normal distribution
                        strategy = 'mean'
                        
                    value = getattr(non_null_values, strategy)()
                    
                else:  # categorical or boolean
                    strategy = 'mode'
                    value = self.data[column].mode()[0]
                
                self.data[column].fillna(value, inplace=True)
                self.imputation_stats[column] = {
                    'missing_count': self.data[column].isnull().sum(),
                    'strategy': strategy,
                    'imputed_value': value
                }
                
        return self.data
    
    def custom_imputation(self, 
                         imputation_rules: Dict[str, Union[Any, callable]]) -> pd.DataFrame:
        """
        Apply custom imputation rules to columns.
        
        Args:
            imputation_rules (Dict[str, Union[Any, callable]]): Dictionary mapping column names to
                either a static value or a function that takes a series and returns imputed values
                
        Returns:
            pd.DataFrame: Dataset with custom imputations applied
        """
        for column, rule in imputation_rules.items():
            if column in self.data.columns:
                if callable(rule):
                    # Rule is a function
                    self.data[column] = self.data[column].apply(
                        lambda x: rule(x) if pd.isnull(x) else x
                    )
                else:
                    # Rule is a static value
                    self.data[column].fillna(rule, inplace=True)
                    
                self.imputation_stats[column] = {
                    'missing_count': self.data[column].isnull().sum(),
                    'strategy': 'custom',
                    'imputed_value': str(rule) if not callable(rule) else 'custom_function'
                }
                
        return self.data
    
    def get_imputation_summary(self) -> pd.DataFrame:
        """
        Get summary of all imputations performed.
        
        Returns:
            pd.DataFrame: Summary of imputation statistics
        """
        if not self.imputation_stats:
            return pd.DataFrame()
            
        summary = pd.DataFrame.from_dict(self.imputation_stats, orient='index')
        summary.index.name = 'column'
        return summary.reset_index()
    
    def normalize_data(self,
                      columns: List[str] = None,
                      method: str = 'minmax',
                      feature_range: tuple = (0, 1)) -> pd.DataFrame:
        """
        Normalize specified columns using various methods.
        
        Args:
            columns (List[str]): List of columns to normalize. If None, normalizes all numeric columns
            method (str): Normalization method ('minmax', 'robust')
            feature_range (tuple): Range for scaling (min, max) for minmax scaling
            
        Returns:
            pd.DataFrame: DataFrame with normalized columns
        """
        if columns is None:
            columns = [col for col in self.data.columns 
                      if self._get_column_type(col) == 'numeric']
        
        # Store original statistics for reference
        self.scaling_stats = {}
        
        for column in columns:
            if self._get_column_type(column) != 'numeric':
                print(f"Warning: Skipping non-numeric column '{column}'")
                continue
                
            original_values = self.data[column].values.reshape(-1, 1)
            
            if method == 'minmax':
                scaler = MinMaxScaler(feature_range=feature_range)
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
                
            # Store original statistics
            self.scaling_stats[column] = {
                'method': method,
                'original_mean': float(np.mean(original_values)),
                'original_std': float(np.std(original_values)),
                'original_min': float(np.min(original_values)),
                'original_max': float(np.max(original_values))
            }
            
            # Apply normalization
            self.data[column] = scaler.fit_transform(original_values).flatten()
            
            # Store scaling parameters
            if method == 'minmax':
                self.scaling_stats[column].update({
                    'scale_factor': float(scaler.scale_[0]),
                    'min_': float(scaler.min_[0]),
                    'feature_range': feature_range
                })
            elif method == 'robust':
                self.scaling_stats[column].update({
                    'center_': float(scaler.center_[0]),
                    'scale_': float(scaler.scale_[0])
                })
        
        return self.data
    
    def standardize_data(self,
                        columns: List[str] = None,
                        with_mean: bool = True,
                        with_std: bool = True) -> pd.DataFrame:
        """
        Standardize specified columns (z-score normalization).
        
        Args:
            columns (List[str]): List of columns to standardize. If None, standardizes all numeric columns
            with_mean (bool): If True, center the data before scaling
            with_std (bool): If True, scale the data to unit variance
            
        Returns:
            pd.DataFrame: DataFrame with standardized columns
        """
        if columns is None:
            columns = [col for col in self.data.columns 
                      if self._get_column_type(col) == 'numeric']
        
        # Initialize StandardScaler
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        
        # Store original statistics for reference
        self.scaling_stats = {}
        
        for column in columns:
            if self._get_column_type(column) != 'numeric':
                print(f"Warning: Skipping non-numeric column '{column}'")
                continue
                
            original_values = self.data[column].values.reshape(-1, 1)
            
            # Store original statistics
            self.scaling_stats[column] = {
                'method': 'standardization',
                'original_mean': float(np.mean(original_values)),
                'original_std': float(np.std(original_values)),
                'original_min': float(np.min(original_values)),
                'original_max': float(np.max(original_values)),
                'with_mean': with_mean,
                'with_std': with_std
            }
            
            # Apply standardization
            self.data[column] = scaler.fit_transform(original_values).flatten()
            
            # Store scaling parameters
            self.scaling_stats[column].update({
                'scale_factor': float(scaler.scale_[0]) if with_std else 1.0,
                'mean_': float(scaler.mean_[0]) if with_mean else 0.0
            })
        
        return self.data
    
    def handle_outliers(self, columns: List[str] = None, method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in specified columns
        
        Args:
            columns (List[str]): Columns to process
            method (str): 'iqr' or 'zscore'
            threshold (float): Threshold for outlier detection
        
        Returns:
            pd.DataFrame: Data with handled outliers
        """
        if columns is None:
            columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers at bounds
                self.data[col] = self.data[col].clip(lower_bound, upper_bound)
            
            elif method == 'zscore':
                mean = self.data[col].mean()
                std = self.data[col].std()
                z_scores = (self.data[col] - mean) / std
                
                # Cap values beyond threshold standard deviations
                self.data[col] = self.data[col].mask(
                    abs(z_scores) > threshold,
                    mean + (threshold * std * np.sign(z_scores))
                )
        
        return self.data
    
    def encode_categorical(self, columns: List[str] = None, method: str = 'onehot',
                         drop_first: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            columns (List[str]): Columns to encode
            method (str): 'onehot' or 'label'
            drop_first (bool): Whether to drop first category in one-hot encoding
        
        Returns:
            pd.DataFrame: Data with encoded categories
        """
        if columns is None:
            columns = self.data.select_dtypes(include=['object', 'category']).columns
        
        for col in columns:
            if method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=drop_first)
                self.data = pd.concat([self.data.drop(columns=[col]), dummies], axis=1)
            elif method == 'label':
                # Label encoding
                self.data[col] = pd.Categorical(self.data[col]).codes
        
        return self.data
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after preprocessing"""
        return self.data.columns.tolist()
    
    def get_preprocessed_data(self) -> pd.DataFrame:
        """Get the preprocessed data"""
        return self.data.copy()
    
    def get_scalers(self) -> Dict[str, Any]:
        """Get fitted scalers"""
        return self._scalers.copy()
    
    def get_scaling_summary(self) -> pd.DataFrame:
        """
        Get summary of all scaling operations performed.
        
        Returns:
            pd.DataFrame: Summary of scaling statistics
        """
        if not hasattr(self, 'scaling_stats'):
            return pd.DataFrame()
            
        summary = pd.DataFrame.from_dict(self.scaling_stats, orient='index')
        summary.index.name = 'column'
        return summary.reset_index()
    
    def _handle_categorical_columns(self):
        """Encode categorical columns using LabelEncoder"""
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                # Convert column to numeric using LabelEncoder
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col].astype(str))
                self.encoders[col] = encoder
            
            self.preprocessing_steps.append('categorical_encoding')
        
        return self.data

    def preprocess_data(self):
        """Run all preprocessing steps"""
        self._handle_infinities()
        self.impute_missing_values()
        self._handle_categorical_columns()  # Add categorical handling
        self.standardize_data()
        return self.data