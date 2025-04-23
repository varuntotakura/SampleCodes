from typing import List, Optional, Union, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import category_encoders as ce

class AdvancedPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 numeric_features: List[str] = None,
                 categorical_features: List[str] = None,
                 datetime_features: List[str] = None,
                 scaler_type: str = 'standard',
                 imputer_type: str = 'simple',
                 categorical_encoder: str = 'target',
                 feature_selection: Optional[str] = None,
                 n_features_to_select: Optional[int] = None,
                 pca_components: Optional[Union[int, float]] = None,
                 handle_imbalance: Optional[str] = None,
                 sampling_strategy: Union[str, float] = 'auto'):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.datetime_features = datetime_features
        self.scaler_type = scaler_type
        self.imputer_type = imputer_type
        self.categorical_encoder = categorical_encoder
        self.feature_selection = feature_selection
        self.n_features_to_select = n_features_to_select
        self.pca_components = pca_components
        self.handle_imbalance = handle_imbalance
        self.sampling_strategy = sampling_strategy
        
        self._initialize_components()
        
    def _initialize_components(self):
        # Initialize scalers
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            
        # Initialize imputers
        if self.imputer_type == 'simple':
            self.numeric_imputer = SimpleImputer(strategy='mean')
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        elif self.imputer_type == 'knn':
            self.numeric_imputer = KNNImputer(n_neighbors=5)
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            
        # Initialize categorical encoder
        if self.categorical_encoder == 'target':
            self.encoder = ce.TargetEncoder()
        elif self.categorical_encoder == 'woe':
            self.encoder = ce.WOEEncoder()
        elif self.categorical_encoder == 'count':
            self.encoder = ce.CountEncoder()
            
        # Initialize feature selector
        if self.feature_selection == 'mutual_info':
            self.selector = None  # Will be initialized during fit
        elif self.feature_selection == 'model_based':
            self.selector = None  # Will be initialized with model
            
        # Initialize PCA
        if self.pca_components:
            self.pca = PCA(n_components=self.pca_components)
        else:
            self.pca = None
            
        # Initialize imbalance handler
        if self.handle_imbalance == 'smote':
            self.sampler = SMOTE(sampling_strategy=self.sampling_strategy)
        elif self.handle_imbalance == 'adasyn':
            self.sampler = ADASYN(sampling_strategy=self.sampling_strategy)
        elif self.handle_imbalance == 'random_under':
            self.sampler = RandomUnderSampler(sampling_strategy=self.sampling_strategy)
        elif self.handle_imbalance == 'smote_tomek':
            self.sampler = SMOTETomek(sampling_strategy=self.sampling_strategy)
        else:
            self.sampler = None

    def _handle_datetime(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.datetime_features:
            return X
            
        X = X.copy()
        for feature in self.datetime_features:
            X[f"{feature}_year"] = pd.to_datetime(X[feature]).dt.year
            X[f"{feature}_month"] = pd.to_datetime(X[feature]).dt.month
            X[f"{feature}_day"] = pd.to_datetime(X[feature]).dt.day
            X[f"{feature}_dayofweek"] = pd.to_datetime(X[feature]).dt.dayofweek
            X = X.drop(columns=[feature])
        return X

    def fit(self, X: pd.DataFrame, y=None):
        # Handle datetime features
        X = self._handle_datetime(X)
        
        # Impute missing values
        if self.numeric_features:
            self.numeric_imputer.fit(X[self.numeric_features])
        if self.categorical_features:
            self.categorical_imputer.fit(X[self.categorical_features])
            
        # Scale numeric features
        if self.numeric_features:
            self.scaler.fit(self.numeric_imputer.transform(X[self.numeric_features]))
            
        # Encode categorical features
        if self.categorical_features:
            self.encoder.fit(X[self.categorical_features], y)
            
        # Feature selection
        if self.feature_selection == 'mutual_info' and y is not None:
            scores = mutual_info_classif(X, y)
            self.selected_features_ = X.columns[np.argsort(scores)[-self.n_features_to_select:]]
            
        # PCA
        if self.pca is not None:
            self.pca.fit(X)
            
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        
        # Handle datetime features
        X = self._handle_datetime(X)
        
        # Impute missing values
        if self.numeric_features:
            X[self.numeric_features] = self.numeric_imputer.transform(X[self.numeric_features])
        if self.categorical_features:
            X[self.categorical_features] = self.categorical_imputer.transform(X[self.categorical_features])
            
        # Scale numeric features
        if self.numeric_features:
            X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
            
        # Encode categorical features
        if self.categorical_features:
            encoded_cats = self.encoder.transform(X[self.categorical_features])
            X = X.drop(columns=self.categorical_features)
            X = pd.concat([X, encoded_cats], axis=1)
            
        # Feature selection
        if self.feature_selection and hasattr(self, 'selected_features_'):
            X = X[self.selected_features_]
            
        # PCA
        if self.pca is not None:
            X = self.pca.transform(X)
            
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        self.fit(X, y)
        X_transformed = self.transform(X)
        
        # Handle imbalance
        if self.sampler and y is not None:
            X_transformed, y = self.sampler.fit_resample(X_transformed, y)
            
        return X_transformed