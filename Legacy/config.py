from typing import Dict, Any
import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Default hyperparameters for different models
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': RANDOM_SEED
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'random_state': RANDOM_SEED
    },
    'catboost': {
        'iterations': 100,
        'depth': 6,
        'learning_rate': 0.1,
        'l2_leaf_reg': 3,
        'random_strength': 1,
        'random_state': RANDOM_SEED
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 6,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': RANDOM_SEED
    }
}

# Feature engineering configs
FEATURE_ENGINEERING_CONFIGS = {
    'numeric_features': {
        'imputation_strategy': 'mean',
        'scaling': 'standard'  # Options: 'standard', 'minmax', 'robust'
    },
    'categorical_features': {
        'imputation_strategy': 'most_frequent',
        'encoding': 'label'  # Options: 'label', 'onehot', 'target'
    }
}

# Model selection and evaluation configs
MODEL_SELECTION_CONFIG = {
    'cv_folds': 5,
    'scoring': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'refit': 'roc_auc'
}

# Class imbalance handling configs
IMBALANCE_CONFIGS = {
    'methods': {
        'smote': {'sampling_strategy': 'auto'},
        'adasyn': {'sampling_strategy': 'auto'},
        'random_under': {'sampling_strategy': 'auto'}
    }
}

# Feature selection configs
FEATURE_SELECTION_CONFIG = {
    'methods': {
        'from_model': {'threshold': 'median'},
        'rfe': {'n_features_to_select': 'auto'},
        'mutual_info': {'percentile': 20}
    }
}

# Experiment tracking configs
MLFLOW_CONFIG = {
    'tracking_uri': 'mlruns',
    'experiment_name': 'classification_pipeline'
}