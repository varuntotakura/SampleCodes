import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Any
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Lasso, Ridge,
    ElasticNet
)
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor,
    BaggingRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import (
    RandomizedSearchCV, GridSearchCV,
    KFold, cross_val_score, train_test_split,
    learning_curve, validation_curve
)
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    explained_variance_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

class ModelBuilder:
    """
    A comprehensive class for building and tuning various regression models
    """
    def __init__(self):
        """Initialize ModelBuilder with available models and their parameter grids"""
        self.models = {}
        self.results = {}
        self.best_params = {}
        
        # Define comprehensive parameter grids for each model
        self.param_grids = {
            'linear': {
                'fit_intercept': [True, False],
                'copy_X': [True, False],
                'n_jobs': [None, -1],
                'positive': [True, False]
            },
            
            'logistic': {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'fit_intercept': [True, False],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [100, 200, 500, 1000],
                'multi_class': ['auto', 'ovr', 'multinomial'],
                'warm_start': [True, False],
                'l1_ratio': [None, 0.0, 0.1, 0.5, 1.0]
            },
            
            'lasso': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                'fit_intercept': [True, False],
                'max_iter': [100, 500, 1000, 2000],
                'tol': [0.0001, 0.001, 0.01],
                'warm_start': [True, False],
                'selection': ['cyclic', 'random'],
                'random_state': [None, 42]
            },
            
            'ridge': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                'fit_intercept': [True, False],
                'max_iter': [None, 500, 1000],
                'tol': [0.0001, 0.001, 0.01],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                'random_state': [None, 42]
            },
            
            'elastic_net': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'fit_intercept': [True, False],
                'max_iter': [100, 500, 1000],
                'tol': [0.0001, 0.001, 0.01],
                'warm_start': [True, False],
                'random_state': [None, 42],
                'selection': ['cyclic', 'random']
            },
            
            'random_forest': {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2'],
                'bootstrap': [True, False],
                'criterion': ['squared_error', 'absolute_error', 'poisson'],
                'max_leaf_nodes': [None, 10, 20, 30],
                'min_impurity_decrease': [0.0, 0.01, 0.1],
                'warm_start': [True, False],
                'n_jobs': [None, -1],
                'random_state': [None, 42]
            },
            
            'svr': {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2],
                'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1],
                'tol': [0.0001, 0.001, 0.01],
                'shrinking': [True, False],
                'cache_size': [100, 200, 500],
                'max_iter': [-1, 1000, 2000]
            },
            
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 20, 30, 40],
                'p': [1, 2, 3],  # Manhattan, Euclidean, and Minkowski
                'metric': ['minkowski', 'euclidean', 'manhattan'],
                'n_jobs': [None, -1]
            },
            
            'adaboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0],
                'loss': ['linear', 'square', 'exponential'],
                'random_state': [None, 42]
            },
            
            'bagging': {
                'n_estimators': [10, 20, 50],
                'max_samples': [0.5, 0.7, 1.0],
                'max_features': [0.5, 0.7, 1.0],
                'bootstrap': [True, False],
                'bootstrap_features': [True, False],
                'n_jobs': [None, -1],
                'random_state': [None, 42],
                'warm_start': [True, False]
            },
            
            'xgboost': {
                'n_estimators': [100, 200, 500],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'colsample_bylevel': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 1.0],
                'reg_lambda': [0, 0.1, 1.0],
                'tree_method': ['auto', 'exact', 'approx', 'hist'],
                'booster': ['gbtree', 'gblinear', 'dart'],
                'random_state': [None, 42],
                'n_jobs': [None, -1]
            },
            
            'gradient_boosting': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.6, 0.8, 1.0],
                'max_features': ['auto', 'sqrt', 'log2'],
                'criterion': ['friedman_mse', 'squared_error'],
                'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                'random_state': [None, 42],
                'warm_start': [True, False],
                'validation_fraction': [0.1, 0.2],
                'n_iter_no_change': [None, 5, 10],
                'tol': [0.0001, 0.001, 0.01]
            }
        }
    
    def create_model(self, 
                    model_type: str,
                    custom_params: Dict = None) -> Any:
        """
        Create a model instance with specified parameters.
        
        Args:
            model_type (str): Type of model to create
            custom_params (Dict): Custom parameters for the model
            
        Returns:
            Any: Created model instance
        """
        params = custom_params if custom_params is not None else {}
        
        # Filter out problematic parameters
        params = {k: v for k, v in params.items() if not k.startswith('mode.')}
        if 'use_inf_as_null' in params:
            del params['use_inf_as_null']
        
        if model_type == 'linear':
            return LinearRegression(**params)
        elif model_type == 'logistic':
            return LogisticRegression(**params)
        elif model_type == 'lasso':
            return Lasso(**params)
        elif model_type == 'ridge':
            return Ridge(**params)
        elif model_type == 'elastic_net':
            return ElasticNet(**params)
        elif model_type == 'random_forest':
            return RandomForestRegressor(**params)
        elif model_type == 'svr':
            return SVR(**params)
        elif model_type == 'knn':
            return KNeighborsRegressor(**params)
        elif model_type == 'adaboost':
            return AdaBoostRegressor(**params)
        elif model_type == 'bagging':
            return BaggingRegressor(**params)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(**params)
        elif model_type == 'xgboost':
            return self.create_xgboost_model(params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_xgboost_model(self, params=None):
        """
        Create an XGBoost regression model with specified parameters.
        
        Args:
            params (dict): Model parameters
            
        Returns:
            XGBRegressor: Configured XGBoost model
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'random_state': 42,
            'tree_method': 'hist',  # Use histogram-based tree method
            'enable_categorical': False  # Disable categorical feature support
        }
        
        if params:
            default_params.update(params)
        
        return XGBRegressor(**default_params)
    
    def bias_variance_analysis(self,
                             X: np.ndarray,
                             y: np.ndarray,
                             model_type: str,
                             params: Dict = None,
                             n_iterations: int = 100,
                             test_size: float = 0.2) -> Dict:
        """
        Perform bias-variance analysis through repeated sampling.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            model_type (str): Type of model to analyze
            params (Dict): Model parameters
            n_iterations (int): Number of bootstrap iterations
            test_size (float): Proportion of test set
            
        Returns:
            Dict: Bias-variance decomposition results
        """
        predictions = []
        
        for _ in range(n_iterations):
            # Split data randomly
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size
            )
            
            # Create and train model
            model = self.create_model(model_type, params)
            model.fit(X_train, y_train)
            
            # Make predictions
            pred = model.predict(X_test)
            predictions.append((y_test, pred))
        
        # Calculate bias and variance
        actual = np.array([p[0] for p in predictions])
        predicted = np.array([p[1] for p in predictions])
        
        bias = np.mean((np.mean(predicted, axis=0) - actual[0])**2)
        variance = np.mean([np.mean((pred - np.mean(predicted, axis=0))**2)
                          for pred in predicted])
        
        return {
            'bias': bias,
            'variance': variance,
            'total_error': bias + variance
        }
    
    def random_search_cv(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        model_type: str,
                        param_distributions: Dict = None,
                        n_iter: int = 100,
                        cv: int = 5) -> Dict:
        """
        Perform random search cross-validation.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            model_type (str): Type of model to tune
            param_distributions (Dict): Parameter distributions
            n_iter (int): Number of parameter settings to try
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict: Best parameters and scores
        """
        if param_distributions is None:
            param_distributions = self.param_grids[model_type]
            
        model = self.create_model(model_type)
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        random_search.fit(X, y)
        
        self.best_params[f'{model_type}_random'] = random_search.best_params_
        return {
            'best_params': random_search.best_params_,
            'best_score': -random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
    
    def grid_search_cv(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      model_type: str,
                      param_grid: Dict = None,
                      cv: int = 5) -> Dict:
        """
        Perform grid search cross-validation.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            model_type (str): Type of model to tune
            param_grid (Dict): Parameter grid
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict: Best parameters and scores
        """
        if param_grid is None:
            param_grid = self.param_grids[model_type]
            
        model = self.create_model(model_type)
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        self.best_params[f'{model_type}_grid'] = grid_search.best_params_
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def kfold_cv_evaluation(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          model_type: str,
                          params: Dict = None,
                          n_splits: int = 5) -> Dict:
        """
        Perform k-fold cross-validation evaluation.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            model_type (str): Type of model to evaluate
            params (Dict): Model parameters
            n_splits (int): Number of folds
            
        Returns:
            Dict: Cross-validation results
        """
        model = self.create_model(model_type, params)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        scores = {
            'mse': [],
            'r2': [],
            'mae': [],
            'explained_variance': []
        }
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            scores['mse'].append(mean_squared_error(y_val, y_pred))
            scores['r2'].append(r2_score(y_val, y_pred))
            scores['mae'].append(mean_absolute_error(y_val, y_pred))
            scores['explained_variance'].append(explained_variance_score(y_val, y_pred))
        
        return {
            'mean_scores': {k: np.mean(v) for k, v in scores.items()},
            'std_scores': {k: np.std(v) for k, v in scores.items()},
            'all_scores': scores
        }
    
    def train_evaluate_model(self, X_train, X_test, y_train, y_test,
                            model_type: str = 'xgboost',
                            params: Dict[str, Any] = None) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate a model.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            model_type (str): Type of model to train
            params (Dict[str, Any]): Model parameters
            
        Returns:
            Dict[str, Dict[str, float]]: Training and test results
        """
        if params is None:
            params = {}
            
        # Convert data to numpy arrays and ensure float32 type for XGBoost
        X_train = np.asarray(X_train).astype(np.float32)
        X_test = np.asarray(X_test).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)
        
        # Initialize and train model
        model = self._initialize_model(model_type, params)
        
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logging.error(f"Error training model {model_type}: {str(e)}")
            raise
            
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_results = self._calculate_metrics(y_train, train_pred)
        test_results = self._calculate_metrics(y_test, test_pred)
        
        # Store model
        self.models[model_type] = model
        
        return {
            'train': train_results,
            'test': test_results
        }
    
    def save_model(self, model_type: str, file_path: str) -> None:
        """
        Save a trained model to disk using joblib.
        Args:
            model_type (str): The type/name of the model to save.
            file_path (str): Path to save the model file.
        """
        if model_type not in self.models:
            raise ValueError(f"Model '{model_type}' not found. Train the model before saving.")
        joblib.dump(self.models[model_type], file_path)
        
    @staticmethod
    def save_model(model, file_path: str) -> None:
        """
        Static method to save any model to disk using joblib.
        Args:
            model: The model object to save
            file_path (str): Path to save the model file.
        """
        joblib.dump(model, file_path)

    @staticmethod
    def load_model(file_path: str):
        """
        Load a model from disk using joblib.
        Args:
            file_path (str): Path to the saved model file.
        Returns:
            The loaded model object.
        """
        return joblib.load(file_path)

    def plot_learning_curves(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           model_type: str,
                           params: Dict = None,
                           train_sizes: np.ndarray = None) -> None:
        """
        Plot learning curves for a model.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            model_type (str): Type of model to analyze
            params (Dict): Model parameters
            train_sizes (np.ndarray): Training set sizes to evaluate
        """
        if (train_sizes is None):
            train_sizes = np.linspace(0.1, 1.0, 10)
            
        model = self.create_model(model_type, params)
        
        # Calculate learning curves
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Calculate mean and std
        train_mean = -np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = -np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, label='Training score')
        plt.fill_between(train_sizes_abs,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.1)
        plt.plot(train_sizes_abs, val_mean, label='Cross-validation score')
        plt.fill_between(train_sizes_abs,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Mean Squared Error')
        plt.title(f'Learning Curves ({model_type})')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    def plot_validation_curve(self,
                            X: np.ndarray,
                            y: np.ndarray,
                            model_type: str,
                            param_name: str,
                            param_range: List,
                            params: Dict = None) -> None:
        """
        Plot validation curve for a specific parameter.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            model_type (str): Type of model to analyze
            param_name (str): Name of parameter to vary
            param_range (List): Range of parameter values to try
            params (Dict): Other model parameters
        """
        model = self.create_model(model_type, params)
        
        # Calculate validation curve
        train_scores, val_scores = validation_curve(
            model, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Calculate mean and std
        train_mean = -np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = -np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot validation curve
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, label='Training score')
        plt.fill_between(param_range,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.1)
        plt.plot(param_range, val_mean, label='Cross-validation score')
        plt.fill_between(param_range,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.1)
        plt.xlabel(param_name)
        plt.ylabel('Mean Squared Error')
        plt.title(f'Validation Curve ({model_type})')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    def _initialize_model(self, model_type: str, params: Dict = None) -> Any:
        """Initialize a model with given parameters.
        
        Args:
            model_type (str): Type of model to initialize
            params (Dict): Model parameters
            
        Returns:
            Any: Initialized model instance
        """
        if params is None:
            params = {}
            
        # Filter out problematic parameters
        params = {k: v for k, v in params.items() if not k.startswith('mode.')}
        if 'use_inf_as_null' in params:
            del params['use_inf_as_null']
        
        # For XGBoost, explicitly set tree_method and disable categorical support
        if model_type == 'xgboost':
            params['tree_method'] = params.get('tree_method', 'hist')
            params['enable_categorical'] = False
            
        return self.create_model(model_type, params)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'predictions': y_pred
        }