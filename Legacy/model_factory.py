import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Union, Dict, Any
from config import MODEL_CONFIGS, RANDOM_SEED

class ModelFactory:
    @staticmethod
    def create_model(
        model_name: str,
        problem_type: str = 'classification',
        custom_params: Dict[str, Any] = None,
        use_gpu: bool = False
    ) -> Union[xgb.XGBModel, lgb.LGBMModel, cb.CatBoost, RandomForestClassifier, RandomForestRegressor]:
        """
        Create a model instance based on name and parameters.
        
        Args:
            model_name: Name of the model ('xgboost', 'lightgbm', 'catboost', 'random_forest')
            problem_type: 'classification' or 'regression'
            custom_params: Optional custom parameters to override defaults
            use_gpu: Whether to use GPU acceleration if available
        """
        # Get default parameters
        params = MODEL_CONFIGS.get(model_name, {}).copy()
        
        # Update with custom parameters if provided
        if custom_params:
            params.update(custom_params)
            
        # Add GPU configuration if requested
        if use_gpu:
            if model_name == 'xgboost':
                params['tree_method'] = 'gpu_hist'
            elif model_name == 'lightgbm':
                params['device'] = 'gpu'
            elif model_name == 'catboost':
                params['task_type'] = 'GPU'

        # Create and return the appropriate model
        if model_name == 'xgboost':
            if problem_type == 'classification':
                return xgb.XGBClassifier(**params)
            return xgb.XGBRegressor(**params)
            
        elif model_name == 'lightgbm':
            if problem_type == 'classification':
                return lgb.LGBMClassifier(**params)
            return lgb.LGBMRegressor(**params)
            
        elif model_name == 'catboost':
            if problem_type == 'classification':
                return cb.CatBoostClassifier(**params)
            return cb.CatBoostRegressor(**params)
            
        elif model_name == 'random_forest':
            if problem_type == 'classification':
                return RandomForestClassifier(**params)
            return RandomForestRegressor(**params)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def get_param_grid(model_name: str) -> Dict[str, Any]:
        """Get hyperparameter search grid for the specified model."""
        from scipy.stats import uniform, randint
        
        if model_name == 'xgboost':
            return {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'min_child_weight': randint(1, 7),
                'gamma': uniform(0, 0.5)
            }
            
        elif model_name == 'lightgbm':
            return {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'num_leaves': randint(20, 40),
                'feature_fraction': uniform(0.6, 0.4),
                'bagging_fraction': uniform(0.6, 0.4)
            }
            
        elif model_name == 'catboost':
            return {
                'iterations': randint(50, 300),
                'depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'l2_leaf_reg': uniform(1, 5),
                'random_strength': uniform(0, 2)
            }
            
        elif model_name == 'random_forest':
            return {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10)
            }
            
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def get_default_metric(problem_type: str) -> str:
        """Get default metric for model evaluation based on problem type."""
        if problem_type == 'classification':
            return 'roc_auc'
        return 'neg_mean_squared_error'