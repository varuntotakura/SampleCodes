from typing import Dict, Any, Optional
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import optuna
from scipy.stats import uniform, randint
import mlflow
from utils import RANDOM_STATE

class ModelFactory:
    @staticmethod
    def create_model(name: str, params: Dict[str, Any], problem_type: str = 'classification', use_gpu: bool = False):
        if name == 'xgboost':
            if problem_type == 'classification':
                model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE)
            else:
                model = xgb.XGBRegressor(**params, random_state=RANDOM_STATE)
                
            if use_gpu:
                model.set_params(tree_method='gpu_hist')
                
        elif name == 'lightgbm':
            if problem_type == 'classification':
                model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE)
            else:
                model = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE)
                
            if use_gpu:
                model.set_params(device='gpu')
                
        elif name == 'catboost':
            if problem_type == 'classification':
                model = cb.CatBoostClassifier(**params, random_state=RANDOM_STATE)
            else:
                model = cb.CatBoostRegressor(**params, random_state=RANDOM_STATE)
                
            if use_gpu:
                model.set_params(task_type='GPU')
                
        elif name == 'random_forest':
            if problem_type == 'classification':
                model = RandomForestClassifier(**params, random_state=RANDOM_STATE)
            else:
                model = RandomForestRegressor(**params, random_state=RANDOM_STATE)
        else:
            raise ValueError(f"Unknown model: {name}")
            
        return model

class ModelOptimizer:
    def __init__(self, model_factory: ModelFactory, optimization_method: str = 'random'):
        self.model_factory = model_factory
        self.optimization_method = optimization_method
        
    def _get_param_space(self, model_name: str) -> Dict[str, Any]:
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
                'num_leaves': randint(20, 100),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'min_child_samples': randint(1, 50)
            }
        elif model_name == 'catboost':
            return {
                'iterations': randint(50, 300),
                'depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'l2_leaf_reg': uniform(1, 10),
                'subsample': uniform(0.6, 0.4)
            }
        elif model_name == 'random_forest':
            return {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 15),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10)
            }
    
    def _get_optuna_space(self, trial, model_name: str) -> Dict[str, Any]:
        if model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 0.5)
            }
        # Add similar spaces for other models...
    
    def optimize(self, 
                model_name: str,
                params: Dict[str, Any],
                X_train: np.ndarray,
                y_train: np.ndarray,
                problem_type: str = 'classification',
                use_gpu: bool = False,
                cv: int = 5,
                n_iter: int = 20,
                scoring: Optional[str] = None) -> Dict[str, Any]:
        
        if scoring is None:
            scoring = 'accuracy' if problem_type == 'classification' else 'neg_mean_squared_error'
            
        if self.optimization_method == 'random':
            model = self.model_factory.create_model(model_name, params, problem_type, use_gpu)
            param_space = self._get_param_space(model_name)
            
            search = RandomizedSearchCV(
                model, param_distributions=param_space,
                n_iter=n_iter, cv=cv, scoring=scoring,
                random_state=RANDOM_STATE
            )
            search.fit(X_train, y_train)
            return search.best_params_
            
        elif self.optimization_method == 'grid':
            model = self.model_factory.create_model(model_name, params, problem_type, use_gpu)
            param_grid = {k: [v] if not isinstance(v, (list, np.ndarray)) else v 
                         for k, v in params.items()}
            
            search = GridSearchCV(
                model, param_grid=param_grid,
                cv=cv, scoring=scoring
            )
            search.fit(X_train, y_train)
            return search.best_params_
            
        elif self.optimization_method == 'optuna':
            def objective(trial):
                trial_params = self._get_optuna_space(trial, model_name)
                model = self.model_factory.create_model(
                    model_name, {**params, **trial_params},
                    problem_type, use_gpu
                )
                score = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring=scoring
                ).mean()
                return -score if 'neg_' in scoring else score
                
            study = optuna.create_study(direction='minimize' if 'neg_' in scoring else 'maximize')
            study.optimize(objective, n_trials=n_iter)
            return study.best_params
            
        elif self.optimization_method == 'hyperopt':
            space = {k: hp.uniform(k, v.args[0], sum(v.args)) 
                    if isinstance(v, uniform)
                    else hp.randint(k, v.args[0], v.args[1])
                    for k, v in self._get_param_space(model_name).items()}
            
            def objective(space):
                model = self.model_factory.create_model(
                    model_name, {**params, **space},
                    problem_type, use_gpu
                )
                score = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring=scoring
                ).mean()
                return {'loss': -score if 'neg_' in scoring else -score,
                        'status': STATUS_OK}
                
            trials = Trials()
            best = fmin(objective, space, algo=tpe.suggest,
                       max_evals=n_iter, trials=trials)
            return best