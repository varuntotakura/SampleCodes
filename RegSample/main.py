import numpy as np
import pandas as pd
import yaml
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from eda import ExploratoryAnalysis
from data_split import DataSplitter
from data_prep import DataPreprocessor
from feat_eng import FeatureEngineer
from feat_sel import FeatureSelector
from create_model import ModelBuilder
from model_eval import ModelEvaluator

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration"""
    log_level = getattr(logging, config['output']['log_level'])
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def print_help_options(config: Dict[str, Any]):
    """Print all available config and CLI options with current/default values."""
    print("\n=== HELP: Configuration and CLI Options ===")
    print("\nYAML Configurable Parameters:")
    def print_dict(d, prefix=""): 
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                print_dict(v, prefix + "  ")
            else:
                print(f"{prefix}{k}: {v}")
    print_dict(config)
    print("\nCLI Arguments:")
    print("  --config           Path to configuration file")
    print("  --input-file       Path to input data file")
    print("  --model            Specific model to run (overrides config)")
    print("  --feature-selection Specific feature selection method (overrides config)")
    print("  --tune-method      Hyperparameter tuning method (random/grid)")
    print("  --show-help        Show this help message and exit")
    print("  --eval-model-path  Path to a saved model file for evaluation")
    print("  --eval-data-path   Path to a CSV file with data for evaluating a saved model")
    print("\nTo override any config param, edit config.yaml or use CLI flags above.")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Machine Learning Pipeline', add_help=False)
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--input-file', type=str, help='Path to input data file')
    parser.add_argument('--model', type=str, help='Specific model to run')
    parser.add_argument('--feature-selection', type=str, help='Specific feature selection method')
    parser.add_argument('--tune-method', type=str, choices=['random', 'grid'], help='Hyperparameter tuning method')
    parser.add_argument('--show-help', action='store_true', help='Show all config and CLI options')
    parser.add_argument('--eval-model-path', type=str, help='Path to a saved model file for evaluation')
    parser.add_argument('--eval-data-path', type=str, help='Path to a CSV file with data for evaluating a saved model')
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit')
    return parser.parse_args()

def load_or_generate_data(config: Dict[str, Any], input_file: str = None) -> pd.DataFrame:
    """Load data from file or generate synthetic data"""
    if (input_file or config['data']['input_file']):
        file_path = input_file or config['data']['input_file']
        return pd.read_csv(file_path)
    elif config['data']['synthetic_data']['enable']:
        return generate_sample_data(
            n_samples=config['data']['synthetic_data']['n_samples'],
            n_features=config['data']['synthetic_data']['n_features']
        )
    else:
        raise ValueError("No data source specified")

def setup_output_directories(config: Dict[str, Any]) -> None:
    """Create output directories if they don't exist"""
    for path in [config['output']['model_path'],
                config['output']['results_path'],
                config['output']['plots_path']]:
        Path(path).mkdir(parents=True, exist_ok=True)

def preprocess_data(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Preprocess the data according to configuration"""
    preprocessor = DataPreprocessor(data)
    
    # Handle mode configuration silently
    mode_config = config.get('mode', {})
    setattr(preprocessor, 'use_inf_as_null', mode_config.get('use_inf_as_null', False))
    
    # Handle infinities silently
    if hasattr(preprocessor, '_handle_infinities'):
        preprocessor._handle_infinities()
    
    # Continue with rest of preprocessing
    if config['preprocessing']['missing_values']['strategy'] == 'auto':
        data = preprocessor.impute_missing_values(strategy='auto')
    else:
        data = preprocessor.impute_missing_values(
            strategy=config['preprocessing']['missing_values']['custom_rules']
        )
    
    # Apply scaling
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if config['preprocessing']['scaling']['method'] == 'standard':
        data = preprocessor.standardize_data(columns=numeric_columns)
    elif config['preprocessing']['scaling']['method'] == 'minmax':
        data = preprocessor.normalize_data(
            columns=numeric_columns,
            method='minmax',
            feature_range=tuple(config['preprocessing']['scaling']['target_range'])
        )
    elif config['preprocessing']['scaling']['method'] == 'robust':
        data = preprocessor.normalize_data(
            columns=numeric_columns,
            method='robust'
        )
    
    return data

def run_feature_engineering(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Apply feature engineering methods based on configuration"""
    if not config['feature_engineering']['enable']:
        return data
        
    engineer = FeatureEngineer(data)
    
    # Apply configured feature engineering methods
    if config['feature_engineering']['methods']['derived_features']['enable']:
        operations = {
            op['name']: {
                'columns': op['columns'],
                'operation': op['operation']
            }
            for op in config['feature_engineering']['methods']['derived_features']['operations']
        }
        data = engineer.create_derived_features(operations)
    
    if config['feature_engineering']['methods']['interactions']['enable']:
        data = engineer.create_interaction_features(
            feature_pairs=config['feature_engineering']['methods']['interactions']['feature_pairs'],
            interaction_type=config['feature_engineering']['methods']['interactions']['interaction_type']
        )
    
    if config['feature_engineering']['methods']['pca']['enable']:
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        data = engineer.apply_pca(
            columns=numeric_columns,
            n_components=config['feature_engineering']['methods']['pca']['n_components']
        )
    
    if config['feature_engineering']['methods']['polynomial']['enable']:
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns[:2]  # Limit to avoid explosion
        data = engineer.create_polynomial_features(
            columns=numeric_columns,
            degree=config['feature_engineering']['methods']['polynomial']['degree'],
            interaction_only=config['feature_engineering']['methods']['polynomial']['interaction_only']
        )
    
    return data

def run_feature_selection(data, config, method_override=None):
    """Run feature selection step in the pipeline"""
    selector = FeatureSelector(data)
    target = config.get('target_column', 'target')
    
    # Get method, either from CLI args or config
    method = method_override or config.get('feature_selection', {}).get('method', 'none')
    
    # Configuration for feature selection
    fs_config = config.get('feature_selection', {})
    
    # Number of features to select, using a default if not specified
    n_features = fs_config.get('n_features', 10)
    
    print(f"Using feature selection method: {method}")
    
    try:
        if method.lower() == 'k_best' or method.lower() == 'k-best':
            data = selector.select_k_best(target, k=n_features)
        elif method.lower() == 'lasso':
            # Get alpha from config or use default
            alpha = fs_config.get('alpha', 0.1)
            data = selector.lasso_selection(target, alpha=alpha)
        elif method.lower() == 'rfe':
            # Get step and estimator parameters from config
            step = fs_config.get('step', 1)
            data = selector.recursive_feature_elimination(target, n_features_to_select=n_features, step=step)
        elif method.lower() == 'random_forest':
            # Get random forest specific parameters
            rf_params = {k: v for k, v in fs_config.items() 
                      if k not in ['method', 'n_features']}
            data = selector.random_forest_selection(target, n_features=n_features, **rf_params)
        elif method.lower() == 'vif':
            # Get threshold from config or use default
            threshold = fs_config.get('threshold', 5.0)
            data = selector.multicollinearity_analysis(target, threshold=threshold)
        elif method.lower() == 'knn':
            # Get neighbors from config or use default
            n_neighbors = fs_config.get('n_neighbors', 5)
            data = selector.knn_feature_selection(target, n_features=n_features, n_neighbors=n_neighbors)
        else:
            logging.warning(f"Unknown feature selection method: {method}, skipping")
    except Exception as e:
        logging.warning(f"Feature selection skipped or failed: {str(e)}")
    
    return data

def run_model_pipeline(data: pd.DataFrame, config: Dict[str, Any], 
                      model_name: str = None, tune_method: str = None) -> Dict[str, Any]:
    """Run the modeling pipeline based on configuration"""
    # Split data
    target = config['data']['target_column']
    
    splitter = DataSplitter(data)
    train_data, test_data = splitter.simple_random_split(
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]
    
    # Initialize model builder
    model_builder = ModelBuilder()
    
    # Get models to run
    models_to_run = [m for m in config['models'] 
                    if m['enable'] and (not model_name or m['name'] == model_name)]
    
    results = {}
    for model_config in models_to_run:
        model_type = model_config['name']
        
        # Apply hyperparameter tuning if specified
        tuning_method = tune_method or config['tuning']['method']
        if tuning_method == 'random':
            search_results = model_builder.random_search_cv(
                X_train, y_train,
                model_type=model_type,
                param_distributions=model_config['params'],
                n_iter=config['tuning']['n_iter'],
                cv=config['tuning']['cv_folds']
            )
            best_params = search_results['best_params']
        elif tuning_method == 'grid':
            search_results = model_builder.grid_search_cv(
                X_train, y_train,
                model_type=model_type,
                param_grid=model_config['params'],
                cv=config['tuning']['cv_folds']
            )
            best_params = search_results['best_params']
        
        # Train and evaluate model
        model_results = model_builder.train_evaluate_model(
            X_train, X_test, y_train, y_test,
            model_type=model_type,
            params=best_params
        )
        
        # Store results
        results[model_type] = {
            'model': model_builder.models[model_type],
            'train_results': model_results['train'],
            'test_results': model_results['test'],
            'best_params': best_params
        }
        
        # Run model evaluation if configured
        if any(config['evaluation']['analysis'].values()):
            evaluator = ModelEvaluator(y_test, model_results['test']['predictions'])
            eval_results = evaluator.get_complete_evaluation(X_test)
            results[model_type]['evaluation'] = eval_results
            
            # Generate plots if configured
            if config['evaluation']['plots']['residuals']:
                evaluator.plot_residual_analysis()
            if config['evaluation']['plots']['learning_curves']:
                model_builder.plot_learning_curves(
                    X_train, y_train,
                    model_type=model_type,
                    params=best_params
                )
    
    return results

def save_results(results: Dict[str, Any], config: Dict[str, Any], current_model: str = None, 
              current_feature_selection: str = None, current_tune_method: str = None) -> None:
    """Save model results and artifacts"""
    if not config['output']['save_model']:
        return
        
    output_dir = Path(config['output']['results_path'])
    model_dir = Path(config['output']['model_path'])
    
    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model results
    results_df = pd.DataFrame()
    for model_name, model_results in results.items():
        model_metrics = {
            'model': model_name,
            'train_r2': float(model_results['train_results']['r2']),
            'test_r2': float(model_results['test_results']['r2']),
            'train_mse': float(model_results['train_results']['mse']),
            'test_mse': float(model_results['test_results']['mse']),
            'train_mae': float(model_results['train_results']['mae']),
            'test_mae': float(model_results['test_results']['mae']),
            'train_explained_variance': float(model_results['train_results']['explained_variance']),
            'test_explained_variance': float(model_results['test_results']['explained_variance']),
        }
        
        # Add best parameters to results
        if 'best_params' in model_results:
            for param_name, param_value in model_results['best_params'].items():
                model_metrics[f'param_{param_name}'] = param_value
                
        results_df = pd.concat([results_df, pd.DataFrame([model_metrics])])
    
    # Save results CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_identifier = current_model if current_model else "all_models"
    feature_sel = current_feature_selection if current_feature_selection else "default"
    tune_method = current_tune_method if current_tune_method else "default"
    
    results_csv_path = output_dir / f"{model_identifier}_{feature_sel}_{tune_method}_{timestamp}_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    logging.info(f"Results saved to {results_csv_path}")
    
    # Save models to model directory
    for model_name, model_results in results.items():
        if 'model' in model_results:
            # Save model
            model_filename = f"{model_name}_{feature_sel}_{tune_method}_{timestamp}_model.pkl"
            model_path = model_dir / model_filename
            try:
                from create_model import ModelBuilder
                ModelBuilder.save_model(model_results['model'], model_path)
                logging.info(f"Model saved to {model_path}")
                
                # Save best configuration in YAML with native Python types
                best_config = {
                    'model': model_name,
                    'feature_selection': feature_sel,
                    'tuning_method': tune_method,
                    'timestamp': timestamp,
                    'best_params': model_results['best_params'] if 'best_params' in model_results else {},
                    'training_metrics': {
                        'r2': float(model_results['train_results']['r2']),
                        'mse': float(model_results['train_results']['mse']),
                        'mae': float(model_results['train_results']['mae']),
                        'explained_variance': float(model_results['train_results']['explained_variance'])
                    },
                    'test_metrics': {
                        'r2': float(model_results['test_results']['r2']),
                        'mse': float(model_results['test_results']['mse']),
                        'mae': float(model_results['test_results']['mae']),
                        'explained_variance': float(model_results['test_results']['explained_variance'])
                    }
                }
                config_path = model_dir / f"{model_name}_{feature_sel}_{tune_method}_{timestamp}_config.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(best_config, f, default_flow_style=False)
                logging.info(f"Best configuration saved to {config_path}")
                
            except Exception as e:
                logging.error(f"Error saving model {model_name}: {e}")
    
    # Save detailed evaluation results
    for model_name, model_results in results.items():
        if 'evaluation' in model_results:
            eval_path = output_dir / f'{model_name}_{feature_sel}_{tune_method}_{timestamp}_evaluation.yaml'
            
            # Convert numpy values to native Python types
            eval_results = {}
            for key, value in model_results['evaluation'].items():
                if hasattr(value, 'dtype'):  # Check if it's a numpy value
                    eval_results[key] = float(value)
                else:
                    eval_results[key] = value
                    
            with open(eval_path, 'w') as f:
                yaml.dump(eval_results, f)
            logging.info(f"Evaluation results saved to {eval_path}")

def generate_sample_data(n_samples=1000, n_features=208):
    """
    Generate sample regression data for demonstration with diverse data types:
    - Numerical (float) features
    - Integer features
    - Boolean features
    - Categorical features
    - Text features
    """
    np.random.seed(42)
    
    # Initialize empty DataFrame
    data = pd.DataFrame()
    
    # 1. Generate continuous numerical features (float) - 100 features
    for i in range(100):
        data[f'float_feature_{i}'] = np.random.randn(n_samples)
    
    # 2. Generate integer features - 40 features
    for i in range(40):
        data[f'int_feature_{i}'] = np.random.randint(0, 100, n_samples)
    
    # 3. Generate boolean features - 20 features
    for i in range(20):
        data[f'bool_feature_{i}'] = np.random.choice([True, False], n_samples)
    
    # 4. Generate categorical features - 30 features
    categories = ['A', 'B', 'C', 'D', 'E']
    for i in range(30):
        data[f'cat_feature_{i}'] = np.random.choice(categories, n_samples)
    
    # 5. Generate text features with random strings - 18 features
    text_options = ['high', 'medium', 'low', 'critical', 'normal']
    for i in range(18):
        data[f'text_feature_{i}'] = np.random.choice(text_options, n_samples)
    
    # Generate target variable with relationships to multiple feature types
    target = (
        2 * data['float_feature_0'] +
        0.5 * data['int_feature_0'] +
        3 * data['bool_feature_0'].astype(int) +
        pd.get_dummies(data['cat_feature_0'], prefix='cat')['cat_A'] +
        (data['text_feature_0'] == 'high').astype(int) +
        np.random.randn(n_samples) * 0.1
    )
    
    # Scale target to reasonable range
    data['target'] = (target - target.mean()) / target.std() * 10 + 50  # Center around 50
    
    return data

def generate_sample_data_with_missing(n_samples=1000, n_features=208):
    """
    Generate sample regression data with missing values for demonstration
    """
    data = generate_sample_data(n_samples, n_features)
    
    # Introduce missing values randomly
    for column in data.columns:
        # Random missing percentage between 5% and 40%
        missing_pct = np.random.uniform(0.05, 0.4)
        mask = np.random.choice(
            [True, False], 
            size=len(data), 
            p=[missing_pct, 1-missing_pct]
        )
        data.loc[mask, column] = np.nan
        
    return data

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate the config structure and required keys. Returns True if valid, else False."""
    required_sections = [
        'data', 'preprocessing', 'models', 'tuning', 'evaluation', 'output'
    ]
    for section in required_sections:
        if section not in config:
            logging.error(f"Missing required config section: {section}")
            return False
            
    # Check for required keys in data section
    for key in ['target_column', 'test_size', 'random_state']:
        if key not in config['data']:
            logging.error(f"Missing required data config key: {key}")
            return False
            
    # Check for at least one enabled model in the models list
    if not any(m.get('enable', False) for m in config['models']):
        logging.error("No models enabled in config['models'].")
        return False
        
    return True

def safe_load_or_generate_data(config, input_file):
    try:
        return load_or_generate_data(config, input_file)
    except Exception as e:
        logging.error(f"Failed to load or generate data: {e}")
        raise

def safe_run_preprocessing(data, config):
    try:
        return preprocess_data(data, config)
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise

def safe_run_feature_engineering(data, config):
    try:
        return run_feature_engineering(data, config)
    except Exception as e:
        logging.warning(f"Feature engineering skipped or failed: {e}")
        return data

def safe_run_feature_selection(data, config, method_override):
    try:
        return run_feature_selection(data, config, method_override)
    except Exception as e:
        logging.warning(f"Feature selection skipped or failed: {e}")
        return data

def safe_run_model_pipeline(data, config, model_name, tune_method):
    """Run the modeling pipeline with proper data type handling"""
    try:
        # Ensure all data is numeric
        numeric_data = data.copy()
        for col in numeric_data.columns:
            if not pd.api.types.is_numeric_dtype(numeric_data[col]):
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
        
        # Split data
        target = config['data']['target_column']
        splitter = DataSplitter(numeric_data)
        train_data, test_data = splitter.simple_random_split(
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        
        # Convert data to numpy arrays with explicit float32 type to avoid categorical data errors
        X_train = train_data.drop(columns=[target]).astype(np.float32).values
        y_train = train_data[target].astype(np.float32).values
        X_test = test_data.drop(columns=[target]).astype(np.float32).values
        y_test = test_data[target].astype(np.float32).values
        
        # Initialize model builder
        model_builder = ModelBuilder()
        
        # Get models to run
        models_to_run = [m for m in config['models'] 
                        if m['enable'] and (not model_name or m['name'] == model_name)]
        
        if not models_to_run:
            logging.warning(f"No enabled models found for: {model_name}")
            return {}
            
        results = {}
        for model_config in models_to_run:
            try:
                model_type = model_config['name']
                logging.info(f"Training model: {model_type}")
                
                # Apply fixed parameters from config for this model
                fixed_params = model_config.get('fixed_params', {})
                
                # For XGBoost, explicitly set tree_method and disable categorical support
                if model_type == 'xgboost':
                    fixed_params['tree_method'] = fixed_params.get('tree_method', 'hist')
                    fixed_params['enable_categorical'] = False
                
                # Apply hyperparameter tuning if specified
                tuning_method = tune_method or config['tuning']['method']
                if tuning_method == 'random':
                    # Merge fixed params with tuning params
                    param_distributions = model_config['params'].copy()
                    for param, value in fixed_params.items():
                        param_distributions[param] = [value]  # Use list for RandomizedSearchCV
                    
                    search_results = model_builder.random_search_cv(
                        X_train, y_train,
                        model_type=model_type,
                        param_distributions=param_distributions,
                        n_iter=config['tuning']['n_iter'],
                        cv=config['tuning']['cv_folds']
                    )
                    best_params = search_results.get('best_params', {})
                elif tuning_method == 'grid':
                    # Merge fixed params with tuning params
                    param_grid = model_config['params'].copy()
                    for param, value in fixed_params.items():
                        param_grid[param] = [value]  # Use list for GridSearchCV
                    
                    search_results = model_builder.grid_search_cv(
                        X_train, y_train,
                        model_type=model_type,
                        param_grid=param_grid,
                        cv=config['tuning']['cv_folds']
                    )
                    best_params = search_results.get('best_params', {})
                
                if not best_params:
                    logging.warning(f"No best parameters found for {model_type}")
                    continue
                
                # Add fixed parameters to best_params
                for param, value in fixed_params.items():
                    best_params[param] = value
                
                # Train and evaluate model
                model_results = model_builder.train_evaluate_model(
                    X_train, X_test, y_train, y_test,
                    model_type=model_type,
                    params=best_params
                )
                
                if not model_results:
                    logging.warning(f"No results obtained for {model_type}")
                    continue
                
                # Store results only if we have valid metrics
                if all(metric > -np.inf and metric < np.inf 
                      for metrics in [model_results['train'], model_results['test']]
                      for metric in metrics.values() if isinstance(metric, (int, float))):
                    results[model_type] = {
                        'model': model_builder.models[model_type],
                        'train_results': model_results['train'],
                        'test_results': model_results['test'],
                        'best_params': best_params
                    }
                    logging.info(f"Successfully trained and evaluated {model_type}")
                else:
                    logging.warning(f"Invalid metrics obtained for {model_type}")
            
            except Exception as e:
                logging.error(f"Error training model {model_type}: {str(e)}")
                continue
        
        return results
    except Exception as e:
        logging.error(f"Model pipeline failed: {e}")
        return {}

def safe_save_results(results, config, current_model=None, current_feature_selection=None, current_tune_method=None):
    try:
        save_results(results, config, current_model, current_feature_selection, current_tune_method)
    except Exception as e:
        logging.warning(f"Saving results failed: {e}")

def run_evaluation(model_path, data_path, config):
    """
    Run evaluation on a saved model
    
    Args:
        model_path (str): Path to the saved model file
        data_path (str): Path to the data file for evaluation
        config (dict): Configuration dictionary
    """
    print(f"\n=== Evaluating Saved Model: {model_path} ===")
    # Load the data
    data = pd.read_csv(data_path)
    target_col = config['data']['target_column']
    X_eval = data.drop(columns=[target_col])
    y_eval = data[target_col]
    
    # Load the model
    try:
        from create_model import ModelBuilder
        model = ModelBuilder.load_model(model_path)
        
        # Create evaluator
        y_pred = model.predict(X_eval)
        evaluator = ModelEvaluator(y_eval, y_pred, config)
        
        # Run evaluation
        eval_results = evaluator.get_complete_evaluation(X_eval)
        
        # Save evaluation results
        output_dir = Path(config['output']['results_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = os.path.basename(model_path).replace('_model.pkl', '')
        eval_path = output_dir / f'{model_name}_evaluation.yaml'
        with open(eval_path, 'w') as f:
            yaml.dump(eval_results, f)
        
        print(f"Evaluation results saved to {eval_path}")
        return eval_results
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

def check_and_create_directories():
    """Create all required directories if they don't exist"""
    required_dirs = ['models', 'plots', 'results']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {dir_path.absolute()}")

def run_pipeline_step(step_function, *args, **kwargs):
    """Helper function to run a pipeline step with error handling."""
    try:
        return step_function(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {step_function.__name__}: {e}")
        return None

def run_eda(data: pd.DataFrame, config: dict) -> dict:
    """Run exploratory data analysis step and return summary results in DataFrames"""
    analyzer = ExploratoryAnalysis(data)

    # Basic summary of data (could also return it)
    analyzer.basic_summary()

    univariate_results = []
    bivariate_results = []

    # Univariate analysis
    if config.get("perform_univariate", True):
        for column in data.columns:
            try:
                stats = analyzer.univariate_analysis(column)
                stats["column"] = column
                univariate_results.append(stats)
            except Exception as e:
                logging.error(f"Error in univariate_analysis for {column}: {str(e)}")

    # Bivariate analysis: numeric-numeric
    if config.get("perform_bivariate", True):
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                x_col, y_col = numeric_cols[i], numeric_cols[j]
                try:
                    bivar_stats = analyzer.bivariate_analysis(x_col, y_col)
                    bivar_stats["x_col"] = x_col
                    bivar_stats["y_col"] = y_col
                    bivariate_results.append(bivar_stats)
                except Exception as e:
                    logging.error(f"Error in bivariate_analysis for {x_col} and {y_col}: {str(e)}")

    # Convert results to DataFrames
    univariate_df = pd.DataFrame(univariate_results)
    print("Univariate Analysis Results:")
    print(univariate_df)

    bivariate_df = pd.DataFrame(bivariate_results)
    print("Bivariate Analysis Results:")
    print(bivariate_df)

    return data

def run_eda_plots(data: pd.DataFrame, config: dict) -> None:
    """Run exploratory data analysis step"""
    analyzer = ExploratoryAnalysis(data)
    analyzer.basic_summary()
    try:
        analyzer.plot_distributions()
    except Exception as e:
        logging.error(f"Error in plot_distributions: {str(e)}")
    return data

def main():
    """Main execution function"""
    # Create required directories first
    check_and_create_directories()
    
    # Parse arguments and load config
    args = parse_arguments()
    config = load_config(args.config)
    if args.show_help:
        print_help_options(config)
        return
    
    # Validate config
    if not validate_config(config):
        print("Config validation failed. Please check your config.yaml.")
        return
    
    # Update config with command line arguments
    if args.input_file:
        config['data']['input_file'] = args.input_file
    if args.tune_method:
        config['tuning']['method'] = args.tune_method
    
    # Normalize feature selection methods for consistency
    if args.feature_selection:
        # Convert hyphens to underscores for method names
        args.feature_selection = args.feature_selection.replace('-', '_')
        # Fix incorrect feature selection method names
        if args.feature_selection == 'knn':
            args.feature_selection = 'k_best'  # Default to k_best if knn is specified
        print(f"Using feature selection method: {args.feature_selection}")
        
    # Setup environment
    setup_logging(config)
    setup_output_directories(config)
    
    # Add logic for evaluating a saved model
    if getattr(args, 'eval_model_path', None) and getattr(args, 'eval_data_path', None):
        run_evaluation(args.eval_model_path, args.eval_data_path, config)
        return
    
    # Load or generate data
    print("\n=== Loading Data ===")
    data = run_pipeline_step(safe_load_or_generate_data, config, args.input_file)
    if data is None:
        return
    print(f"Data loaded: {data.shape} - Memory usage: {data.memory_usage().sum() / 1024**2:.2f} MB")

    # EDA usage (before preprocessing)
    print("\n=== Exploratory Data Analysis (EDA) ===")
    run_pipeline_step(run_eda, data, config)
    
    # Split the data immediately after EDA
    print("\n=== Splitting Data ===")
    target = config['data']['target_column']
    splitter = DataSplitter(data)
    train_data, test_data = splitter.simple_random_split(
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    
    # Preprocessing - apply to both train and test sets separately
    print("\n=== Preprocessing Data ===")
    train_data = run_pipeline_step(safe_run_preprocessing, train_data, config)
    if train_data is None:
        return
    test_data = run_pipeline_step(safe_run_preprocessing, test_data, config)
    if test_data is None:
        return
    print(f"After preprocessing - Train: {train_data.shape}, Test: {test_data.shape}")

    # Feature engineering - apply to both train and test sets separately
    print("\n=== Feature Engineering ===")
    train_data = run_pipeline_step(safe_run_feature_engineering, train_data, config)
    if train_data is None:
        return
    test_data = run_pipeline_step(safe_run_feature_engineering, test_data, config)
    if test_data is None:
        return
    print(f"After feature engineering - Train: {train_data.shape}, Test: {test_data.shape}")

    # Feature selection - fit on train data, apply to both train and test
    print("\n=== Feature Selection ===")
    selector = FeatureSelector(train_data)
    fs_method = args.feature_selection or config.get('feature_selection', {}).get('method', 'none')
    fs_config = config.get('feature_selection', {})
    n_features = fs_config.get('n_features', 10)
    
    print(f"Using feature selection method: {fs_method}")
    try:
        if fs_method.lower() in ('k_best', 'k-best'):
            train_data = selector.select_k_best(target, k=n_features)
            # Apply same feature selection to test data
            test_data = test_data[train_data.columns]
        elif fs_method.lower() == 'lasso':
            alpha = fs_config.get('alpha', 0.1)
            train_data = selector.lasso_selection(target, alpha=alpha)
            test_data = test_data[train_data.columns]
        elif fs_method.lower() == 'rfe':
            step = fs_config.get('step', 1)
            train_data = selector.recursive_feature_elimination(target, n_features_to_select=n_features, step=step)
            test_data = test_data[train_data.columns]
        elif fs_method.lower() == 'random_forest':
            rf_params = {k: v for k, v in fs_config.items() if k not in ['method', 'n_features']}
            train_data = selector.random_forest_selection(target, n_features=n_features, **rf_params)
            test_data = test_data[train_data.columns]
        elif fs_method.lower() == 'vif':
            threshold = fs_config.get('threshold', 5.0)
            train_data = selector.multicollinearity_analysis(target, threshold=threshold)
            test_data = test_data[train_data.columns]
        elif fs_method.lower() == 'knn':
            n_neighbors = fs_config.get('n_neighbors', 5)
            train_data = selector.knn_feature_selection(target, n_features=n_features, n_neighbors=n_neighbors)
            test_data = test_data[train_data.columns]
    except Exception as e:
        logging.warning(f"Feature selection skipped or failed: {str(e)}")
    
    print(f"After feature selection - Train: {train_data.shape}, Test: {test_data.shape}")

    # Model training
    print(f"\n=== Training Model: {args.model if args.model else 'all enabled models'} ===")
    
    # Extract features and target from train and test sets
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]
    
    # Initialize model builder
    model_builder = ModelBuilder()
    
    # Get models to run
    models_to_run = [m for m in config['models'] 
                    if m['enable'] and (not args.model or m['name'] == args.model)]
    
    results = {}
    for model_config in models_to_run:
        try:
            model_type = model_config['name']
            print(f"Training model: {model_type}")
            
            # Apply fixed parameters from config for this model
            fixed_params = model_config.get('fixed_params', {})
            
            # For XGBoost, explicitly set tree_method and disable categorical support
            if model_type == 'xgboost':
                fixed_params['tree_method'] = fixed_params.get('tree_method', 'hist')
                fixed_params['enable_categorical'] = False
            
            # Apply hyperparameter tuning if specified
            tuning_method = args.tune_method or config['tuning']['method']
            if tuning_method == 'random':
                # Merge fixed params with tuning params
                param_distributions = model_config['params'].copy()
                for param, value in fixed_params.items():
                    param_distributions[param] = [value]  # Use list for RandomizedSearchCV
                
                search_results = model_builder.random_search_cv(
                    X_train, y_train,
                    model_type=model_type,
                    param_distributions=param_distributions,
                    n_iter=config['tuning']['n_iter'],
                    cv=config['tuning']['cv_folds']
                )
                best_params = search_results.get('best_params', {})
            elif tuning_method == 'grid':
                # Merge fixed params with tuning params
                param_grid = model_config['params'].copy()
                for param, value in fixed_params.items():
                    param_grid[param] = [value]  # Use list for GridSearchCV
                
                search_results = model_builder.grid_search_cv(
                    X_train, y_train,
                    model_type=model_type,
                    param_grid=param_grid,
                    cv=config['tuning']['cv_folds']
                )
                best_params = search_results.get('best_params', {})
            
            if not best_params:
                print(f"No best parameters found for {model_type}")
                continue
            
            # Add fixed parameters to best_params
            for param, value in fixed_params.items():
                best_params[param] = value
            
            # Train and evaluate model
            model_results = model_builder.train_evaluate_model(
                X_train, X_test, y_train, y_test,
                model_type=model_type,
                params=best_params
            )
            
            # Store results
            results[model_type] = {
                'model': model_builder.models[model_type],
                'train_results': model_results['train'],
                'test_results': model_results['test'],
                'best_params': best_params
            }
            
            # Run model evaluation if configured
            if any(config['evaluation']['analysis'].values()):
                evaluator = ModelEvaluator(y_test, model_results['test']['predictions'], config)
                eval_results = evaluator.get_complete_evaluation(X_test, model_builder.models[model_type])
                results[model_type]['evaluation'] = eval_results
                
                # Generate plots if configured
                if config['evaluation']['plots']['residuals']:
                    evaluator.plot_residual_analysis()
                if config['evaluation']['plots']['learning_curves']:
                    model_builder.plot_learning_curves(
                        X_train, y_train,
                        model_type=model_type,
                        params=best_params
                    )
        except Exception as e:
            logging.error(f"Error training model {model_config['name']}: {e}")
            continue

    # Save results
    print("\n=== Saving Results ===")
    run_pipeline_step(safe_save_results, results, config, args.model, args.feature_selection, args.tune_method)
    print("\n=== Process Complete ===")

    # Print a summary of the results
    if results:
        print("\n=== Results Summary ===")
        for model_name, model_results in results.items():
            print(f"Model: {model_name}")
            print(f"  Training R²: {model_results['train_results']['r2']:.4f}")
            print(f"  Test R²: {model_results['test_results']['r2']:.4f}")
            print(f"  Training MSE: {model_results['train_results']['mse']:.4f}")
            print(f"  Test MSE: {model_results['test_results']['mse']:.4f}")
            if 'best_params' in model_results:
                print("  Best parameters:")
                for param, value in model_results['best_params'].items():
                    print(f"    {param}: {value}")

    print(f"\n=== Evaluating Training Model ===")
    eval_results = ModelEvaluator.evaluate_training_model(
        model_builder.models[model_type], X_test.values, y_test.values, X_full=X_test.values
    )
    print(yaml.dump(eval_results, sort_keys=False, allow_unicode=True))

if __name__ == "__main__":
    args = parse_arguments()
    main()