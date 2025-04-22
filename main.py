import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Any

from preprocessing import AdvancedPreprocessor
from models import ModelFactory, ModelOptimizer
from evaluation import ModelEvaluator
from utils import ModelConfig, setup_mlflow, save_model, RANDOM_STATE

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_synthetic_data(n_samples: int = 1000, n_features: int = 208):
    """Create synthetic dataset with mixed types"""
    from sklearn.datasets import make_classification
    
    # Generate base numeric features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features-8,  # Reserve space for categorical and boolean
        n_classes=2,
        random_state=RANDOM_STATE
    )
    
    # Convert to DataFrame
    X = pd.DataFrame(X, columns=[f'numeric_{i}' for i in range(X.shape[1])])
    
    # Add categorical features
    for i in range(4):
        X[f'category_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
        
    # Add boolean features
    for i in range(4):
        X[f'boolean_{i}'] = np.random.choice([True, False], size=n_samples)
        
    # Add some missing values
    mask = np.random.random(X.shape) < 0.05
    X[mask] = np.nan
    
    return X, y

def main(config_path: str):
    # Load configuration
    config = load_config(config_path)
    
    # Setup MLflow
    setup_mlflow(
        experiment_name=config['experiment_name'],
        tracking_uri=config.get('mlflow_tracking_uri')
    )
    
    # Generate or load data
    if config['data']['synthetic']:
        X, y = create_synthetic_data(
            n_samples=config['data']['n_samples'],
            n_features=config['data']['n_features']
        )
    else:
        # Load your actual dataset here
        data = pd.read_csv(config['data']['path'])
        X = data.drop(columns=[config['data']['target_column']])
        y = data[config['data']['target_column']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=RANDOM_STATE
    )
    
    # Initialize preprocessor
    preprocessor = AdvancedPreprocessor(
        numeric_features=config['preprocessing']['numeric_features'],
        categorical_features=config['preprocessing']['categorical_features'],
        datetime_features=config['preprocessing'].get('datetime_features'),
        scaler_type=config['preprocessing']['scaler'],
        imputer_type=config['preprocessing']['imputer'],
        categorical_encoder=config['preprocessing']['categorical_encoder'],
        feature_selection=config['preprocessing'].get('feature_selection'),
        n_features_to_select=config['preprocessing'].get('n_features_to_select'),
        pca_components=config['preprocessing'].get('pca_components'),
        handle_imbalance=config['preprocessing'].get('handle_imbalance'),
        sampling_strategy=config['preprocessing'].get('sampling_strategy', 'auto')
    )
    
    # Preprocess data
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Initialize model factory
    model_factory = ModelFactory()
    
    # Train and evaluate each selected model
    results = {}
    for model_name in config['models_to_use']:
        print(f"\nTraining {model_name}...")
        
        # Get model config
        model_config = ModelConfig(
            name=model_name,
            params=config['models'][model_name],
            use_gpu=config.get('use_gpu', False)
        )
        
        # Initialize optimizer
        optimizer = ModelOptimizer(
            model_factory,
            optimization_method=config['optimization']['method']
        )
        
        # Optimize hyperparameters
        best_params = optimizer.optimize(
            model_name=model_name,
            params=model_config.params,
            X_train=X_train_processed,
            y_train=y_train,
            problem_type=config['problem_type'],
            use_gpu=model_config.use_gpu,
            cv=config['optimization']['cv_folds'],
            n_iter=config['optimization']['n_iter']
        )
        
        # Create model with best parameters
        model = model_factory.create_model(
            name=model_name,
            params=best_params,
            problem_type=config['problem_type'],
            use_gpu=model_config.use_gpu
        )
        
        # Train final model
        model.fit(X_train_processed, y_train)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            problem_type=config['problem_type'],
            feature_names=X.columns.tolist()
        )
        
        # Generate evaluation report
        evaluator.generate_report(
            model=model,
            X=X_test_processed,
            y=y_test,
            output_path=f"model_reports/{model_name}"
        )
        
        # Save model
        save_model(
            model=model,
            model_path=f"models/{model_name}/model"
        )
        
        # Store results
        results[model_name] = {
            'model': model,
            'best_params': best_params,
            'metrics': evaluator.evaluate(model, X_test_processed, y_test)
        }
    
    # Print final comparison
    print("\nModel Comparison:")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        for metric, value in result['metrics'].items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    args = parser.parse_args()
    main(args.config)

# pip install -r requirements.txt
# python main.py --config config.yaml