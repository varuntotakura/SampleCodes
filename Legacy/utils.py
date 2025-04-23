from typing import Dict, Any
import yaml
from dataclasses import dataclass
from pathlib import Path
import mlflow

RANDOM_STATE = 42

@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any]
    use_gpu: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str, model_name: str):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(
            name=model_name,
            params=config['models'][model_name],
            use_gpu=config.get('use_gpu', False)
        )

def setup_mlflow(experiment_name: str, tracking_uri: str = None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
def save_model(model, model_path: str):
    """Save model with proper format based on type"""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(model, model_path)
    
def load_model(model_path: str):
    """Load saved model"""
    return mlflow.sklearn.load_model(model_path)