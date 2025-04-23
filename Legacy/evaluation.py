from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import shap
from sklearn.inspection import permutation_importance
import mlflow

class ModelEvaluator:
    def __init__(self, 
                 problem_type: str = 'classification',
                 feature_names: Optional[List[str]] = None):
        self.problem_type = problem_type
        self.feature_names = feature_names
        
    def evaluate(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = model.predict(X)
        metrics = {}
        
        if self.problem_type == 'classification':
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['precision'] = precision_score(y, y_pred, average='weighted')
            metrics['recall'] = recall_score(y, y_pred, average='weighted')
            metrics['f1'] = f1_score(y, y_pred, average='weighted')
            
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X)
                if y_prob.shape[1] == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y, y_prob[:, 1])
                    
        else:  # regression
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y, y_pred)
            metrics['r2'] = r2_score(y, y_pred)
            
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            save_path: Optional[str] = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importance(self, model, X: np.ndarray,
                              importance_type: str = 'built_in',
                              save_path: Optional[str] = None):
        """Plot feature importance"""
        plt.figure(figsize=(12, 6))
        
        if importance_type == 'built_in' and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = (self.feature_names if self.feature_names 
                           else [f'feature_{i}' for i in range(X.shape[1])])
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Feature Importance (Built-in)')
            
        elif importance_type == 'permutation':
            result = permutation_importance(model, X, y, n_repeats=10)
            feature_names = (self.feature_names if self.feature_names 
                           else [f'feature_{i}' for i in range(X.shape[1])])
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': result.importances_mean
            }).sort_values('importance', ascending=False)
            
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Feature Importance (Permutation)')
            
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def explain_predictions(self, model, X: np.ndarray,
                          method: str = 'shap',
                          save_path: Optional[str] = None):
        """Generate model explanations"""
        if method == 'shap':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            plt.figure(figsize=(12, 6))
            if isinstance(shap_values, list):  # For multi-class
                shap.summary_plot(shap_values[1], X,
                                feature_names=self.feature_names,
                                show=False)
            else:
                shap.summary_plot(shap_values, X,
                                feature_names=self.feature_names,
                                show=False)
                
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
            return shap_values
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to MLflow"""
        mlflow.log_metrics(metrics)
    
    def log_artifacts(self, artifact_paths: Dict[str, str]):
        """Log artifacts to MLflow"""
        for name, path in artifact_paths.items():
            mlflow.log_artifact(path)
            
    def generate_report(self, model, X: np.ndarray, y: np.ndarray,
                       output_path: str = "model_report"):
        """Generate comprehensive model evaluation report"""
        # Create output directory
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Get metrics
        metrics = self.evaluate(model, X, y)
        
        # Generate plots
        if self.problem_type == 'classification':
            y_pred = model.predict(X)
            self.plot_confusion_matrix(
                y, y_pred,
                save_path=os.path.join(output_path, "confusion_matrix.png")
            )
            
        self.plot_feature_importance(
            model, X,
            save_path=os.path.join(output_path, "feature_importance.png")
        )
        
        self.explain_predictions(
            model, X,
            save_path=os.path.join(output_path, "shap_summary.png")
        )
        
        # Save metrics to file
        with open(os.path.join(output_path, "metrics.txt"), "w") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
                
        # Log everything to MLflow
        self.log_metrics(metrics)
        self.log_artifacts({
            'confusion_matrix': os.path.join(output_path, "confusion_matrix.png"),
            'feature_importance': os.path.join(output_path, "feature_importance.png"),
            'shap_summary': os.path.join(output_path, "shap_summary.png"),
            'metrics': os.path.join(output_path, "metrics.txt")
        })