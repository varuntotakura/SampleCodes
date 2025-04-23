import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    explained_variance_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
import os

class ModelEvaluator:
    """
    A comprehensive class for model evaluation and statistical analysis:
    - RMSE and Error Metrics
    - Correlation Analysis
    - Homoscedasticity Tests
    - Autocorrelation Analysis
    - Variance Analysis
    - Percentile Distribution Analysis
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, config=None):
        """
        Initialize ModelEvaluator with actual and predicted values.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            config (dict, optional): Configuration dictionary
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.config = config or {}
        # Set default configuration values
        self.mode_config = self.config.get('mode', {})
        self.use_inf_as_null = self.mode_config.get('use_inf_as_null', False)
        self.residuals = y_true - y_pred
        
        # Set up output directories from config
        output_config = self.config.get('output', {})
        self.plots_dir = output_config.get('plots_path', 'plots')
        self.results_dir = output_config.get('results_path', 'results')
        
        # Create directories if they don't exist
        for directory in [self.plots_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensuring directory exists: {os.path.abspath(directory)}")
            
    def calculate_error_metrics(self) -> Dict[str, float]:
        """
        Calculate various error metrics.
        
        Returns:
            Dict[str, float]: Dictionary containing error metrics
        """
        mse = mean_squared_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_true, self.y_pred)
        r2 = r2_score(self.y_true, self.y_pred)
        
        # Calculate EMSE (Estimated Mean Square Error)
        n = len(self.y_true)
        p = 1  # number of parameters, adjust if needed
        emse = (n * mse) / (n - p)
        
        # Calculate adjusted R-squared
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'emse': emse,
            'adjusted_r2': adj_r2
        }
    
    def analyze_correlation(self, X: np.ndarray = None) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Analyze correlations between features and predictions.
        
        Args:
            X (np.ndarray, optional): Feature matrix for correlation analysis
            
        Returns:
            Dict: Correlation analysis results
        """
        results = {}
        
        # Correlation between true and predicted values
        results['prediction_correlation'] = np.corrcoef(self.y_true, self.y_pred)[0, 1]
        
        # Feature correlations if X is provided
        if X is not None:
            feature_correlations = pd.DataFrame(X).corrwith(pd.Series(self.residuals))
            results['feature_residual_correlations'] = feature_correlations
            
            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            correlation_matrix = pd.DataFrame(X).corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'feature_correlation_heatmap.png'))
            plt.close()
        
        return results
    
    def test_homoscedasticity(self) -> Dict[str, Union[float, str]]:
        """
        Test for homoscedasticity using Breusch-Pagan test.
        
        Returns:
            Dict: Test results and interpretation
        """
        # Prepare data for Breusch-Pagan test
        X = sm.add_constant(self.y_pred)
        
        # Perform Breusch-Pagan test
        bp_test = het_breuschpagan(self.residuals, X)
        
        # Interpret results
        interpretation = "Homoscedastic" if bp_test[1] > 0.05 else "Heteroscedastic"
        
        # Plot residuals vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_pred, self.residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        plt.savefig(os.path.join(self.plots_dir, 'residuals_vs_predicted.png'))
        plt.close()
        
        return {
            'bp_statistic': bp_test[0],
            'p_value': bp_test[1],
            'interpretation': interpretation
        }
    
    def analyze_autocorrelation(self) -> Dict[str, Union[float, str]]:
        """
        Analyze autocorrelation in residuals using Durbin-Watson test.
        
        Returns:
            Dict: Autocorrelation analysis results
        """
        # Calculate Durbin-Watson statistic
        dw_statistic = durbin_watson(self.residuals)
        
        # Interpret Durbin-Watson statistic
        if dw_statistic < 1.5:
            interpretation = "Positive autocorrelation"
        elif dw_statistic > 2.5:
            interpretation = "Negative autocorrelation"
        else:
            interpretation = "No significant autocorrelation"
            
        # Plot autocorrelation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residuals over time
        ax1.plot(self.residuals)
        ax1.set_title('Residuals Over Time')
        ax1.set_xlabel('Observation')
        ax1.set_ylabel('Residual')
        
        # Lag plot
        pd.plotting.lag_plot(pd.Series(self.residuals), ax=ax2)
        ax2.set_title('Lag Plot of Residuals')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'autocorrelation_analysis.png'))
        plt.close()
        
        return {
            'durbin_watson': dw_statistic,
            'interpretation': interpretation
        }
    
    def analyze_variance(self) -> Dict[str, float]:
        """
        Analyze variance in predictions and residuals.
        
        Returns:
            Dict: Variance analysis results
        """
        return {
            'prediction_variance': np.var(self.y_pred),
            'residual_variance': np.var(self.residuals),
            'total_variance': np.var(self.y_true),
            'explained_variance_ratio': explained_variance_score(self.y_true, self.y_pred)
        }
    
    def analyze_percentile_distribution(self, 
                                     percentiles: List[float] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze distribution of predictions and residuals using percentiles.
        
        Args:
            percentiles (List[float], optional): List of percentiles to calculate
            
        Returns:
            Dict: Percentile analysis results
        """
        if percentiles is None:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            
        results = {
            'predictions': {},
            'residuals': {},
            'true_values': {}
        }
        
        # Calculate percentiles
        for p in percentiles:
            results['predictions'][f'p{p}'] = np.percentile(self.y_pred, p)
            results['residuals'][f'p{p}'] = np.percentile(self.residuals, p)
            results['true_values'][f'p{p}'] = np.percentile(self.y_true, p)
            
        # Plot distribution comparisons
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # True vs Predicted distributions
        sns.kdeplot(data=self.y_true, ax=ax1, label='True Values')
        sns.kdeplot(data=self.y_pred, ax=ax1, label='Predictions')
        ax1.set_title('Distribution: True vs Predicted Values')
        ax1.legend()
        
        # Residuals distribution
        sns.histplot(data=self.residuals, kde=True, ax=ax2)
        ax2.set_title('Distribution of Residuals')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'distribution_analysis.png'))
        plt.close()
        
        return results
    
    def plot_residual_analysis(self):
        """
        Create comprehensive residual analysis plots.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals vs Predicted
        ax1.scatter(self.y_pred, self.residuals)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        
        # Residual Distribution
        sns.histplot(data=self.residuals, kde=True, ax=ax2)
        ax2.set_title('Residual Distribution')
        
        # Q-Q Plot
        stats.probplot(self.residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot')
        
        # True vs Predicted
        ax4.scatter(self.y_true, self.y_pred)
        ax4.plot([self.y_true.min(), self.y_true.max()],
                [self.y_true.min(), self.y_true.max()],
                'r--', lw=2)
        ax4.set_xlabel('True Values')
        ax4.set_ylabel('Predicted Values')
        ax4.set_title('True vs Predicted Values')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'residual_analysis.png'))
        plt.close()
    
    def get_complete_evaluation(self, X: np.ndarray = None) -> Dict:
        """
        Get complete evaluation results including all metrics and analyses.
        
        Args:
            X (np.ndarray, optional): Feature matrix for correlation analysis
            
        Returns:
            Dict: Complete evaluation results
        """
        return {
            'error_metrics': self.calculate_error_metrics(),
            'correlation': self.analyze_correlation(X),
            'homoscedasticity': self.test_homoscedasticity(),
            'autocorrelation': self.analyze_autocorrelation(),
            'variance': self.analyze_variance(),
            'percentile_distribution': self.analyze_percentile_distribution()
        }
    
    @staticmethod
    def evaluate_saved_model(model_path: str, X: np.ndarray, y_true: np.ndarray, X_full: np.ndarray = None) -> dict:
        """
        Load a saved model, predict on X, and evaluate using ModelEvaluator.
        Args:
            model_path (str): Path to the saved model file.
            X (np.ndarray): Features to predict on.
            y_true (np.ndarray): True target values.
            X_full (np.ndarray, optional): Full feature matrix for correlation analysis.
        Returns:
            dict: Complete evaluation results.
        """
        import joblib
        model = joblib.load(model_path)
        y_pred = model.predict(X)
        evaluator = ModelEvaluator(y_true, y_pred)
        return evaluator.get_complete_evaluation(X_full)