import numpy as np
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, roc_curve, accuracy_score

def calculate_mse(y_true, y_pred):
    """Calculate Mean Squared Error (MSE)."""
    return mean_squared_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_f1_score(y_true, y_pred):
    """Calculate F1 Score."""
    return f1_score(y_true, y_pred, average='weighted')

def calculate_roc_auc(y_true, y_scores):
    """Calculate Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""
    return roc_auc_score(y_true, y_scores)

def calculate_roc_curve(y_true, y_scores):
    """Calculate ROC Curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return fpr, tpr, thresholds

def calculate_accuracy(y_true, y_pred):
    """Calculate Accuracy."""
    return accuracy_score(y_true, y_pred)