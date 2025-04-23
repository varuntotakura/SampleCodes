from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor

# Linear Regression Model
def linear_regression_model():
    """
    Returns a Linear Regression model.
    """
    model = LinearRegression()
    return model

# Logistic Regression Model
def logistic_regression_model(max_iter=1000):
    """
    Returns a Logistic Regression model.

    Parameters:
    - max_iter (int): Maximum number of iterations. Default is 1000.

    Example:
    logistic_regression_model(max_iter=200)
    """
    model = LogisticRegression(max_iter=max_iter)
    return model

# Random Forest Regressor with Hyperparameter Tuning
def random_forest_regressor_with_hyperparams(param_grid):
    """
    Returns a Random Forest Regressor with hyperparameter tuning.

    Parameters:
    - param_grid (dict): Dictionary of hyperparameters for GridSearchCV.

    Example:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    random_forest_regressor_with_hyperparams(param_grid)
    """
    model = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    return model

# Random Forest Classifier with Hyperparameter Tuning
def random_forest_classifier_with_hyperparams(param_grid):
    """
    Returns a Random Forest Classifier with hyperparameter tuning.

    Parameters:
    - param_grid (dict): Dictionary of hyperparameters for GridSearchCV.

    Example:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    random_forest_classifier_with_hyperparams(param_grid)
    """
    model = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
    return model

# Support Vector Regressor with Hyperparameter Tuning
def svr_with_hyperparams(param_grid):
    """
    Returns a Support Vector Regressor with hyperparameter tuning.

    Parameters:
    - param_grid (dict): Dictionary of hyperparameters for GridSearchCV.

    Example:
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svr_with_hyperparams(param_grid)
    """
    model = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
    return model

# Support Vector Classifier with Hyperparameter Tuning
def svc_with_hyperparams(param_grid):
    """
    Returns a Support Vector Classifier with hyperparameter tuning.

    Parameters:
    - param_grid (dict): Dictionary of hyperparameters for GridSearchCV.

    Example:
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svc_with_hyperparams(param_grid)
    """
    model = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    return model

# AdaBoost Regressor with Hyperparameter Tuning
def adaboost_regressor_with_hyperparams(param_grid):
    """
    Returns an AdaBoost Regressor with hyperparameter tuning.

    Parameters:
    - param_grid (dict): Dictionary of hyperparameters for GridSearchCV.

    Example:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }
    adaboost_regressor_with_hyperparams(param_grid)
    """
    model = GridSearchCV(AdaBoostRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    return model

# AdaBoost Classifier with Hyperparameter Tuning
def adaboost_classifier_with_hyperparams(param_grid):
    """
    Returns an AdaBoost Classifier with hyperparameter tuning.

    Parameters:
    - param_grid (dict): Dictionary of hyperparameters for GridSearchCV.

    Example:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }
    adaboost_classifier_with_hyperparams(param_grid)
    """
    model = GridSearchCV(AdaBoostClassifier(), param_grid, cv=5, scoring='accuracy')
    return model

# Bagging Regressor with Hyperparameter Tuning
def bagging_regressor_with_hyperparams(param_grid):
    """
    Returns a Bagging Regressor with hyperparameter tuning.

    Parameters:
    - param_grid (dict): Dictionary of hyperparameters for GridSearchCV.

    Example:
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0]
    }
    bagging_regressor_with_hyperparams(param_grid)
    """
    model = GridSearchCV(BaggingRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    return model

# Bagging Classifier with Hyperparameter Tuning
def bagging_classifier_with_hyperparams(param_grid):
    """
    Returns a Bagging Classifier with hyperparameter tuning.

    Parameters:
    - param_grid (dict): Dictionary of hyperparameters for GridSearchCV.

    Example:
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0]
    }
    bagging_classifier_with_hyperparams(param_grid)
    """
    model = GridSearchCV(BaggingClassifier(), param_grid, cv=5, scoring='accuracy')
    return model

# XGBoost Regressor with Hyperparameter Tuning
def xgboost_regressor_with_hyperparams(param_grid):
    """
    Returns an XGBoost Regressor with hyperparameter tuning.

    Parameters:
    - param_grid (dict): Dictionary of hyperparameters for GridSearchCV.

    Example:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    xgboost_regressor_with_hyperparams(param_grid)
    """
    model = GridSearchCV(XGBRegressor(objective='reg:squarederror'), param_grid, cv=5, scoring='neg_mean_squared_error')
    return model

# XGBoost Classifier with Hyperparameter Tuning
def xgboost_classifier_with_hyperparams(param_grid):
    """
    Returns an XGBoost Classifier with hyperparameter tuning.

    Parameters:
    - param_grid (dict): Dictionary of hyperparameters for GridSearchCV.

    Example:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    xgboost_classifier_with_hyperparams(param_grid)
    """
    model = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid, cv=5, scoring='accuracy')
    return model