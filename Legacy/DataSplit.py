from sklearn.model_selection import train_test_split

def split_data(data, labels, train_ratio=0.7):
    """
    Splits data into training and testing sets based on the given ratio.

    Parameters:
        data (array-like): The input features.
        labels (array-like): The target labels.
        train_ratio (float): The ratio of training data (default is 0.7).

    Returns:
        X_train, X_test, y_train, y_test: Split data and labels.
    """
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.")
    
    test_ratio = 1 - train_ratio
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_ratio, random_state=42)
    return X_train, X_test, y_train, y_test