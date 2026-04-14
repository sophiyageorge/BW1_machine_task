"""
train.py

This module trains a machine learning model on the Iris dataset
and saves the trained model to disk using joblib.

Author: Your Name
Date: 2026
"""

from typing import Tuple
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data(test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Load the Iris dataset and split it into training and testing sets.

    Args:
        test_size (float): Proportion of dataset to include in test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target  # ensures balanced split
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> float:
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        float: Accuracy score
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


def save_model(model, file_path: str = "model.pkl") -> None:
    """
    Save the trained model to disk.

    Args:
        model: Trained model
        file_path (str): Path to save the model
    """
    joblib.dump(model, file_path)


def main() -> None:
    """
    Main execution function:
    - Loads data
    - Trains model
    - Evaluates model
    - Saves model
    """
    # Load dataset
    X_train, X_test, y_train, y_test = load_data()

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save model
    save_model(model)
    print("Model trained and saved successfully!")


if __name__ == "__main__":
    main()