"""
train_model.py
Trains a machine learning model to predict Houston air quality (AQI).
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")


def load_processed_data(filename: str = "aqi_processed.csv") -> pd.DataFrame:
    """Load the processed dataset."""
    filepath = os.path.join(PROCESSED_DATA_DIR, filename)
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def train(df: pd.DataFrame, target_col: str = "AQI", test_size: float = 0.2):
    """Split data, train a Random Forest model, and evaluate it."""
    # Keep only numeric columns and drop any non-feature columns
    numeric_df = df.select_dtypes(include="number")
    if target_col not in numeric_df.columns:
        raise ValueError(f"Target column '{target_col}' not found or not numeric.")
    feature_cols = [c for c in numeric_df.columns if c != target_col]
    X = numeric_df[feature_cols]
    y = numeric_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"R²:  {r2:.4f}")

    return model, mae, r2


if __name__ == "__main__":
    df = load_processed_data()
    model, mae, r2 = train(df)
