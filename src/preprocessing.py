"""
preprocessing.py
Cleans and preprocesses raw air quality data for model training.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_raw_data(filename: str = "aqi_raw.csv") -> pd.DataFrame:
    """Load raw data from the raw data directory."""
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and rows with missing values."""
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features from existing columns."""
    if "DateObserved" in df.columns:
        df["DateObserved"] = pd.to_datetime(df["DateObserved"])
        df["Month"] = df["DateObserved"].dt.month
        df["DayOfWeek"] = df["DateObserved"].dt.dayofweek
    return df


def scale_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Standardise numeric feature columns."""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df


def save_processed_data(df: pd.DataFrame, filename: str = "aqi_processed.csv") -> str:
    """Save processed data to the processed data directory."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    return output_path


if __name__ == "__main__":
    df = load_raw_data()
    df = clean_data(df)
    df = feature_engineering(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df = scale_features(df, numeric_cols)
    save_processed_data(df)
