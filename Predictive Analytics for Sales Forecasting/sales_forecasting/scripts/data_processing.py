import pandas as pd
import yaml
import os

# Get the absolute path to config.yaml
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "config.yaml")

# Ensure the path is correct
print(f"Looking for config at: {config_path}")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "sales", "sales.csv")
PROCESSED_DATA_PATH = config["paths"]["processed_data"]

def load_data():
    """Load raw sales data from CSV."""
    return pd.read_csv(RAW_DATA_PATH)

def preprocess_data(df):
    """Clean and preprocess data."""
    # Convert 'date' to datetime format
    df["date"] = pd.to_datetime(df["date"])
    
    # Set 'date' as index
    df.set_index("date", inplace=True)
    
    # Drop missing values
    df.dropna(inplace=True)
    
    # Extract time-based features
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["day_of_week"] = df.index.dayofweek
    df["week_of_year"] = df.index.isocalendar().week
    df["quarter"] = df.index.quarter

    return df

def save_processed_data(df):
    """Save processed data to CSV."""
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    save_processed_data(df)
