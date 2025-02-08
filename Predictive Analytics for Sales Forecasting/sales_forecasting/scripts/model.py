import os
import pickle
import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Get the absolute path to config.yaml
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "..", "config", "config.yaml")

# Ensure the path is correct
logging.info(f"Looking for config at: {config_path}")

try:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    logging.error("Config file not found. Please check the path.")
    raise
except yaml.YAMLError as e:
    logging.error(f"Error reading YAML file: {e}")
    raise

PROCESSED_DATA_PATH = config.get("paths", {}).get("processed_data")
MODEL_SAVE_PATH = os.path.join("sales_forecasting", "models", "best_model.pkl")

if not PROCESSED_DATA_PATH:
    logging.error("Processed data path is missing in the config file.")
    raise ValueError("Processed data path is not defined in config.yaml")

def load_data():
    """Load processed sales data."""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col="date", parse_dates=True)
        logging.info("Data successfully loaded.")
        return df
    except FileNotFoundError:
        logging.error(f"Processed data file not found at {PROCESSED_DATA_PATH}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def train_model():
    """Train a forecasting model with hyperparameter tuning."""
    df = load_data()
    
    # Validate if 'sales' column exists
    if "sales" not in df.columns:
        logging.error("The dataset is missing the required 'sales' column.")
        raise ValueError("Dataset must contain a 'sales' column for training.")
    
    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Feature Engineering: Add time-based features
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Features & Target
    X = df.drop(columns=["sales"], errors='ignore')  
    y = df["sales"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Model Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42))
    ])

    # Hyperparameter Tuning
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5, 10],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best Model Selection
    best_model = grid_search.best_estimator_
    logging.info(f"Best Model: {best_model}")

    # Evaluate Performance
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    logging.info(f"Model Performance - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, "wb") as file:
        pickle.dump(best_model, file)
    logging.info(f"Best trained model saved at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
