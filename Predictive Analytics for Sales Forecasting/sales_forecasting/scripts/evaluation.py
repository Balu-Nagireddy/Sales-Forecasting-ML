import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, and R²."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R²": r2}

def evaluate_model(model_path, X_test, y_test, output_csv="models/model_evaluation.csv"):
    """Evaluates the trained model and saves results."""
    # Load the model
    model = joblib.load(model_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Save results
    results = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    results.to_csv(output_csv, index=False)
    
    print("Model evaluation completed. Results saved to", output_csv)
    return results

# Example usage (to be removed or commented out in production)
# model_path = "models/best_model.pkl"
# X_test, y_test = load_test_data()  # Replace with actual test data
# evaluate_model(model_path, X_test, y_test)
