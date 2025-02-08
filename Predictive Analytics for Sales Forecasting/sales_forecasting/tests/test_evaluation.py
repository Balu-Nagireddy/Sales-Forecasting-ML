import numpy as np
import pytest
from sales_forecasting.scripts.evaluation import evaluate_model, calculate_metrics

def test_calculate_metrics():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 290])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert "MAE" in metrics, "MAE not computed"
    assert "MSE" in metrics, "MSE not computed"
    assert "R2 Score" in metrics, "R2 Score missing"
    
    assert 0 <= metrics["R2 Score"] <= 1, "R2 Score should be between 0 and 1"
    
    print("✅ test_calculate_metrics PASSED!")

@pytest.mark.parametrize("y_true, y_pred", [
    (np.array([10, 20, 30]), np.array([10, 20, 30])), # Perfect prediction
    (np.array([10, 20, 30]), np.array([0, 0, 0])),   # Worst-case prediction
])
def test_extreme_cases(y_true, y_pred):
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics["MAE"] >= 0, "MAE should be non-negative"
    assert metrics["MSE"] >= 0, "MSE should be non-negative"
    
    print("✅ test_extreme_cases PASSED!")
