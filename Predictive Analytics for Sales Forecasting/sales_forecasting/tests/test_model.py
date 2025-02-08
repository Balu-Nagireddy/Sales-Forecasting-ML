import joblib
import numpy as np
import pytest
from sales_forecasting.scripts.model import train_model

def test_train_model():
    X_train = np.random.rand(200, 10)
    y_train = np.random.rand(200)
    
    model = train_model(X_train, y_train)
    
    assert model is not None, "Model training failed"
    assert hasattr(model, "predict"), "Model does not have a predict method"
    
    print("✅ test_train_model PASSED!")

def test_model_prediction():
    model = joblib.load("sales_forecasting/models/best_model.pkl")
    sample_input = np.random.rand(5, 10)
    
    prediction = model.predict(sample_input)
    
    assert prediction.shape == (5,), "Prediction output shape incorrect"
    assert isinstance(prediction[0], (float, np.float64)), "Prediction should be numeric"
    
    print("✅ test_model_prediction PASSED!")

@pytest.mark.parametrize("invalid_input", [
    np.random.rand(1, 8),  # Incorrect feature size
    np.random.rand(0, 10), # Empty input
])
def test_model_invalid_inputs(invalid_input):
    model = joblib.load("sales_forecasting/models/best_model.pkl")
    
    with pytest.raises(ValueError):
        model.predict(invalid_input)

    print(f"✅ test_model_invalid_inputs PASSED for input shape {invalid_input.shape}!")
