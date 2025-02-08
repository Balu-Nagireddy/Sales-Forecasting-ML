import pandas as pd
import numpy as np
import pytest
from sales_forecasting.scripts.data_processing import preprocess_data  

def test_preprocess_data():
    sample_data = pd.DataFrame({
        'product_id': ['P001', 'P002', 'P003'],
        'sales': [100, np.nan, 200],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03']
    })
    
    processed_data = preprocess_data(sample_data)
    
    assert 'sales' in processed_data.columns, "Sales column missing after processing"
    assert processed_data.isnull().sum().sum() == 0, "Missing values not handled correctly"
    assert processed_data['sales'].dtype in [np.float64, np.int64], "Sales column should be numeric"
    
    print("✅ test_preprocess_data PASSED!")

@pytest.mark.parametrize("input_value, expected_output", [
    ("P0395", "UNKNOWN"),
    (np.nan, "UNKNOWN"),
    ("P0123", "P0123"),
])
def test_product_id_handling(input_value, expected_output):
    sample_data = pd.DataFrame({
        'product_id': [input_value],
        'sales': [150],
        'date': ['2024-02-01']
    })
    
    processed_data = preprocess_data(sample_data)
    assert processed_data['product_id'].iloc[0] == expected_output, "Product ID transformation failed"
    
    print(f"✅ test_product_id_handling PASSED for input {input_value}!")
