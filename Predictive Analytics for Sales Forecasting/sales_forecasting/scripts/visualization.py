import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Data
df = pd.read_csv("C://Users//DELL//Desktop//Projects//Predictive Analytics for Sales Forecasting//sales_forecasting//data//processed//sales_data_cleaned.csv", index_col="date", parse_dates=True)
X = df.drop(columns=["sales"])
y = df["sales"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load Model
model = joblib.load("C://Users//DELL//Desktop//Projects//Predictive Analytics for Sales Forecasting//sales_forecasting//models//best_model.pkl")

# Identify Categorical Columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Encode Categorical Variables in Both Train & Test
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])  # Fit & Transform on Train
    X_test[col] = le.transform(X_test[col])  # Transform on Test (Use same mapping)
    label_encoders[col] = le

# Ensure No Missing Values
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Check Shapes
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Make Predictions
y_pred = model.predict(X_test)

# Ensure y_pred matches y_test in size
if y_pred.shape[0] != y_test.shape[0]:
    print("Warning: y_pred and y_test size mismatch!")
    y_pred = y_pred[:y_test.shape[0]]  # Trim if necessary

# Visualization - Residual Plot
plt.figure(figsize=(8,6))
sns.residplot(x=y_pred, y=y_test, lowess=True, line_kws={"color": "red"})
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Visualization - Predicted vs Actual Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values")
plt.show()

# Histogram of Residuals
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

# Save Plots
plt.savefig("sales_forecasting//visualizations//residual_plot.png")
plt.savefig("sales_forecasting//visualizations//predicted_vs_actual.png")
plt.savefig("sales_forecasting//visualizations//residual_histogram.png")
