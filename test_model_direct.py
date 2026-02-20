"""Direct model test to check predictions"""
import joblib
import pandas as pd
import numpy as np

# Load model
model_path = "models/crime_grid_xgb_model.pkl"
model = joblib.load(model_path)

# Features that would be sent for Mission District at 22:00 Friday
features = {
    'index_right': 1,  # Grid ID from the API response
    'time_bin': 5,      # 20:00-23:59
    'day_of_week': 4,   # Friday (0=Monday)
    'month': 2,
    'year': 2026,
    'is_weekend': 0,
    'lag_1': 0.0,
    'lag_6': 0.0,
    'lag_42': 0.0,
    'rolling_6_mean': 0.0,
    'rolling_42_mean': 0.0
}

# Create DataFrame
df = pd.DataFrame([features])

print("Features being sent to model:")
print(df)
print("\nFeature dtypes:")
print(df.dtypes)
print("\nFeature values:")
print(df.values)

# Make prediction
proba = model.predict_proba(df)
pred = model.predict(df)

print("\n" + "="*60)
print("Model Predictions:")
print("="*60)
print(f"Probability [safe, unsafe]: {proba[0]}")
print(f"Unsafe probability: {proba[0][1]:.4f}")
print(f"Binary prediction: {pred[0]}")

# Test with historical values
print("\n" + "="*60)
print("Testing with historical crime values:")
print("="*60)

features_with_history = features.copy()
features_with_history['lag_1'] = 2.0
features_with_history['lag_6'] = 3.0
features_with_history['lag_42'] = 2.5
features_with_history['rolling_6_mean'] = 2.2
features_with_history['rolling_42_mean'] = 2.8

df2 = pd.DataFrame([features_with_history])
proba2 = model.predict_proba(df2)
pred2 = model.predict(df2)

print("Features with history:")
print(df2)
print(f"\nProbability [safe, unsafe]: {proba2[0]}")
print(f"Unsafe probability: {proba2[0][1]:.4f}")
print(f"Binary prediction: {pred2[0]}")

# Test with year 2003 (original training year)
print("\n" + "="*60)
print("Testing with year 2003 (training data year):")
print("="*60)

features_2003 = features.copy()
features_2003['year'] = 2003

df3 = pd.DataFrame([features_2003])
proba3 = model.predict_proba(df3)
pred3 = model.predict(df3)

print("Features with 2003 year:")
print(df3)
print(f"\nProbability [safe, unsafe]: {proba3[0]}")
print(f"Unsafe probability: {proba3[0][1]:.4f}")
print(f"Binary prediction: {pred3[0]}")
