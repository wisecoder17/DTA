import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from dl_model import ScratchMLPRegressor
import xgboost as xgb


# Features 

FEATURES = [
    "distance_km",
    "hour",
    "day_of_week",
    "is_weekend",
    "Weather",
    "traffic_level"
]

TARGET = "Delivery_Time"

MODEL_DIR = "models"
DATA_PATH = "data/processed/amazon_processed.csv"

os.makedirs(MODEL_DIR, exist_ok=True)


# Load dataset

df = pd.read_csv(DATA_PATH)

# Ensure temporal ordering
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df.sort_values("Order_Date", inplace=True)


# Temporal Train / Val / Test Split 70/15/15

n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_df = df.iloc[:train_end]
val_df   = df.iloc[train_end:val_end]
test_df  = df.iloc[val_end:]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# Encode categorical feature (Weather)
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)


X_train_Weather = encoder.fit_transform(X_train[["Weather"]])
X_val_Weather   = encoder.transform(X_val[["Weather"]])
X_test_Weather  = encoder.transform(X_test[["Weather"]])

X_train_num = X_train.drop(columns=["Weather"]).values
X_val_num   = X_val.drop(columns=["Weather"]).values
X_test_num  = X_test.drop(columns=["Weather"]).values

X_train_final = np.hstack([X_train_num, X_train_Weather])
X_val_final   = np.hstack([X_val_num, X_val_Weather])
X_test_final  = np.hstack([X_test_num, X_test_Weather])

assert X_train_final.shape[1] == X_test_final.shape[1]


# Models
models = {}

models["LinearRegression"] = LinearRegression()
models["KNN"] = KNeighborsRegressor(n_neighbors=5)
models["RandomForest"] = RandomForestRegressor(
    n_estimators=200, random_state=42, n_jobs=-1
)
models["XGBoost"] = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

models["SimpleNN"] = ScratchMLPRegressor(
    input_size=X_train_final.shape[1],
    hidden_size=10,
    lr=0.001,
    epochs=500
)


# Training & Evaluation
results = {}

for name, model in models.items():
    model.fit(X_train_final, y_train)
    
    if hasattr(model, "model"):
        model.model.verbose = False

    y_train_pred = model.predict(X_train_final)
    y_test_pred  = model.predict(X_test_final)

    results[name] = {
        "model": model,
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "test_r2": r2_score(y_test, y_test_pred),
    }

    print(f"\n{name}")
    print(f"  Train MAE: {results[name]['train_mae']:.2f}")
    print(f"  Test  MAE: {results[name]['test_mae']:.2f}")
    print(f"  Test  RMSE: {results[name]['test_rmse']:.2f}")
    print(f"  Test  R2: {results[name]['test_r2']:.3f}")
    print("-" * 60)


# Save Results Summary

results_df = pd.DataFrame({
    model_name: {
        "MAE": metrics["test_mae"],
        "RMSE": metrics["test_rmse"],
        "R2": metrics["test_r2"]
    }
    for model_name, metrics in results.items()
}).T.round(3)

print("\n\tModel \tPerformance \tSummary")
print(results_df)
print("Model results saved successfully.")

results_df.to_csv("results/amazon_model_results.csv")


# Select Best Model (by Test MAE)

best_model_name = min(results, key=lambda k: results[k]["test_mae"])
best_model = results[best_model_name]["model"]

print("\n" + "=" * 70)
print(f"BEST MODEL: {best_model_name}")
print(f"Test MAE: {results[best_model_name]['test_mae']:.2f}")
print(f"Test R2 : {results[best_model_name]['test_r2']:.3f}")
print("=" * 70)


# Save Model and  encoders

joblib.dump(best_model, f"{MODEL_DIR}/amazon_best_model.pkl")
joblib.dump(encoder, f"{MODEL_DIR}/Weather_encoder.pkl")

print("Model and encoder saved successfully.")
