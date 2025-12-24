import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


MODEL_PATH = "models/amazon_best_model.pkl"
ENCODER_PATH = "models/Weather_encoder.pkl"
DATA_PATH  = "data/processed/amazon_processed.csv"

FIG_DIR   = "results/figures"
TABLE_DIR = "results/tables"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


model   = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
df      = pd.read_csv(DATA_PATH)


FEATURES = ["distance_km", "hour", "day_of_week", "is_weekend", "Weather", "traffic_level"]
TARGET   = "Delivery_Time"


# Temporal Train/Test Split
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df.sort_values("Order_Date", inplace=True)

n = len(df)
train_end = int(0.7 * n)
val_end   = int(0.85 * n)

train_df = df.iloc[:train_end]
test_df  = df.iloc[val_end:]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test  = test_df[FEATURES]
y_test  = test_df[TARGET]

# encoding weather

X_train_weather = encoder.transform(X_train[["Weather"]])
X_test_weather  = encoder.transform(X_test[["Weather"]])

X_train_num = X_train.drop(columns=["Weather"]).values
X_test_num  = X_test.drop(columns=["Weather"]).values

X_train_final = np.hstack([X_train_num, X_train_weather])
X_test_final  = np.hstack([X_test_num, X_test_weather])

# Predictions
y_pred = model.predict(X_test_final)
y_test_df = test_df.copy()
y_test_df["predicted_delivery_time"] = y_pred
y_test_df["error"] = y_test_df["predicted_delivery_time"] - y_test_df[TARGET]

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2]
})

metrics_df.to_csv(f"{TABLE_DIR}/amazon_test_metrics.csv", index=False)
print("Metrics saved to CSV")
print(metrics_df)

# Feature Importance (if XGBoost)
if hasattr(model, "feature_importances_"):
    feature_names = ["distance_km", "hour", "day_of_week", "is_weekend"] + list(encoder.get_feature_names_out(["Weather"])) + ["traffic_level"]
    importances = model.feature_importances_
    
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    fi_df.to_csv(f"{TABLE_DIR}/amazon_feature_importance.csv", index=False)
    
    plt.figure(figsize=(10,6))
    sns.barplot(x="importance", y="feature", data=fi_df, palette="viridis")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/amazon_feature_importance_xgb.png")
    plt.close()
    print("Feature importance plot saved")

# Predicted vs Actual Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Delivery Time (minutes)")
plt.ylabel("Predicted Delivery Time (minutes)")
plt.title("Predicted vs Actual Delivery Time")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/amazon_actual_vs_predicted.png")
plt.close()
print("Predicted vs Actual plot saved")

# Error Analysis

# Absolute error
y_test_df["abs_error"] = y_test_df["error"].abs()

# Top 5% largest errors
threshold = y_test_df["abs_error"].quantile(0.95)
worst_cases = y_test_df[y_test_df["abs_error"] >= threshold].copy()
worst_cases.to_csv(f"{TABLE_DIR}/amazon_error_analysis_top5.csv", index=False)
print("Error analysis (top 5%) saved")

# Histogram of errors
plt.figure(figsize=(8,6))
sns.histplot(y_test_df["error"], bins=50, kde=True, color='coral')
plt.title("Distribution of Prediction Errors")
plt.xlabel("Prediction Error (minutes)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/amazon_prediction_error_distribution.png")
plt.close()
print("Error distribution plot saved")
