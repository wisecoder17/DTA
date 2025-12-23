import pandas as pd
import numpy as np
import joblib
from haversine import haversine


MODEL_PATH = "models/amazon_best_model.pkl"
ENCODER_PATH = "models/Weather_encoder.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

FEATURES = [
    "distance_km",
    "hour",
    "day_of_week",
    "is_weekend",
    "weather",
    "traffic_level"
]


# Haversine distance calculation
def compute_distance_km(row):
    coord_store = (row["Store_Latitude"], row["Store_Longitude"])
    coord_drop  = (row["Drop_Latitude"], row["Drop_Longitude"])
    return haversine(coord_store, coord_drop)



def predict_delivery(df):
    # Compute distance
    df["distance_km"] = df.apply(compute_distance_km, axis=1)

    # Temporal features
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["day_of_week"] = df["Order_Date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["Order_Time"] = pd.to_datetime(df["Order_Time"], errors="coerce")
    df["hour"] = df["Order_Time"].dt.hour

    # Traffic mapping
    traffic_mapping = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    df["traffic_level"] = df["Traffic"].str.strip().map(traffic_mapping)

    # One-hot encode weather
    weather_encoded = encoder.transform(df[["Weather"]])
    df_numeric = df[["distance_km", "hour", "day_of_week", "is_weekend", "traffic_level"]].values
    X_final = np.hstack([df_numeric, weather_encoded])

    # Predict
    df["predicted_delivery_time"] = model.predict(X_final)
    return df[["Order_ID", "predicted_delivery_time"]]

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Load new orders
    new_orders = pd.read_csv("data/new_orders.csv")
    predictions = predict_delivery(new_orders)
    print(predictions)

    # Save predictions
    predictions.to_csv("data/predicted_delivery_times.csv", index=False)
    print("Predictions saved to data/predicted_delivery_times.csv")
