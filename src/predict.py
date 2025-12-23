import os
import pandas as pd
import numpy as np
import joblib
from haversine import haversine

MODEL_PATH = "models/amazon_best_model.pkl"
ENCODER_PATH = "models/Weather_encoder.pkl"

DAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}

TRAFFIC_MAP = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}

FEATURES_NUMERIC = ["distance_km", "hour", "day_of_week", "is_weekend", "traffic_level"]
FEATURES_CAT = ["Weather"]

def init_model(model_path=MODEL_PATH, encoder_path=ENCODER_PATH):
    """Load trained model and encoder once."""
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError("Model or encoder not found. Train the model first.")
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

def compute_distance_km(row):
    """Compute Haversine distance if coordinates are provided."""
    coord_store = (row.get("Store_Latitude"), row.get("Store_Longitude"))
    coord_drop = (row.get("Drop_Latitude"), row.get("Drop_Longitude"))
    if None in coord_store or None in coord_drop:
        raise ValueError("Coordinates missing for Haversine calculation.")
    return haversine(coord_store, coord_drop)

def preprocess_input(df):
    """Prepare features for prediction."""
    # Compute distance if missing
    if "distance_km" not in df.columns:
        df["distance_km"] = df.apply(compute_distance_km, axis=1)

    # Map day_of_week string to integer if needed
    if df["day_of_week"].dtype == object:
        df["day_of_week"] = df["day_of_week"].map(DAY_MAP)

    # Map traffic_level
    if df["traffic_level"].dtype == object:
        df["traffic_level"] = df["traffic_level"].str.strip().map(TRAFFIC_MAP)

    # Check mandatory columns
    missing_cols = [col for col in FEATURES_NUMERIC + FEATURES_CAT if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for prediction: {missing_cols}")

    return df

def predict_delivery(df, model, encoder):
    """Return predicted delivery times for a DataFrame of orders."""
    df = preprocess_input(df)

    # Encode Weather
    weather_encoded = encoder.transform(df[["Weather"]])
    df_numeric = df[FEATURES_NUMERIC].values
    X_final = np.hstack([df_numeric, weather_encoded])

    df["predicted_delivery_time"] = model.predict(X_final)
    return df[["predicted_delivery_time"]]

def predict_from_csv(csv_path, output_path="results/tables/predictions.csv"):
    """Load CSV, predict, and save results."""
    df = pd.read_csv(csv_path)
    model, encoder = init_model()
    predictions_df = predict_delivery(df, model, encoder)

    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return predictions_df

# Example usage
if __name__ == "__main__":
    model, encoder = init_model()
    sample_df = pd.DataFrame([{
        "distance_km": 19.5,
        "day_of_week": "Monday",
        "hour": 14,
        "Weather": "Clear",
        "traffic_level": "Medium",
        "is_weekend": 0
    }])
    print(predict_delivery(sample_df, model, encoder))
