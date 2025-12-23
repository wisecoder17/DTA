import pandas as pd

def load_dataset(file_path):
    return pd.read_csv(file_path)

def haversine(coord1, coord2):
    import numpy as np
    R = 6371  # Earth radius in km
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def preprocess_amazon(df):
    # Distance
    df['distance_km'] = df.apply(lambda row: haversine(
        (row['Store_Latitude'], row['Store_Longitude']),
        (row['Drop_Latitude'], row['Drop_Longitude'])
    ), axis=1)

    # Temporal features
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df['day_of_week'] = df['Order_Date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['Order_Time'] = pd.to_datetime(df['Order_Time'], errors='coerce')
    assert df['Order_Date'].notna().mean() > 0.95
    df['hour'] = df['Order_Time'].dt.hour
    
    # Traffic condition
    df['Traffic'] = df['Traffic'].str.strip().str.capitalize()
    df = df[df['Traffic'].isin(['Low', 'Medium', 'High', 'Jam'])]
    
    traffic_mapping = {'Low':1, 'Medium':2, 'High':3, 'Jam':4}
    df['traffic_level'] = df['Traffic'].map(traffic_mapping).astype(int)

    df.drop(columns=['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude', 'Order_Time', 'Traffic'], inplace=True)
    
    df['Weather'] = df['Weather'].astype('category')
    df['Vehicle'] = df['Vehicle'].astype('category')
   
    return df


datasets = {
    'amazon': 'data/raw/amazon_delivery.csv',
}

for name, path in datasets.items():
    df = load_dataset(path)
    
    # Call the dataset-specific feature engineering
    if name == 'amazon':
        df = preprocess_amazon(df)
    else:
        pass
    
    # Save processed CSV
    df.to_csv(f'data/processed/{name}_processed.csv', index=False)
    
    print(f"{name} processed: {df.shape[0]} rows, {df.shape[1]} columns")

