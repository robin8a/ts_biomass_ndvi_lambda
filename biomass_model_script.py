
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def generate_synthetic_data(num_samples=1000):
    """Generates synthetic data for NDVI, EVI, and biomass."""
    np.random.seed(42)
    # Simulate NDVI values (0.1 to 0.9, typically)
    ndvi = np.random.uniform(0.1, 0.9, num_samples)
    # Simulate EVI values (0.0 to 0.8, typically, often correlated with NDVI)
    evi = ndvi * 0.9 + np.random.uniform(-0.1, 0.1, num_samples)
    evi = np.clip(evi, 0.0, 0.8) # Clip to valid EVI range

    # Simulate biomass based on a non-linear relationship with NDVI and EVI
    # Add some noise to simulate real-world variability
    biomass = 50 + 150 * ndvi + 100 * evi**2 + np.random.normal(0, 15, num_samples)
    biomass = np.clip(biomass, 0, 500) # Biomass cannot be negative

    data = pd.DataFrame({
        'NDVI': ndvi,
        'EVI': evi,
        'Biomass': biomass
    })
    return data

def train_model(data):
    """Trains a RandomForestRegressor model."""
    X = data[['NDVI', 'EVI']]
    y = data['Biomass']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

if __name__ == "__main__":
    print("Generating synthetic data...")
    synthetic_data = generate_synthetic_data()
    print("Synthetic data head:\n", synthetic_data.head())

    print("Training regression model...")
    model = train_model(synthetic_data)
    print("Model training complete.")

    # Save the trained model
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "biomass_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Simple test prediction
    test_input = pd.DataFrame({'NDVI': [0.7], 'EVI': [0.5]})
    prediction = model.predict(test_input)
    print(f"Test prediction for NDVI=0.7, EVI=0.5: {prediction[0]:.2f}")


