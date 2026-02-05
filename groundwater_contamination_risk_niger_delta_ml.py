
# ============================================================
# Groundwater Contamination Risk Mapping in the Niger Delta
# Using GIS and Machine Learning (Synthetic Data)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# 1. Synthetic GIS & Hydrogeological Data Generator
# ------------------------------------------------------------

def generate_groundwater_data(samples=3500):
    np.random.seed(42)

    depth_to_water_m = np.random.normal(12, 5, samples).clip(2, 40)
    soil_permeability = np.random.uniform(0.2, 0.9, samples)
    distance_to_river_km = np.random.exponential(2.0, samples).clip(0.1, 10)
    distance_to_industry_km = np.random.exponential(3.5, samples).clip(0.2, 15)
    nitrate_mgL = np.random.normal(18, 7, samples).clip(1, 60)
    population_density = np.random.normal(450, 220, samples).clip(50, 1500)

    risk_index = (
        0.25 * (1 - depth_to_water_m / 40) +
        0.20 * soil_permeability +
        0.15 * (1 - distance_to_river_km / 10) +
        0.20 * (1 - distance_to_industry_km / 15) +
        0.10 * (nitrate_mgL / 60) +
        0.10 * (population_density / 1500)
    )

    contamination_risk = np.select(
        [risk_index < 0.35, risk_index < 0.6, risk_index >= 0.6],
        [0, 1, 2]
    )

    return pd.DataFrame({
        "depth_to_water_m": depth_to_water_m,
        "soil_permeability": soil_permeability,
        "distance_to_river_km": distance_to_river_km,
        "distance_to_industry_km": distance_to_industry_km,
        "nitrate_mgL": nitrate_mgL,
        "population_density": population_density,
        "contamination_risk": contamination_risk
    })

# Generate dataset
df = generate_groundwater_data()

# ------------------------------------------------------------
# 2. Feature Scaling & Train-Test Split
# ------------------------------------------------------------

X = df.drop("contamination_risk", axis=1)
y = df["contamination_risk"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# 3. Machine Learning Model
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("Groundwater Contamination Risk Model Performance")
print(classification_report(
    y_test, y_pred,
    target_names=["Low Risk", "Moderate Risk", "High Risk"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------
# 5. Feature Importance
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.barh(X.columns, model.feature_importances_)
plt.xlabel("Importance Score")
plt.title("Groundwater Contamination Drivers")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Spatial Risk Mapping (GIS-style Output)
# ------------------------------------------------------------

def simulate_spatial_risk_map(grid_size=100):
    depth = np.random.uniform(3, 35, (grid_size, grid_size))
    permeability = np.random.uniform(0.2, 0.9, (grid_size, grid_size))
    river_dist = np.random.uniform(0.1, 10, (grid_size, grid_size))
    industry_dist = np.random.uniform(0.2, 15, (grid_size, grid_size))
    nitrate = np.random.uniform(2, 55, (grid_size, grid_size))
    pop_density = np.random.uniform(100, 1400, (grid_size, grid_size))

    stacked = np.stack([
        depth, permeability, river_dist,
        industry_dist, nitrate, pop_density
    ], axis=-1)

    reshaped = stacked.reshape(-1, 6)
    reshaped_scaled = scaler.transform(reshaped)

    prediction = model.predict(reshaped_scaled)
    risk_map = prediction.reshape(grid_size, grid_size)

    plt.figure(figsize=(6, 6))
    plt.imshow(risk_map, cmap="RdYlGn_r")
    plt.title("Predicted Groundwater Contamination Risk Map")
    plt.colorbar(label="0=Low | 1=Moderate | 2=High")
    plt.axis("off")
    plt.show()

simulate_spatial_risk_map()
