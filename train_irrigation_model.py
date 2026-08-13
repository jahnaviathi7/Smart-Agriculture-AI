import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 1. CREATE DIRECTORIES
# ============================================================

os.makedirs("data/irrigation", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# 2. GENERATE DATASET
# ============================================================

np.random.seed(42)

n_samples = 5000

crops = [
    "Rice",
    "Wheat",
    "Maize",
    "Cotton",
    "Tomato",
    "Groundnut",
    "Sugarcane",
    "Potato"
]

soil_types = [
    "Sandy",
    "Loamy",
    "Clay",
    "Silty"
]

growth_stages = [
    "Seedling",
    "Vegetative",
    "Flowering",
    "Fruiting",
    "Maturity"
]

data = {
    "Crop": np.random.choice(crops, n_samples),
    "Soil_Type": np.random.choice(soil_types, n_samples),
    "Temperature": np.random.uniform(15, 45, n_samples),
    "Humidity": np.random.uniform(20, 95, n_samples),
    "Soil_Moisture": np.random.uniform(10, 90, n_samples),
    "Rainfall": np.random.uniform(0, 200, n_samples),
    "Growth_Stage": np.random.choice(growth_stages, n_samples)
}

df = pd.DataFrame(data)

# ============================================================
# 3. CREATE IRRIGATION LABEL
# ============================================================

def irrigation_decision(row):

    score = 0

    # Low soil moisture → more irrigation
    if row["Soil_Moisture"] < 30:
        score += 3
    elif row["Soil_Moisture"] < 45:
        score += 2

    # High temperature → more irrigation
    if row["Temperature"] > 35:
        score += 2
    elif row["Temperature"] > 30:
        score += 1

    # Low humidity → more irrigation
    if row["Humidity"] < 40:
        score += 2
    elif row["Humidity"] < 55:
        score += 1

    # Low rainfall → more irrigation
    if row["Rainfall"] < 20:
        score += 2
    elif row["Rainfall"] < 50:
        score += 1

    # Crop growth stage
    if row["Growth_Stage"] in ["Flowering", "Fruiting"]:
        score += 1

    if score >= 5:
        return "High"
    elif score >= 3:
        return "Medium"
    else:
        return "Low"


df["Irrigation_Need"] = df.apply(irrigation_decision, axis=1)

# ============================================================
# 4. SAVE DATASET
# ============================================================

dataset_path = "data/irrigation/irrigation_dataset.csv"

df.to_csv(dataset_path, index=False)

print("\nDataset created successfully!")
print(f"Saved to: {dataset_path}")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nIrrigation distribution:")
print(df["Irrigation_Need"].value_counts())

# ============================================================
# 5. ENCODE CATEGORICAL FEATURES
# ============================================================

crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
stage_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Soil_Type"] = soil_encoder.fit_transform(df["Soil_Type"])
df["Growth_Stage"] = stage_encoder.fit_transform(df["Growth_Stage"])

target_encoder = LabelEncoder()

df["Irrigation_Need"] = target_encoder.fit_transform(
    df["Irrigation_Need"]
)

# ============================================================
# 6. FEATURES AND TARGET
# ============================================================

X = df[
    [
        "Crop",
        "Soil_Type",
        "Temperature",
        "Humidity",
        "Soil_Moisture",
        "Rainfall",
        "Growth_Stage"
    ]
]

y = df["Irrigation_Need"]

# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 8. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ============================================================
# 9. EVALUATION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("IRRIGATION MODEL RESULTS")
print("========================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_
    )
)

# ============================================================
# 10. SAVE MODEL AND ENCODERS
# ============================================================

import joblib

joblib.dump(
    model,
    "models/irrigation_model.pkl"
)

joblib.dump(
    crop_encoder,
    "models/irrigation_crop_encoder.pkl"
)

joblib.dump(
    soil_encoder,
    "models/irrigation_soil_encoder.pkl"
)

joblib.dump(
    stage_encoder,
    "models/irrigation_stage_encoder.pkl"
)

joblib.dump(
    target_encoder,
    "models/irrigation_target_encoder.pkl"
)

print("\nModels saved successfully!")

print("\nFiles created:")
print("models/irrigation_model.pkl")
print("models/irrigation_crop_encoder.pkl")
print("models/irrigation_soil_encoder.pkl")
print("models/irrigation_stage_encoder.pkl")
print("models/irrigation_target_encoder.pkl")