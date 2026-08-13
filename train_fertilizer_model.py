import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 1. CREATE DIRECTORIES
# ============================================================

os.makedirs("data/fertilizer", exist_ok=True)
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

fertilizers = [
    "Urea",
    "DAP",
    "NPK",
    "Potassium",
    "Organic Compost"
]

data = {
    "Crop": np.random.choice(crops, n_samples),
    "Soil_Type": np.random.choice(soil_types, n_samples),
    "Nitrogen": np.random.uniform(10, 150, n_samples),
    "Phosphorus": np.random.uniform(5, 100, n_samples),
    "Potassium": np.random.uniform(5, 100, n_samples),
    "pH": np.random.uniform(4.5, 8.5, n_samples),
    "Organic_Matter": np.random.uniform(1, 8, n_samples)
}

df = pd.DataFrame(data)

# ============================================================
# 3. CREATE FERTILIZER RECOMMENDATION
# ============================================================

def fertilizer_decision(row):

    N = row["Nitrogen"]
    P = row["Phosphorus"]
    K = row["Potassium"]
    pH = row["pH"]
    OM = row["Organic_Matter"]

    # Very low nitrogen
    if N < 40:
        return "Urea"

    # Low phosphorus
    elif P < 25:
        return "DAP"

    # Low potassium
    elif K < 25:
        return "Potassium"

    # Low organic matter
    elif OM < 2.5:
        return "Organic Compost"

    # Balanced nutrients
    else:
        return "NPK"


df["Fertilizer_Recommendation"] = df.apply(
    fertilizer_decision,
    axis=1
)

# ============================================================
# 4. SAVE DATASET
# ============================================================

dataset_path = "data/fertilizer/fertilizer_dataset.csv"

df.to_csv(dataset_path, index=False)

print("\nDataset created successfully!")
print(f"Saved to: {dataset_path}")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nFertilizer distribution:")
print(
    df["Fertilizer_Recommendation"].value_counts()
)

# ============================================================
# 5. ENCODE CATEGORICAL FEATURES
# ============================================================

crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
target_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])

df["Soil_Type"] = soil_encoder.fit_transform(
    df["Soil_Type"]
)

df["Fertilizer_Recommendation"] = (
    target_encoder.fit_transform(
        df["Fertilizer_Recommendation"]
    )
)

# ============================================================
# 6. FEATURES AND TARGET
# ============================================================

X = df[
    [
        "Crop",
        "Soil_Type",
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "pH",
        "Organic_Matter"
    ]
]

y = df["Fertilizer_Recommendation"]

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
# 8. RANDOM FOREST MODEL
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

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================================")
print("FERTILIZER MODEL RESULTS")
print("========================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_
    )
)

# ============================================================
# 10. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/fertilizer_model.pkl"
)

joblib.dump(
    crop_encoder,
    "models/fertilizer_crop_encoder.pkl"
)

joblib.dump(
    soil_encoder,
    "models/fertilizer_soil_encoder.pkl"
)

joblib.dump(
    target_encoder,
    "models/fertilizer_target_encoder.pkl"
)

print("\nModels saved successfully!")

print("\nFiles created:")

print(
    "models/fertilizer_model.pkl"
)

print(
    "models/fertilizer_crop_encoder.pkl"
)

print(
    "models/fertilizer_soil_encoder.pkl"
)

print(
    "models/fertilizer_target_encoder.pkl"
)