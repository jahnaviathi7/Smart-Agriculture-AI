import joblib
import pandas as pd

# Load model
model = joblib.load("models/fertilizer_model.pkl")

# Load encoders
crop_encoder = joblib.load(
    "models/fertilizer_crop_encoder.pkl"
)

soil_encoder = joblib.load(
    "models/fertilizer_soil_encoder.pkl"
)

target_encoder = joblib.load(
    "models/fertilizer_target_encoder.pkl"
)

# -------------------------------------------------
# SAMPLE FARMER INPUT
# -------------------------------------------------

crop = "Rice"
soil_type = "Loamy"

nitrogen = 25
phosphorus = 50
potassium = 45
ph = 6.5
organic_matter = 4.0

# -------------------------------------------------
# ENCODE INPUT
# -------------------------------------------------

input_data = pd.DataFrame([{
    "Crop": crop_encoder.transform([crop])[0],
    "Soil_Type": soil_encoder.transform([soil_type])[0],
    "Nitrogen": nitrogen,
    "Phosphorus": phosphorus,
    "Potassium": potassium,
    "pH": ph,
    "Organic_Matter": organic_matter
}])

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

prediction = model.predict(input_data)[0]

result = target_encoder.inverse_transform(
    [prediction]
)[0]

# Confidence
probabilities = model.predict_proba(input_data)[0]
confidence = max(probabilities) * 100

# -------------------------------------------------
# RESULT
# -------------------------------------------------

print("\n====================================")
print("🧪 SMART FERTILIZER RECOMMENDATION")
print("====================================")

print(f"Crop              : {crop}")
print(f"Soil Type         : {soil_type}")
print(f"Nitrogen          : {nitrogen}")
print(f"Phosphorus        : {phosphorus}")
print(f"Potassium         : {potassium}")
print(f"pH                : {ph}")
print(f"Organic Matter    : {organic_matter}")

print("------------------------------------")

print(f"Recommended Fertilizer : {result}")
print(f"Confidence             : {confidence:.2f}%")

print("====================================")