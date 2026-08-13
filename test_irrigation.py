import joblib
import pandas as pd

# Load model
model = joblib.load("models/irrigation_model.pkl")

# Load encoders
crop_encoder = joblib.load("models/irrigation_crop_encoder.pkl")
soil_encoder = joblib.load("models/irrigation_soil_encoder.pkl")
stage_encoder = joblib.load("models/irrigation_stage_encoder.pkl")
target_encoder = joblib.load("models/irrigation_target_encoder.pkl")

# -------------------------------------------------
# SAMPLE FARMER INPUT
# -------------------------------------------------

crop = "Rice"
soil_type = "Loamy"
temperature = 36
humidity = 35
soil_moisture = 25
rainfall = 10
growth_stage = "Flowering"

# -------------------------------------------------
# ENCODE INPUT
# -------------------------------------------------

input_data = pd.DataFrame([{
    "Crop": crop_encoder.transform([crop])[0],
    "Soil_Type": soil_encoder.transform([soil_type])[0],
    "Temperature": temperature,
    "Humidity": humidity,
    "Soil_Moisture": soil_moisture,
    "Rainfall": rainfall,
    "Growth_Stage": stage_encoder.transform([growth_stage])[0]
}])

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

prediction = model.predict(input_data)[0]

result = target_encoder.inverse_transform([prediction])[0]

# Probability
probabilities = model.predict_proba(input_data)[0]

confidence = max(probabilities) * 100

# -------------------------------------------------
# RESULT
# -------------------------------------------------

print("\n====================================")
print("💧 SMART IRRIGATION RECOMMENDATION")
print("====================================")

print(f"Crop              : {crop}")
print(f"Soil Type         : {soil_type}")
print(f"Temperature       : {temperature} °C")
print(f"Humidity          : {humidity}%")
print(f"Soil Moisture     : {soil_moisture}%")
print(f"Rainfall          : {rainfall} mm")
print(f"Growth Stage      : {growth_stage}")

print("------------------------------------")

print(f"Irrigation Need   : {result}")
print(f"Confidence        : {confidence:.2f}%")

print("====================================")