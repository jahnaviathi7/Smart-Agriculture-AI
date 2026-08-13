import joblib
import pandas as pd
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model and encoder paths
MODEL_PATH = BASE_DIR / "models" / "yield_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "crop_encoder.pkl"


# Load trained model and crop encoder
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


def predict_yield(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    crop
):
    """
    Predict crop yield in kg/hectare.
    """

    crop_encoded = encoder.transform([crop])[0]

    input_data = pd.DataFrame({
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall],
        "crop_encoded": [crop_encoded]
    })

    prediction = model.predict(input_data)

    return prediction[0]