import joblib
import pandas as pd
from pathlib import Path


# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Model path
MODEL_PATH = BASE_DIR / "models" / "crop_model.pkl"


# Load trained model
model = joblib.load(MODEL_PATH)


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):

    input_data = pd.DataFrame({
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall]
    })

    prediction = model.predict(input_data)

    return prediction[0]