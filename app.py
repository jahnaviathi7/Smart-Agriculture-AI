
import streamlit as st
import sys
from pathlib import Path
import joblib
import pandas as pd
from agriculture_chatbot import ask_agriculture_ai
# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🌱 Smart Agriculture AI")

page = st.sidebar.selectbox(
    "Select Module",
    [
        "🌾 Crop Recommendation",
        "📊 Crop Yield Prediction",
        "🍃 Plant Disease Detection",
        "💧 Irrigation Recommendation",
        "🧪 Fertilizer Recommendation",
        "🌦️ Weather Analysis",
        "🤖 Agriculture AI Chatbot"
    ]
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

sys.path.append(str(BASE_DIR / "src"))


# --------------------------------------------------
# Import ML modules
# --------------------------------------------------

from crop_prediction import predict_crop
from yield_prediction import predict_yield


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌱",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🌱 Smart Agriculture AI")

st.write(
    "AI-powered crop recommendation and crop yield prediction "
    "using soil and environmental conditions."
)

st.divider()


# --------------------------------------------------
# Farm Inputs
# --------------------------------------------------

st.header("🌾 Enter Farm Conditions")

col1, col2 = st.columns(2)


with col1:

    N = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        max_value=200.0,
        value=90.0
    )

    P = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        max_value=200.0,
        value=42.0
    )

    K = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        max_value=200.0,
        value=43.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=25.0
    )


with col2:

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=5000.0,
        value=200.0
    )


# --------------------------------------------------
# Crop selection
# --------------------------------------------------

st.subheader("🌾 Crop Selection")

crops = [
    "rice",
    "maize",
    "wheat",
    "cotton",
    "sugarcane",
    "groundnut",
    "chickpea",
    "banana",
    "mango",
    "tomato"
]

selected_crop = st.selectbox(
    "Select crop for yield prediction",
    crops
)


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🚜 Analyze Farm",
    type="primary",
    use_container_width=True
):

    # Crop recommendation
    recommended_crop = predict_crop(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    )

    # Yield prediction
    predicted_yield = predict_yield(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall,
        selected_crop
    )


    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    st.header("📊 AI Prediction Results")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.success(
            f"🌱 Recommended Crop\n\n"
            f"### {recommended_crop.upper()}"
        )


    with result_col2:

        st.info(
            f"📊 Estimated Yield\n\n"
            f"### {predicted_yield:,.2f} kg/hectare"
        )
        import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌱",
    layout="wide"
)


# --------------------------------------------------
# 38 PLANTVILLAGE CLASSES
# --------------------------------------------------

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "notebook"
    / "plant_disease_mobilenetv2_final_38_classes.keras"
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found:\n{MODEL_PATH}")
        st.stop()

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()



# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_disease(image):

    image = image.resize((224, 224))

    image_array = np.array(image)

    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    disease = CLASS_NAMES[predicted_index]

    confidence = predictions[predicted_index] * 100

    return disease, confidence, predictions


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🌱 Smart Agriculture AI")

st.sidebar.info(
    """
    AI-powered plant disease detection
    using MobileNetV2 Transfer Learning.

    Dataset:
    PlantVillage

    Classes:
    38
    """
)


# --------------------------------------------------
# MAIN PAGE
# --------------------------------------------------

st.title("🌱 Smart Agriculture AI")

st.subheader(
    "AI-Based Plant Disease Detection System"
)

st.write(
    "Upload a plant leaf image to identify "
    "the possible disease using our trained "
    "38-class MobileNetV2 model."
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Leaf",
            use_container_width=True
        )

    with col2:

        st.write("### 🔍 Analysis")

        if st.button(
            "🚀 Predict Disease",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing leaf image..."
            ):

                disease, confidence, predictions = (
                    predict_disease(image)
                )

            st.success(
                "Prediction completed!"
            )

            st.write(
                "### 🌿 Predicted Disease"
            )

            st.info(
                disease.replace("_", " ")
            )

            st.write(
                "### 🎯 Confidence Score"
            )

            st.progress(
                min(int(confidence), 100)
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


            # --------------------------------------
            # TOP 5 PREDICTIONS
            # --------------------------------------

            st.write(
                "### 🔝 Top 5 Predictions"
            )

            top5 = np.argsort(
                predictions
            )[-5:][::-1]

            for index in top5:

                st.write(
                    f"**{CLASS_NAMES[index].replace('_', ' ')}** "
                    f"— {predictions[index] * 100:.2f}%"
                )


else:

    st.info(
        "👆 Upload a leaf image to start prediction."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Smart Agriculture AI | "
    "PlantVillage 38-Class Disease Detection"
)
@st.cache_resource
def load_irrigation_models():

    model = joblib.load(
        "models/irrigation_model.pkl"
    )

    crop_encoder = joblib.load(
        "models/irrigation_crop_encoder.pkl"
    )

    soil_encoder = joblib.load(
        "models/irrigation_soil_encoder.pkl"
    )

    stage_encoder = joblib.load(
        "models/irrigation_stage_encoder.pkl"
    )

    target_encoder = joblib.load(
        "models/irrigation_target_encoder.pkl"
    )

    return (
        model,
        crop_encoder,
        soil_encoder,
        stage_encoder,
        target_encoder
    )
(
    irrigation_model,
    irrigation_crop_encoder,
    irrigation_soil_encoder,
    irrigation_stage_encoder,
    irrigation_target_encoder
) = load_irrigation_models()
st.header("💧 Smart Irrigation Recommendation")

crop = st.selectbox(
    "🌱 Select Crop",
    irrigation_crop_encoder.classes_
)

soil_type = st.selectbox(
    "🌍 Select Soil Type",
    irrigation_soil_encoder.classes_
)

temperature = st.number_input(
    "🌡️ Temperature (°C)",
    min_value=0.0,
    max_value=60.0,
    value=30.0
)

humidity = st.number_input(
    "💦 Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

soil_moisture = st.number_input(
    "💧 Soil Moisture (%)",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

rainfall = st.number_input(
    "🌧️ Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=20.0
)

growth_stage = st.selectbox(
    "🌱 Growth Stage",
    irrigation_stage_encoder.classes_
)

if st.button("💧 Predict Irrigation Requirement"):

    input_data = pd.DataFrame([{
        "Crop": irrigation_crop_encoder.transform([crop])[0],
        "Soil_Type": irrigation_soil_encoder.transform([soil_type])[0],
        "Temperature": temperature,
        "Humidity": humidity,
        "Soil_Moisture": soil_moisture,
        "Rainfall": rainfall,
        "Growth_Stage": irrigation_stage_encoder.transform(
            [growth_stage]
        )[0]
    }])

    prediction = irrigation_model.predict(input_data)[0]

    result = irrigation_target_encoder.inverse_transform(
        [prediction]
    )[0]

    probabilities = irrigation_model.predict_proba(
        input_data
    )[0]

    confidence = max(probabilities) * 100

    st.success(
        f"💧 Irrigation Requirement: **{result}**"
    )

    st.metric(
        "🎯 Confidence",
        f"{confidence:.2f}%"
    )

    if result == "High":
        st.warning(
            "🚨 High irrigation requirement. "
            "Consider irrigating the crop soon."
        )

    elif result == "Medium":
        st.info(
            "⚠️ Moderate irrigation requirement. "
            "Monitor soil moisture and weather conditions."
        )

    else:
        st.success(
            "✅ Low irrigation requirement. "
            "Immediate irrigation may not be necessary."
        )
if page == "🧪 Fertilizer Recommendation":

    st.header("🧪 Fertilizer Recommendation")

    import joblib
    import pandas as pd

    # Load models
    fertilizer_model = joblib.load(
        "models/fertilizer_model.pkl"
    )

    fertilizer_crop_encoder = joblib.load(
        "models/fertilizer_crop_encoder.pkl"
    )

    fertilizer_soil_encoder = joblib.load(
        "models/fertilizer_soil_encoder.pkl"
    )

    fertilizer_target_encoder = joblib.load(
        "models/fertilizer_target_encoder.pkl"
    )

    # Inputs
    crop = st.selectbox(
        "🌾 Select Crop",
        fertilizer_crop_encoder.classes_
    )

    soil_type = st.selectbox(
        "🌱 Select Soil Type",
        fertilizer_soil_encoder.classes_
    )

    nitrogen = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        max_value=200.0,
        value=25.0
    )

    phosphorus = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        max_value=150.0,
        value=50.0
    )

    potassium = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        max_value=150.0,
        value=45.0
    )

    ph = st.number_input(
        "Soil pH",
        min_value=3.0,
        max_value=10.0,
        value=6.5
    )

    organic_matter = st.number_input(
        "Organic Matter",
        min_value=0.0,
        max_value=15.0,
        value=4.0
    )

    if st.button("🧪 Recommend Fertilizer"):

        crop_encoded = fertilizer_crop_encoder.transform(
            [crop]
        )[0]

        soil_encoded = fertilizer_soil_encoder.transform(
            [soil_type]
        )[0]

        input_data = pd.DataFrame([{
            "Crop": crop_encoded,
            "Soil_Type": soil_encoded,
            "Nitrogen": nitrogen,
            "Phosphorus": phosphorus,
            "Potassium": potassium,
            "pH": ph,
            "Organic_Matter": organic_matter
        }])

        prediction = fertilizer_model.predict(
            input_data
        )[0]

        fertilizer = fertilizer_target_encoder.inverse_transform(
            [prediction]
        )[0]

        probabilities = fertilizer_model.predict_proba(
            input_data
        )[0]

        confidence = max(probabilities) * 100

        st.success(
            f"🧪 Recommended Fertilizer: **{fertilizer}**"
        )

        st.info(
            f"🎯 Confidence: **{confidence:.2f}%**"
        )
    elif page == "🌦️ Weather Analysis":

         st.header("🌦️ Weather Analysis")

    from weather_analysis import (
        get_location,
        get_weather,
        weather_description,
        agriculture_advisory
    )

    location = st.text_input(
        "📍 Enter your city/location",
        value="Tirupati"
    )

    if st.button("🌦️ Get Weather"):

        try:

            location_data = get_location(location)

            if location_data is None:

                st.error("❌ Location not found.")

            else:

                st.success(
                    f"📍 {location_data['name']}, "
                    f"{location_data['country']}"
                )

                weather = get_weather(
                    location_data["latitude"],
                    location_data["longitude"]
                )

                current = weather["current"]

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "🌡️ Temperature",
                        f"{current['temperature_2m']} °C"
                    )

                with col2:
                    st.metric(
                        "💧 Humidity",
                        f"{current['relative_humidity_2m']}%"
                    )

                with col3:
                    st.metric(
                        "🌧️ Precipitation",
                        f"{current['precipitation']} mm"
                    )

                with col4:
                    st.metric(
                        "💨 Wind Speed",
                        f"{current['wind_speed_10m']} km/h"
                    )

                st.subheader("☁️ Weather Condition")

                st.write(
                    weather_description(
                        current["weather_code"]
                    )
                )

                st.subheader("🌾 Farming Advisory")

                advice = agriculture_advisory(weather)

                for item in advice:
                    st.write(item)

        except Exception as e:

            st.error(
                f"Weather error: {e}"
            )

if page == "🤖 Agriculture AI Chatbot":

    st.header("🤖 Agriculture AI Chatbot")

    st.write(
        "Ask questions about crops, diseases, irrigation, "
        "fertilizers, soil, weather and farming."
    )

    # Initialize chat history
    if "agri_messages" not in st.session_state:
        st.session_state.agri_messages = []

    # Display previous messages
    for message in st.session_state.agri_messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input(
        "Ask your agriculture question..."
    )

    if prompt:

        st.session_state.agri_messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner(
                "🌾 Agriculture AI is thinking..."
            ):

                try:

                    response = ask_agriculture_ai(
                        prompt
                    )

                    st.markdown(response)

                    st.session_state.agri_messages.append({
                        "role": "assistant",
                        "content": response
                    })

                except Exception as e:

                    st.error(
                        f"Chatbot error: {e}"
                    )

    # Clear chat
    if st.button("🗑️ Clear Chat"):

        st.session_state.agri_messages = []

        st.rerun()