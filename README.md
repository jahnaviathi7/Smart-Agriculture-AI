# 🌱 Smart Agriculture AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge\&logo=python\&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge\&logo=streamlit\&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge\&logo=github)

**Smart Agriculture AI** is an AI and Machine Learning based agriculture assistance system designed to help farmers make better decisions using data-driven recommendations.

The system combines multiple agricultural intelligence modules including crop recommendation, crop yield prediction, irrigation recommendation, fertilizer recommendation, weather analysis, plant disease detection, and AI-powered farmer assistance.

---

## 📑 Table of Contents

* [✨ Features](#-features)
* [🧠 System Architecture](#-system-architecture)
* [🌾 Agriculture Modules](#-agriculture-modules)
* [📂 Repository Structure](#-repository-structure)
* [🚀 Quickstart & Installation](#-quickstart--installation)
* [📊 Machine Learning Models](#-machine-learning-models)
* [🔬 Technical Pipeline](#-technical-pipeline)
* [🎯 Objectives](#-objectives)
* [🛣️ Roadmap & Improvements](#️-roadmap--improvements)
* [👩‍💻 Author & License](#-author--license)

---

## ✨ Features

* **🌾 Crop Recommendation:** Recommends suitable crops based on agricultural and environmental parameters.
* **📊 Crop Yield Prediction:** Predicts expected crop yield using machine learning techniques.
* **💧 Irrigation Recommendation:** Classifies irrigation requirements into Low, Medium, and High.
* **🧪 Fertilizer Recommendation:** Provides fertilizer recommendations based on crop and soil information.
* **🌦️ Weather Analysis:** Analyzes weather information to support agricultural decision-making.
* **🍃 Plant Disease Detection:** Identifies plant diseases from plant leaf images.
* **🤖 Agriculture AI Chatbot:** Provides AI-based assistance for agriculture-related questions.
* **📚 RAG Knowledge Assistant:** Provides agriculture-related knowledge using a retrieval-based approach.
* **📈 Farmer Dashboard:** Provides a centralized interface for accessing agricultural predictions and recommendations.

---

## 🧠 System Architecture

Smart Agriculture AI follows a modular machine learning architecture:

```text
┌────────────────────────────────────────────────────────┐
│                  Farmer / User Input                   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│              Data Preprocessing & Validation            │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                Agricultural AI Modules                  │
│                                                        │
│ Crop Recommendation │ Yield Prediction                 │
│ Irrigation          │ Fertilizer Recommendation        │
│ Disease Detection   │ Weather Analysis                 │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│              Prediction / Recommendation               │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│        AI Chatbot / Knowledge Assistant / Dashboard    │
└────────────────────────────────────────────────────────┘
```

---

## 🌾 Agriculture Modules

### 1. Crop Recommendation

Uses agricultural and environmental parameters to recommend a suitable crop for cultivation.

### 2. Crop Yield Prediction

Uses historical agricultural data and machine learning techniques to estimate expected crop yield.

### 3. Irrigation Recommendation

Analyzes relevant soil and environmental conditions and classifies irrigation requirements as:

* Low
* Medium
* High

### 4. Fertilizer Recommendation

Provides fertilizer recommendations using crop and soil-related information.

### 5. Weather Analysis

Analyzes weather conditions to provide useful agricultural insights and support farming decisions.

### 6. Plant Disease Detection

Uses plant leaf images and machine learning/deep learning techniques to identify possible plant diseases.

### 7. Agriculture AI Chatbot

Provides AI-based responses to agriculture-related questions and assists farmers with farming guidance.

### 8. RAG Knowledge Assistant

Provides agriculture-related information by retrieving relevant knowledge and generating useful responses.

### 9. Farmer Dashboard

Provides a centralized interface through which users can access the different agricultural AI modules.

---

## 📂 Repository Structure

```text
Smart-Agriculture-AI/
│
├── datasets/
│   ├── crop_recommendation.csv
│   ├── crop_yield.csv
│   ├── irrigation_dataset.csv
│   └── fertilizer_dataset.csv
│
├── models/
│   ├── crop_recommendation_model.pkl
│   ├── yield_model.pkl
│   ├── irrigation_model.pkl
│   └── fertilizer_model.pkl
│
├── notebooks/
│   ├── 01_crop_recommendation.ipynb
│   ├── 02_yield_prediction.ipynb
│   └── 03_disease_detection.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites

* Python 3.10+
* pip
* Git
* VS Code or Jupyter Notebook

### 2. Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Smart-Agriculture-AI
```

### 3. Create Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## 📊 Machine Learning Models

### Crop Recommendation

Uses agricultural parameters to recommend an appropriate crop.

### Crop Yield Prediction

Predicts expected agricultural yield using trained regression models.

### Irrigation Recommendation

Classifies irrigation requirements into Low, Medium, and High categories.

### Fertilizer Recommendation

Recommends suitable fertilizers based on available crop and soil information.

### Plant Disease Detection

Analyzes plant leaf images to identify possible diseases.

---

## 🔬 Technical Pipeline

### 1. Data Collection

Agricultural datasets are collected for crop, yield, irrigation, fertilizer, weather, and plant disease analysis.

### 2. Data Preprocessing

The data is cleaned, transformed, and prepared for machine learning.

### 3. Feature Engineering

Relevant agricultural and environmental features are selected and prepared for model training.

### 4. Model Training

Machine learning models are trained independently for different agricultural prediction and recommendation tasks.

### 5. Model Evaluation

Models are evaluated using appropriate metrics depending on whether the task is classification or regression.

### 6. Prediction

The trained models generate predictions or recommendations based on user input.

### 7. Application Integration

The trained models are integrated into the Streamlit-based application and farmer dashboard.

---

## 🎯 Objectives

* Help farmers make data-driven agricultural decisions.
* Recommend suitable crops based on agricultural conditions.
* Predict crop yield using machine learning.
* Improve irrigation and fertilizer management.
* Assist with plant disease identification.
* Provide weather-based agricultural insights.
* Provide AI-powered farming assistance.

---

## 🛣️ Roadmap & Improvements

* **IoT Integration:** Connect real-time soil and environmental sensors.
* **Real-Time Weather:** Integrate live weather and forecast information.
* **Multilingual Support:** Provide farmer assistance in regional languages.
* **Mobile Application:** Develop an Android/mobile version.
* **Advanced Disease Detection:** Improve image-based plant disease recognition.
* **Satellite Data:** Integrate satellite and remote-sensing information.
* **Personalized Recommendations:** Generate recommendations based on individual farm conditions.

---

## 👩‍💻 Author & License

Developed by **Athi Jahnavi**

* 🎓 B.Tech Computer Science & Information Technology
* 💻 AI & Machine Learning Project

### 📄 License

This project is developed for academic and educational purposes.
