# 🌱 Smart Agriculture AI

> An AI-powered agriculture application for detecting plant diseases from leaf images and providing actionable treatment and preventive guidance.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)](https://opencv.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/)

---

## 🚀 Live Demo

### 🌐 Try the Application

👉 **[Smart Agriculture AI – Live Demo](https://smart-agriculture-ai-n4x5u3fdh6s2fys5vify5p.streamlit.app/)**

The application allows users to upload a plant leaf image and receive an AI-based disease classification result along with confidence information and recommended preventive/treatment guidance.

---

## 📌 Project Overview

Agriculture plays an important role in food production and the economy. Plant diseases can significantly affect crop productivity when they are not identified at an early stage.

**Smart Agriculture AI** is an AI-powered application designed to assist farmers and agricultural users in identifying plant diseases from leaf images.

The system uses **Convolutional Neural Networks (CNNs)** and **transfer learning** to analyze uploaded leaf images and classify them into supported disease categories.

The application provides a simple **Streamlit-based web interface**, making AI-based plant disease detection accessible through a browser.

---

## 🎯 Problem Statement

Farmers often struggle to identify crop diseases early and accurately, leading to crop loss and reduced productivity.

---

## 💡 Proposed Solution

Smart Agriculture AI provides an automated image-based plant disease detection system.

The user uploads a leaf image, which is:

1. Preprocessed
2. Analyzed using a trained deep learning model
3. Classified into a disease category
4. Assigned a confidence score
5. Used to provide treatment and preventive guidance

---

## ✨ Key Features

- 🌿 **Plant Disease Detection**
- 📷 **Leaf Image Upload**
- 🤖 **CNN-based Image Classification**
- 🧠 **Transfer Learning**
- 📊 **Disease Classification with Confidence Score**
- 🌾 **Multi-Crop Disease Support**
- 💊 **Treatment Recommendations**
- 🛡️ **Preventive Measures**
- 👨‍🌾 **Farmer-Friendly Interface**
- ⚡ **Fast Image Analysis**
- 🌐 **Web-Based Streamlit Application**
- ☁️ **Cloud Deployment**

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │       User          │
                    │     / Farmer        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Upload Leaf Image │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Image Preprocess  │
                    │ Resize / Normalize  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   CNN / Transfer    │
                    │      Learning       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Disease Prediction  │
                    │   + Confidence      │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
      ┌──────────────────┐             ┌──────────────────┐
      │ Treatment Advice │             │ Preventive Tips  │
      └──────────────────┘             └──────────────────┘
🔄 Project Workflow
Leaf Image
    ↓
Image Upload
    ↓
Image Preprocessing
    ↓
Feature Extraction
    ↓
CNN Model
    ↓
Disease Classification
    ↓
Confidence Score
    ↓
Treatment & Preventive Guidance
    ↓
Display Result
🧠 Machine Learning Methodology

The project uses deep learning-based image classification.

1. Image Collection

Plant leaf images are used as input data for training and validation.

2. Image Preprocessing

The input images undergo preprocessing operations such as:

Image resizing
Pixel normalization
Data augmentation
RGB image processing
3. Feature Extraction

The CNN learns important visual characteristics from leaf images, including patterns associated with plant diseases.

4. Transfer Learning

Pre-trained deep learning architectures such as ResNet50 / VGG16 can be used as feature extraction backbones.

5. Classification

The extracted features are passed through dense layers and a Softmax classification layer to identify the disease category.

6. Prediction

The system generates:

Predicted disease
Confidence score
Recommended action
📊 Dataset

The project uses the PlantVillage dataset for plant disease classification.

The project documentation specifies:

Attribute	Details
Dataset	PlantVillage
Images	54,306
Crop Species	14
Diseases	26
Split	80% Training / 20% Validation
Sampling	Stratified

The project presentation reports classification performance of 99%+ accuracy for the developed model.

Note: Reported accuracy depends on the dataset, preprocessing, model configuration, and evaluation procedure.

🛠️ Technologies Used
Programming Language
Python
Machine Learning / Deep Learning
TensorFlow
Keras
Scikit-learn
Computer Vision
OpenCV
Pillow
NumPy
Data Processing
Pandas
NumPy
Web Application
Streamlit
Database
SQLite
Development Tools
Visual Studio Code
Jupyter Notebook
Version Control
Git
GitHub
Deployment
Streamlit Community Cloud
📂 Project Structure
Smart-Agriculture-AI/
│
├── app.py
│
├── data/
│   ├── crop_recommendation.csv
│   └── crop_yield.csv
│
├── models/
│   └── plant_disease_mobilenetv2_38_classes.keras
│
├── notebook/
│   ├── 01_crop_recommendation.ipynb
│   ├── 02_yield_prediction.ipynb
│   └── 03_disease_detection.ipynb
│
├── utils/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore

Update the folder names above if your GitHub repository uses a different final structure.

🖥️ Application Modules
🌱 Crop Recommendation

Recommends suitable crops based on agricultural input parameters.

📈 Crop Yield Prediction

Uses agricultural information to estimate crop yield.

🍃 Plant Disease Detection

Analyzes leaf images using a deep learning model and predicts the corresponding disease category.

💧 Irrigation Recommendation

Provides irrigation recommendations based on relevant agricultural and environmental parameters.

🧪 Fertilizer Recommendation

Provides fertilizer recommendations according to agricultural input conditions.

🌦️ Weather Analysis

Provides weather-related information and agricultural advisory support.

🤖 Agriculture AI Chatbot

Provides an AI-based conversational interface for agriculture-related questions.

📸 How to Use
Step 1 — Open the Application

Visit the live application:

Smart Agriculture AI

Step 2 — Upload an Image

Select a clear image of a plant leaf.

Step 3 — Image Processing

The application processes the uploaded image before sending it to the trained model.

Step 4 — View Prediction

The model displays the predicted disease category and confidence information.

Step 5 — Follow Recommendations

Use the displayed treatment and preventive guidance as an initial agricultural reference.

⚙️ Installation
1. Clone the Repository
git clone https://github.com/jahnaviathi7/Smart-Agriculture-AI.git
2. Navigate to the Project
cd Smart-Agriculture-AI
3. Create a Virtual Environment
python -m venv venv
4. Activate the Environment
Windows
venv\Scripts\activate
5. Install Dependencies
pip install -r requirements.txt
6. Run the Application
streamlit run app.py

The application will open in your browser.

📦 Requirements

Example requirements.txt:

streamlit
tensorflow
numpy
pandas
opencv-python-headless
Pillow
scikit-learn
joblib

Make sure the requirements match the packages imported by the final application before deployment.

☁️ Deployment

The application is deployed using Streamlit Community Cloud.

Deployment Flow
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Application Build
       ↓
Dependency Installation
       ↓
Model Loading
       ↓
Public Web Application
Live Application

👉 Open Smart Agriculture AI

🧪 Testing

The application can be tested using different plant leaf images.

Testing Areas
Test Case	Expected Result
Upload valid leaf image	Image is accepted
Upload supported disease image	Disease is predicted
View prediction	Disease name is displayed
Check confidence	Confidence information is displayed
View recommendations	Guidance is displayed
Access application through browser	Application loads successfully
📈 Expected Outcomes

The system aims to:

Detect plant diseases automatically.
Reduce dependence on manual visual inspection.
Support early identification of crop diseases.
Provide disease classification results.
Provide treatment and preventive guidance.
Make AI-based agricultural assistance accessible through a web interface.
✅ Advantages
Automated disease detection
Reduces manual inspection effort
Fast image-based analysis
Supports multiple crop disease categories
Easy-to-use interface
Accessible through a web browser
Can be deployed as a cloud application
Provides actionable agricultural guidance
⚠️ Limitations
Prediction quality depends on image quality.
The model can only classify diseases represented in its training data.
Real-world field images may differ from controlled dataset images.
Internet connectivity may be required for the deployed application.
AI predictions should be treated as decision-support information rather than a replacement for professional agricultural diagnosis.
🔮 Future Scope

Future improvements can include:

📱 Android/iOS mobile application
📷 Real-time camera-based disease detection
🌍 GPS-based agricultural recommendations
🌦️ Integration with real-time weather APIs
🛰️ Satellite and drone-based crop monitoring
📊 Advanced farmer analytics dashboard
🌱 IoT-based soil monitoring
💧 Automated smart irrigation
🗣️ Regional-language voice assistant
🤖 Advanced AI agricultural assistant
📚 RAG-based agricultural knowledge system
🔄 Continuous model improvement using new field images
🎓 Academic Project Information

Project Title:
Smart Agriculture AI

Domain:
Artificial Intelligence & Machine Learning

Technologies:
Python, TensorFlow, CNN, Transfer Learning, OpenCV, Streamlit, SQLite

Application Type:
AI-powered Web Application

Deployment:
Streamlit Community Cloud


📚 References
PlantVillage Dataset
TensorFlow / Keras Documentation
OpenCV Documentation
Scikit-learn Documentation
Streamlit Documentation
🌟 Project Status
Project: Smart Agriculture AI
Status: Deployed
Platform: Streamlit
Domain: Artificial Intelligence & Agriculture



