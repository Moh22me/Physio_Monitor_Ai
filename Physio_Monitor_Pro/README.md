Physio Monitor AI System

📌 About The Project

Physio Monitor AI is a comprehensive Clinical Decision Support System (CDSS) and real-time vital signs monitoring dashboard. Built for healthcare professionals and physical therapy rehabilitation, this system leverages Machine Learning to analyze patient hemodynamics and predict potential diseases with a confidence score.

✨ Key Features

🩺 Live Patient Monitor: Real-time visualization of vital parameters (Heart Rate, SpO2, Blood Pressure, Body Temperature).

📈 Dynamic ECG Simulation: Live Electrocardiogram (ECG) waveform generation based on the patient's actual heart rate using Plotly.

🧪 AI Disease Predictor: A robust Machine Learning engine (Random Forest) hyperparameter-tuned to predict health conditions based on custom clinical inputs.

🚦 Clinical Confidence System: Intelligent alerting mechanism that evaluates the AI's confidence score (Red/Yellow/Green) to recommend physician intervention.

📊 Data Analysis: Interactive dataset overview and condition distribution charts.

✨ Modern UI: Neon/X-ray aesthetics with Lottie animations for an immersive medical monitoring experience.

🛠️ Tech Stack

Frontend & Dashboard: Streamlit

Machine Learning: Scikit-Learn (RandomForestClassifier, GridSearchCV)

Data Manipulation: Pandas, NumPy

Data Visualization: Plotly

Model Persistence: Joblib

📁 Project Structure

Physio_Monitor_Pro/
│
├── models/                              # Generated ML models and scalers
│   ├── physio_monitor_rf_optimized.pkl
│   ├── feature_scaler.pkl
│   └── le_target.pkl
│
├── data/                                # Dataset folder
│   └── Synthetic_patient-HealthCare-Monitoring_dataset.csv
│
├── app.py                               # Main Streamlit Dashboard Application
├── train.py                             # ML Training and artifact generation script
├── requirements.txt                     # Project dependencies
└── README.md                            # Project documentation


🚀 Getting Started

Prerequisites

Make sure you have Python 3.9+ installed on your system.

Installation & Setup

Clone the repository

git clone https://github.com/Moh22me/Physio_Monitor_Ai.git
cd Physio_Monitor_Ai


Create and activate a virtual environment

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate


Install the dependencies

pip install -r requirements.txt


Train the AI Model (Generate .pkl files)

python train.py


Run the Application

streamlit run app.py


The application will automatically open in your browser at http://localhost:8501.

🧠 Model Training Details

The prediction engine uses a Random Forest Classifier. The dataset undergoes preprocessing where missing values are handled, categorical variables (like Fall Detection and Alerts) are encoded using LabelEncoder, and numerical features are scaled using StandardScaler. The final model achieves a high accuracy rate (approx. 95.7%) on the synthetic healthcare dataset.

👨‍💻 Developer

Mohamed Ayman - AI & Machine Learning Engineer | Database Developer

Developed as a technical blueprint for an AI-enabled rehabilitation monitoring system.