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

Deployment:
<img width="1834" height="791" alt="Screenshot 2026-10-31 090118" src="https://github.com/user-attachments/assets/23f263b9-c506-4af5-b27f-2cef61f8c1e7" />
<img width="1827" height="719" alt="Screenshot 2026-10-31 090303" src="https://github.com/user-attachments/assets/17918fc3-2596-42ff-9567-130d290177d8" />
<img width="1826" height="799" alt="Screenshot 2026-10-31 090427" src="https://github.com/user-attachments/assets/e132b771-4b2b-48ae-8db8-583c025b177c" />
<img width="1494" height="546" alt="Screenshot 2026-10-31 090515" src="https://github.com/user-attachments/assets/abc91129-971d-4de3-b7a4-f5ebd805f8cd" />
<img width="1828" height="858" alt="Screenshot 2026-10-31 090650" src="https://github.com/user-attachments/assets/77c9ae65-085d-49e4-be68-b0b44b86091d" />
<img width="1832" height="857" alt="Screenshot 2026-10-31 130039" src="https://github.com/user-attachments/assets/bd22dfc0-0b29-41e2-83d5-302c15744099" />
<img width="1822" height="852" alt="Screenshot 2026-10-31 130952" src="https://github.com/user-attachments/assets/2f62eaf7-592b-4da9-8e7b-21c3feaebb80" />


👨‍💻 Developer

Mohamed Ayman - AI & Machine Learning Engineer | Database Developer

Developed as a technical blueprint for an AI-enabled rehabilitation monitoring system.
