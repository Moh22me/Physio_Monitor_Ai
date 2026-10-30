import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import joblib
import os
from streamlit_lottie import st_lottie

# 1. Page Configuration
st.set_page_config(page_title="Physio Monitor AI", page_icon="🫀", layout="wide")

# 2. Custom CSS
st.markdown("""
    <style>
    .glow-text {
        font-size: 38px; color: #00ffcc; text-align: center; font-weight: bold;
        text-shadow: 0 0 10px #00ffcc, 0 0 20px #00ffcc; margin-bottom: 20px;
    }
    .metric-card {
        background-color: #11151c; border-radius: 10px; padding: 20px;
        border: 1px solid #1f2937; text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #0d2b2a 0%, #11151c 100%);
        border: 2px solid #00ffcc; border-radius: 12px; padding: 20px; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Helper Functions
@st.cache_data
def load_data():
    file_path = "data/Synthetic_patient-HealthCare-Monitoring_dataset.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

@st.cache_resource
def load_ai_models():
    model_path = "models/physio_monitor_rf_optimized.pkl"
    scaler_path = "models/feature_scaler.pkl"
    target_path = "models/le_target.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(target_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le_target = joblib.load(target_path)
        return model, scaler, le_target
    return None, None, None

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def generate_ecg(heart_rate, duration=10, sampling=100):
    t = np.linspace(0, duration, duration * sampling)
    freq = heart_rate / 60.0
    wave = np.sin(2 * np.pi * freq * t) * 0.15
    qrs = np.zeros_like(t)
    step = int(sampling / max(freq, 0.1))
    for idx in np.arange(0, len(t), step if step > 0 else 100):
        if idx < len(t) - 5:
            qrs[idx] = 1.8; qrs[idx-2] = -0.5; qrs[idx+2] = -0.7
    return t, wave + qrs + np.random.normal(0, 0.03, len(t))

# 4. Load Data & Models
df = load_data()
model, scaler, le_target = load_ai_models()
lottie_heart = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_5n8kx1yq.json")

# Header
st.markdown("<div class='glow-text'>Physio Monitor AI System</div>", unsafe_allow_html=True)

if df is None:
    st.error("⚠️ Data file not found in 'data/' folder.")
    st.stop()

# 5. Build Application Tabs
tab1, tab2, tab3 = st.tabs(["🫀 Live Monitor", "🧪 New Patient Predictor", "📊 Data Analysis"])

# TAB 1: LIVE MONITORING
with tab1:
    with st.sidebar:
        st.header("⚙️ Select Patient")
        disease_filter = st.selectbox("Condition Filter", df['Predicted Disease'].unique())
        filtered_df = df[df['Predicted Disease'] == disease_filter]
        selected_id = st.selectbox("Patient ID", filtered_df['Patient Number'].tolist())
        patient = filtered_df[filtered_df['Patient Number'] == selected_id].iloc[0]

    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Vitals")
        st.metric("Heart Rate", f"{patient['Heart Rate (bpm)']} bpm", patient['Heart Rate Alert'], delta_color="inverse")
        st.metric("SpO2 Level", f"{patient['SpO2 Level (%)']}%", patient['SpO2 Level Alert'], delta_color="inverse")
        st.metric("Body Temp", f"{patient['Body Temperature (°C)']} °C", patient['Temperature Alert'], delta_color="inverse")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if lottie_heart:
            st_lottie(lottie_heart, height=280, key="heart_monitor")
        st.markdown(f"<h3 style='text-align: center;'>Current Diagnosis: <span style='color:#00ffcc;'>{patient['Predicted Disease']}</span></h3>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Hemodynamics")
        st.metric("Systolic BP", f"{patient['Systolic Blood Pressure (mmHg)']} mmHg")
        st.metric("Diastolic BP", f"{patient['Diastolic Blood Pressure (mmHg)']} mmHg")
        st.metric("Fall Detected", patient['Fall Detection'])
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Real-Time ECG Waveform")
    t_wave, ecg_wave = generate_ecg(patient['Heart Rate (bpm)'])
    fig = go.Figure(go.Scatter(x=t_wave, y=ecg_wave, mode='lines', line=dict(color='#00ffcc', width=2)))
    fig.update_layout(
        plot_bgcolor='#050505', paper_bgcolor='#0e1117',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='#222', zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=10, b=0), height=280
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: LIVE AI PREDICTOR
with tab2:
    st.subheader("🧪 Real-Time Disease Prediction Engine")
    st.write("Enter custom patient vital signs to run real-time inference using the Random Forest model.")
    
    if model is None:
        st.warning("⚠️ ML Model files missing in 'models/' folder. Please copy .pkl files to enable real-time prediction.")
    else:
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            hr_in = st.slider("Heart Rate (bpm)", 40, 180, 85)
            spo2_in = st.slider("SpO2 Level (%)", 70, 100, 97)
            sys_bp_in = st.slider("Systolic BP (mmHg)", 80, 200, 120)
            dia_bp_in = st.slider("Diastolic BP (mmHg)", 50, 130, 80)
            
        with col_in2:
            temp_in = st.slider("Body Temperature (°C)", 35.0, 41.0, 37.0)
            fall_in = st.selectbox("Fall Event Detected?", ["No", "Yes"])
            accuracy_in = st.slider("Sensor Accuracy (%)", 80, 100, 95)

        if st.button("🚀 Run AI Diagnosis", use_container_width=True):
            # Calculate dynamic alerts
            hr_alert = 1 if (60 <= hr_in <= 100) else 0
            spo2_alert = 1 if spo2_in >= 95 else 0
            bp_alert = 1 if (sys_bp_in <= 120 and dia_bp_in <= 80) else 0
            temp_alert = 1 if (36.5 <= temp_in <= 37.5) else 0
            fall_encoded = 1 if fall_in == "Yes" else 0
            
            # Prepare feature array
            input_features = [
                hr_in, spo2_in, sys_bp_in, dia_bp_in, temp_in,
                fall_encoded, accuracy_in, hr_alert, spo2_alert, bp_alert, temp_alert
            ]
            
            # Prediction
            scaled_input = scaler.transform([input_features])
            pred_class = model.predict(scaled_input)[0]
            probs = model.predict_proba(scaled_input)[0]
            confidence = np.max(probs) * 100
            disease_name = le_target.inverse_transform([pred_class])[0]
            
            # --- MEDICAL PROFESSIONALISM LOGIC ---
            if confidence >= 80:
                box_color = "#00ffcc" # Green/Cyan
                medical_msg = f"<div style='color: {box_color}; margin-top: 15px; font-size: 18px;'>✅ <b>التشخيص قوي:</b> القراءات تتطابق بشكل موثوق مع الأنماط الحيوية المسجلة.</div>"
            elif confidence >= 60:
                box_color = "#ffcc00" # Yellow
                medical_msg = f"<div style='color: {box_color}; margin-top: 15px; font-size: 18px;'>ℹ️ <b>توقع مقبول:</b> توجد بعض التداخلات في القراءات، يفضل وضع المريض تحت الملاحظة.</div>"
            else:
                box_color = "#ff4444" # Red
                medical_msg = f"<div style='color: {box_color}; margin-top: 15px; font-size: 18px;'>⚠️ <b>يرجى استشارة الطبيب فوراً:</b> التوقع غير حاسم لوجود تضارب شديد في المؤشرات الحيوية!</div>"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class='prediction-box' style='border-color: {box_color}; box-shadow: 0 0 15px {box_color};'>
                    <h2>Predicted Diagnosis: <span style='color:{box_color};'>{disease_name}</span></h2>
                    <h4>Model Confidence: <span style='color:{box_color};'>{confidence:.2f}%</span></h4>
                    {medical_msg}
                </div>
            """, unsafe_allow_html=True)

# TAB 3: DATA ANALYSIS
with tab3:
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(100), use_container_width=True)
    c_a, c_b = st.columns(2)
    with c_a:
        st.write("Disease Distribution")
        st.bar_chart(df['Predicted Disease'].value_counts())
    with c_b:
        st.write("Fall Detection Events")
        st.bar_chart(df['Fall Detection'].value_counts())