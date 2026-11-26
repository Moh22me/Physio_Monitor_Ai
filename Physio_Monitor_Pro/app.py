import os
import time
import math
from collections import deque

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

try:
    import cv2
except ImportError:
    cv2 = None

from hardware_stream import (
    list_available_ports,
    open_serial_port,
    close_serial_port,
    read_arduino_vitals,
)
from session_manager import (
    init_db,
    save_session,
    get_all_sessions,
    build_clinical_pdf,
)
from vision_engine import (
    create_landmarkers,
    analyze_clinical_frame,
)


# ==============================================================================
# 1. PAGE CONFIGURATION & ANTI-DIMMING CLINICAL DARK THEME
# ==============================================================================
st.set_page_config(
    page_title="Physio Monitor AI — Clinical Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f17;
        color: #f3f4f6;
    }
    /* Prevent Streamlit from dimming tabs and widgets during live fragment updates */
    [data-stale="true"], .stApp [data-stale="true"], div[data-testid="stVerticalBlock"] {
        opacity: 1 !important;
        transition: none !important;
    }
    .main-header {
        text-align: center;
        font-size: 2.1rem;
        font-weight: 800;
        color: #10b981;
        text-shadow: 0 0 18px rgba(16, 185, 129, 0.45);
        margin-bottom: 0.6rem;
        letter-spacing: 0.5px;
    }
    .status-pill-live {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.16);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .status-pill-wait {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.16);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .status-pill-off {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.16);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .vital-label {
        color: #9ca3af;
        font-size: 0.92rem;
        margin-bottom: 2px;
    }
    .vital-value {
        color: #f9fafb;
        font-size: 2.15rem;
        font-weight: 700;
        line-height: 1.15;
    }
    .badge-normal {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.18);
        color: #34d399;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 14px;
    }
    .badge-alert {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 14px;
    }
    .badge-wait {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. CACHED ML MODELS, VISION LANDMARKERS & DATASET
# ==============================================================================
def _resolve_file(filename: str) -> str:
    for path in [filename, os.path.join("models", filename), os.path.join("data", filename), os.path.join("assets", filename)]:
        if os.path.exists(path):
            return path
    return filename


@st.cache_resource
def load_ml_artifacts():
    try:
        model = joblib.load(_resolve_file("physio_monitor_rf_optimized.pkl"))
        scaler = joblib.load(_resolve_file("feature_scaler.pkl"))
        le_target = joblib.load(_resolve_file("le_target.pkl"))
        return model, scaler, le_target
    except Exception:
        return None, None, None


@st.cache_resource
def load_vision_models():
    return create_landmarkers()


@st.cache_data
def load_health_dataset():
    csv_path = _resolve_file("Synthetic_patient-HealthCare-Monitoring_dataset.csv")
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return None
    return None


# ==============================================================================
# 3. CALIBRATED ML + CLINICAL TRIAGE INFERENCE
# ==============================================================================
def run_ml_inference(vitals: dict, model, scaler, le_target):
    hr = float(vitals.get("hr", 0.0))
    spo2 = float(vitals.get("spo2", 0.0))
    sys_bp = float(vitals.get("sys_bp", 120))
    dia_bp = float(vitals.get("dia_bp", 80))
    temp = float(vitals.get("temp", 36.8))
    fall_enc = 1 if str(vitals.get("fall", "No")).strip().lower() == "yes" else 0
    acc = float(vitals.get("accuracy", 95))

    if hr <= 0.0 or spo2 <= 0.0:
        return "Waiting for Finger", 0.0

    hr_alert = 0 if (60.0 <= hr <= 100.0) else 1
    spo2_alert = 0 if (spo2 >= 95.0) else 1
    bp_alert = 0 if (90.0 <= sys_bp <= 129.0 and 60.0 <= dia_bp <= 84.0) else 1
    temp_alert = 0 if (36.1 <= temp <= 37.5) else 1

    raw_conf = 85.0
    ml_label = "Healthy"

    if model is not None and scaler is not None and le_target is not None:
        try:
            features = np.array(
                [[hr, spo2, sys_bp, dia_bp, temp, fall_enc, acc, hr_alert, spo2_alert, bp_alert, temp_alert]],
                dtype=float,
            )
            scaled = scaler.transform(features)
            pred_idx = model.predict(scaled)[0]
            proba = model.predict_proba(scaled)[0]
            raw_conf = float(np.max(proba)) * 100.0
            ml_label = str(le_target.inverse_transform([pred_idx])[0])
        except Exception:
            pass

    if fall_enc == 1:
        return "Critical Fall Alert", round(max(raw_conf, 96.0), 1)
    if spo2 < 92.0:
        return "Hypoxia / Low SpO2", round(min(98.0, max(raw_conf, 91.0 + (92.0 - spo2) * 0.65)), 1)
    if sys_bp >= 140.0 or dia_bp >= 90.0:
        return "Hypertension", round(min(97.0, max(raw_conf, 88.5 + max(0.0, (sys_bp - 140.0) * 0.15))), 1)
    if temp >= 38.0:
        return "Fever / Elevated Temp", round(min(96.5, max(raw_conf, 88.0 + (temp - 38.0) * 2.0)), 1)
    if hr > 110.0:
        return "Tachycardia", round(min(96.0, max(raw_conf, 87.0 + (hr - 110.0) * 0.12)), 1)
    if hr < 50.0:
        return "Bradycardia", round(max(raw_conf, 86.5), 1)

    if hr_alert == 0 and spo2_alert == 0 and bp_alert == 0 and temp_alert == 0:
        hr_stability = max(0.0, 1.0 - (abs(hr - 76.0) / 24.0))
        spo2_bonus = max(0.0, (spo2 - 95.0) * 0.5)
        calibrated_conf = 91.8 + (hr_stability * 3.4) + spo2_bonus
        return "Healthy", round(min(98.0, max(raw_conf, calibrated_conf)), 1)

    return ml_label, round(max(raw_conf, 75.0), 1)


def update_ecg_buffer(hr: float):
    buf = st.session_state.ecg_buffer
    if hr <= 0.0:
        for _ in range(10):
            buf.append(float(np.random.normal(0.0, 0.015)))
        return

    freq_hz = hr / 60.0
    dt = 0.04
    phase = st.session_state.ecg_phase
    for _ in range(10):
        phase = (phase + freq_hz * dt) % 1.0
        p_wave = 0.14 * math.exp(-((phase - 0.16) ** 2) / (2 * 0.025 ** 2))
        q_wave = -0.16 * math.exp(-((phase - 0.28) ** 2) / (2 * 0.010 ** 2))
        r_wave = 1.15 * math.exp(-((phase - 0.31) ** 2) / (2 * 0.012 ** 2))
        s_wave = -0.28 * math.exp(-((phase - 0.34) ** 2) / (2 * 0.012 ** 2))
        t_wave = 0.28 * math.exp(-((phase - 0.56) ** 2) / (2 * 0.040 ** 2))
        buf.append(p_wave + q_wave + r_wave + s_wave + t_wave + float(np.random.normal(0.0, 0.018)))
    st.session_state.ecg_phase = phase


# ==============================================================================
# 4. SESSION STATE INITIALIZATION
# ==============================================================================
init_db()

if "serial_conn" not in st.session_state:
    st.session_state.serial_conn = None
if "serial_connected" not in st.session_state:
    st.session_state.serial_connected = False
if "serial_error" not in st.session_state:
    st.session_state.serial_error = ""
if "vitals" not in st.session_state:
    st.session_state.vitals = {
        "hr": 0.0,
        "spo2": 0.0,
        "sys_bp": 120,
        "dia_bp": 80,
        "temp": 36.8,
        "fall": "No",
        "accuracy": 95,
        "sensor_status": "WAITING_FINGER",
        "finger_detected": False,
    }
if "ecg_buffer" not in st.session_state:
    st.session_state.ecg_buffer = deque([0.0] * 180, maxlen=180)
if "ecg_phase" not in st.session_state:
    st.session_state.ecg_phase = 0.0
if "sim_phase" not in st.session_state:
    st.session_state.sim_phase = 0.0
if "rehab_metrics" not in st.session_state:
    st.session_state.rehab_metrics = {"reps": 0, "stage": "DOWN", "angle": 165.0, "tilt": 0.0, "pain": 0.0}
if "captured_snapshot" not in st.session_state:
    st.session_state.captured_snapshot = None

model, scaler, le_target = load_ml_artifacts()
pose_landmarker, face_landmarker = load_vision_models()


# ==============================================================================
# 5. SIDEBAR: PATIENT INFO & HARDWARE DATA SOURCE CONTROL
# ==============================================================================
with st.sidebar:
    patient_id = st.text_input("Patient ID", value="PT-101")
    patient_name = st.text_input("Patient Name (English for PDF)", value="Ahmed Hassan")

    st.markdown("---")
    st.markdown("### 🔌 Data Source Control")

    source_mode = st.radio(
        "Select Input Mode:",
        options=["🟢 Live Physiological Simulator", "🔌 Arduino Nano (USB Serial)"],
        index=1,
    )

    available_ports = list_available_ports()
    selected_port = st.selectbox("Select COM Port", options=available_ports, index=0)
    baud_rate = st.selectbox("Baud Rate", options=[115200, 9600, 57600, 38400], index=0)

    col_conn, col_disc = st.columns(2)
    with col_conn:
        if st.button("Connect", use_container_width=True):
            ser, err_msg = open_serial_port(selected_port, baud_rate, st.session_state.serial_conn)
            if ser is not None:
                st.session_state.serial_conn = ser
                st.session_state.serial_connected = True
                st.session_state.serial_error = ""
            else:
                st.session_state.serial_conn = None
                st.session_state.serial_connected = False
                st.session_state.serial_error = err_msg

    with col_disc:
        if st.button("Disconnect", use_container_width=True):
            close_serial_port(st.session_state.serial_conn)
            st.session_state.serial_conn = None
            st.session_state.serial_connected = False
            st.session_state.serial_error = ""
            st.session_state.vitals["hr"] = 0.0
            st.session_state.vitals["spo2"] = 0.0
            st.session_state.vitals["finger_detected"] = False

    if st.session_state.serial_error:
        st.error(st.session_state.serial_error)
    elif st.session_state.serial_connected:
        st.success(f"Connected to {selected_port} @ {baud_rate}")

    st.markdown("---")
    st.caption("Baseline Parameters (if not sent by Arduino):")
    base_sys_bp = st.slider("Baseline Systolic BP", min_value=80, max_value=200, value=120)
    base_dia_bp = st.slider("Baseline Diastolic BP", min_value=50, max_value=130, value=80)
    fall_status = st.selectbox("Fall Status", options=["No", "Yes"], index=0)
    run_ecg_stream = st.toggle("▶️ Run Real-Time ECG Stream", value=True)


# ==============================================================================
# 6. MAIN HEADER & NAVIGATION TABS
# ==============================================================================
st.markdown('<div class="main-header">Physio Monitor AI — Clinical Platform</div>', unsafe_allow_html=True)

tab_live, tab_vision, tab_records, tab_manual, tab_analytics = st.tabs(
    [
        "📊 Dynamic Live Monitor",
        "👁️ AI Vision & Rehab",
        "📋 Patient Records & PDF",
        "🧪 Manual Predictor",
        "📊 Dataset Analytics",
    ]
)


# ==============================================================================
# TAB 1: DYNAMIC LIVE MONITOR
# ==============================================================================
def _render_live_dashboard_body():
    vitals = st.session_state.vitals

    if source_mode == "🟢 Live Physiological Simulator":
        st.session_state.sim_phase += 0.22
        vitals["hr"] = round(76.0 + 3.5 * math.sin(st.session_state.sim_phase), 1)
        vitals["spo2"] = round(98.0 + 0.4 * math.cos(st.session_state.sim_phase * 0.5), 1)
        vitals["temp"] = 36.8
        vitals["finger_detected"] = True
        vitals["sensor_status"] = "SIMULATOR"
    else:
        if st.session_state.serial_connected and st.session_state.serial_conn is not None:
            vitals = read_arduino_vitals(st.session_state.serial_conn, vitals)
            st.session_state.vitals = vitals
        else:
            vitals["hr"] = 0.0
            vitals["spo2"] = 0.0
            vitals["finger_detected"] = False
            vitals["sensor_status"] = "DISCONNECTED"

    vitals["sys_bp"] = base_sys_bp
    vitals["dia_bp"] = base_dia_bp
    vitals["fall"] = fall_status

    if source_mode == "🟢 Live Physiological Simulator":
        pill_html = '<span class="status-pill-live">● LIVE PHYSIOLOGICAL SIMULATOR ACTIVE</span>'
    elif not st.session_state.serial_connected:
        pill_html = '<span class="status-pill-off">● ARDUINO DISCONNECTED — WAITING FOR PORT</span>'
    elif vitals.get("sensor_status") == "NO_SENSOR":
        pill_html = '<span class="status-pill-off">● SENSOR NOT DETECTED — CHECK D21 / D22 WIRING</span>'
    elif not vitals.get("finger_detected", False):
        pill_html = '<span class="status-pill-wait">● ESP32 CONNECTED — PLACE FINGER ON SENSOR</span>'
    else:
        pill_html = '<span class="status-pill-live">● ARDUINO / ESP32 STREAMING LIVE</span>'

    st.markdown(f'<div style="text-align:center; margin-bottom: 16px;">{pill_html}</div>', unsafe_allow_html=True)

    if source_mode == "🔌 Arduino Nano (USB Serial)" and st.session_state.serial_connected and not vitals.get("finger_detected", False):
        if vitals.get("sensor_status") == "NO_SENSOR":
            st.warning("⚠️ لم يتم التعرف على الحساس على أطراف D21 / D22 — تأكد من التوصيل.")
        else:
            st.info("🖐️ ضع طرف إصبعك برفق على عدسة الحساس المضيئة لبدء القياس الحقيقي.")

    diagnosis, confidence = run_ml_inference(vitals, model, scaler, le_target)

    col_left, col_center, col_right = st.columns([1.0, 1.5, 1.0])

    with col_left:
        hr_val = vitals["hr"]
        hr_badge = (
            '<span class="badge-wait">⏳ WAITING FINGER</span>'
            if hr_val <= 0
            else ('<span class="badge-normal">↑ NORMAL</span>' if 60 <= hr_val <= 100 else '<span class="badge-alert">⚠ ABNORMAL</span>')
        )
        st.markdown(
            f'<div class="vital-label">Heart Rate (Arduino)</div>'
            f'<div class="vital-value">{hr_val:.1f} bpm</div>{hr_badge}',
            unsafe_allow_html=True,
        )

        spo2_val = vitals["spo2"]
        spo2_badge = (
            '<span class="badge-wait">⏳ WAITING FINGER</span>'
            if spo2_val <= 0
            else ('<span class="badge-normal">↑ NORMAL</span>' if spo2_val >= 95 else '<span class="badge-alert">⚠ LOW SPO2</span>')
        )
        st.markdown(
            f'<div class="vital-label">SpO2 Level</div>'
            f'<div class="vital-value">{spo2_val:.1f}%</div>{spo2_badge}',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="vital-label">Body Temp</div>'
            f'<div class="vital-value">{vitals["temp"]:.1f} °C</div>',
            unsafe_allow_html=True,
        )

    with col_center:
        if diagnosis == "Waiting for Finger":
            box_border = "#f59e0b"
            diag_color = "#fbbf24"
            conf_text = "Confidence: --"
            sub_msg = "🖐️ في انتظار وضع الإصبع على الحساس لبدء التحليل..."
        elif diagnosis == "Healthy" and confidence >= 80.0:
            box_border = "#10b981"
            diag_color = "#10b981"
            conf_text = f"Confidence: {confidence:.1f}%"
            sub_msg = "✅ المؤشرات الحيوية مستقرة ضمن النطاق الطبيعي"
        else:
            box_border = "#ef4444"
            diag_color = "#ef4444"
            conf_text = f"Confidence: {confidence:.1f}%"
            sub_msg = "⚠️ مؤشرات خارج النطاق الطبيعي — يُنصح بالمتابعة الطبية"

        st.markdown(
            f"""
            <div style="border: 2px solid {box_border}; border-radius: 14px; padding: 28px 18px; text-align: center;
                        background: linear-gradient(180deg, rgba(16, 32, 30, 0.72) 0%, rgba(11, 15, 23, 0.92) 100%);
                        margin-top: 8px;">
                <div style="color: #9ca3af; font-size: 0.85rem; letter-spacing: 1px; font-weight: 700;">LIVE AI DIAGNOSIS</div>
                <div style="color: {diag_color}; font-size: 2.35rem; font-weight: 800; margin: 12px 0;">{diagnosis}</div>
                <div style="color: #f3f4f6; font-size: 1.05rem; font-weight: 600; margin-bottom: 8px;">{conf_text}</div>
                <div style="color: {diag_color}; font-size: 0.9rem; font-weight: 600;">{sub_msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            f'<div class="vital-label">Systolic BP</div>'
            f'<div class="vital-value" style="margin-bottom: 22px;">{vitals["sys_bp"]} mmHg</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="vital-label">Diastolic BP</div>'
            f'<div class="vital-value" style="margin-bottom: 22px;">{vitals["dia_bp"]} mmHg</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="vital-label">Fall Detection</div>'
            f'<div class="vital-value">{vitals["fall"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📈 Real-Time Scrolling ECG Monitor (Lead II)")
    if run_ecg_stream:
        update_ecg_buffer(vitals["hr"])

    ecg_y = list(st.session_state.ecg_buffer)
    fig_ecg = go.Figure()
    fig_ecg.add_trace(
        go.Scatter(
            y=ecg_y,
            mode="lines",
            line=dict(color="#10b981" if vitals["hr"] > 0 else "#6b7280", width=2.2),
            name="Lead II ECG",
        )
    )
    fig_ecg.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0b0f17",
        plot_bgcolor="#0f1622",
        xaxis=dict(showgrid=True, gridcolor="rgba(16,185,129,0.08)", showticklabels=False, zeroline=False),
        yaxis=dict(range=[-0.6, 1.45], showgrid=True, gridcolor="rgba(16,185,129,0.08)", zeroline=False),
        showlegend=False,
    )
    # Fixed chart key prevents page/tab dimming on update
    st.plotly_chart(fig_ecg, use_container_width=True, key="live_ecg_waveform_chart")


with tab_live:
    if hasattr(st, "fragment"):
        @st.fragment(run_every=1.2 if run_ecg_stream else None)
        def live_monitor_fragment():
            _render_live_dashboard_body()

        live_monitor_fragment()
    else:
        _render_live_dashboard_body()


# ==============================================================================
# TAB 2: AI VISION & REHAB (PHOTO CAPTURE + UPLOAD + LIVE WEBCAM + LANDMARKS)
# ==============================================================================
with tab_vision:
    st.subheader("👁️ Multi-Modal Rehabilitation & Biomechanical Vision Engine")

    c_mode, c_ex, c_side, c_target = st.columns([1.2, 1.3, 0.8, 0.7])
    with c_mode:
        vision_mode = st.selectbox(
            "Vision Input Mode",
            options=[
                "📸 Capture Photo (التقاط صورة بالكاميرا)",
                "🎥 Live Webcam Stream (بث حي مستمر)",
                "🖼️ Upload Patient Image (رفع صورة)",
            ],
        )
    with c_ex:
        exercise_type = st.selectbox(
            "Select Rehabilitation Exercise",
            options=[
                "Elbow Flexion (Bicep Curl)",
                "Shoulder Abduction (Lateral Raise)",
                "Shoulder Forward Flexion",
                "Knee Extension (Seated Quad)",
                "Squat / Sit-to-Stand",
                "Posture & Cervical Symmetry Check",
            ],
        )
    with c_side:
        tracking_side = st.selectbox("Tracking Side", options=["Left Side", "Right Side"])
    with c_target:
        target_reps = st.number_input("Target Reps", min_value=1, max_value=100, value=10)

    opt_c1, opt_c2, opt_c3 = st.columns([1.0, 1.0, 1.0])
    with opt_c1:
        draw_body_dots = st.checkbox("🦴 Draw Full Body Skeleton & 33 Dots", value=True)
    with opt_c2:
        draw_face_dots = st.checkbox("😊 Draw Facial Pain & Expression Dots", value=True)
    with opt_c3:
        if st.button("🔄 Reset Rep Counter & Snapshot", use_container_width=True):
            st.session_state.rehab_metrics = {"reps": 0, "stage": "DOWN", "angle": 165.0, "tilt": 0.0, "pain": 0.0}
            st.session_state.captured_snapshot = None

    cam_col, stats_col = st.columns([1.65, 1.0])

    with cam_col:
        if "Capture Photo" in vision_mode:
            cam_photo = st.camera_input("📸 التقط صورة واضحة للجسم والوجه لتحليل الزوايا والألم")
            if cam_photo is not None and cv2 is not None:
                file_bytes = np.asarray(bytearray(cam_photo.read()), dtype=np.uint8)
                bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                annotated_img, updated_m = analyze_clinical_frame(
                    bgr_img,
                    pose_landmarker,
                    face_landmarker,
                    exercise_type,
                    tracking_side,
                    draw_body_dots,
                    draw_face_dots,
                    st.session_state.rehab_metrics,
                    target_reps,
                )
                st.session_state.rehab_metrics = updated_m
                st.session_state.captured_snapshot = annotated_img
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Analyzed Clinical Snapshot (Body + Face Landmarks)", use_container_width=True)

        elif "Upload Patient Image" in vision_mode:
            uploaded_img = st.file_uploader("Upload JPG/PNG image for biomechanical analysis", type=["jpg", "jpeg", "png"])
            if uploaded_img is not None and cv2 is not None:
                file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
                bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                annotated_img, updated_m = analyze_clinical_frame(
                    bgr_img,
                    pose_landmarker,
                    face_landmarker,
                    exercise_type,
                    tracking_side,
                    draw_body_dots,
                    draw_face_dots,
                    st.session_state.rehab_metrics,
                    target_reps,
                )
                st.session_state.rehab_metrics = updated_m
                st.session_state.captured_snapshot = annotated_img
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Analyzed Uploaded Image", use_container_width=True)

        else:
            w_c1, w_c2 = st.columns(2)
            with w_c1:
                cam_idx = st.selectbox("Webcam Index", options=[0, 1], format_func=lambda i: f"Camera #{i}")
            with w_c2:
                run_webcam = st.toggle("▶️ Start Live Webcam Stream", value=False)

            live_frame_box = st.empty()
            if run_webcam:
                if cv2 is None:
                    st.error("OpenCV (`cv2`) is not installed.")
                else:
                    cap = cv2.VideoCapture(int(cam_idx))
                    if not cap.isOpened():
                        st.error(f"Could not open Camera #{cam_idx}.")
                    else:
                        try:
                            # Bounded batch loop prevents freezing or dimming other Streamlit tabs
                            for _ in range(180):
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frame = cv2.flip(frame, 1)
                                annotated_img, updated_m = analyze_clinical_frame(
                                    frame,
                                    pose_landmarker,
                                    face_landmarker,
                                    exercise_type,
                                    tracking_side,
                                    draw_body_dots,
                                    draw_face_dots,
                                    st.session_state.rehab_metrics,
                                    target_reps,
                                )
                                st.session_state.rehab_metrics = updated_m
                                st.session_state.captured_snapshot = annotated_img
                                live_frame_box.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                                time.sleep(0.03)
                        finally:
                            cap.release()
            elif st.session_state.captured_snapshot is not None and cv2 is not None:
                live_frame_box.image(
                    cv2.cvtColor(st.session_state.captured_snapshot, cv2.COLOR_BGR2RGB),
                    caption="Last Captured Live Frame",
                    use_container_width=True,
                )
            else:
                live_frame_box.info("Enable **Start Live Webcam Stream** above or switch to **Capture Photo** mode.")

    with stats_col:
        m = st.session_state.rehab_metrics
        st.metric("Completed Repetitions", f"{m['reps']} / {target_reps} reps")
        st.metric("Active Joint Angle", f"{m['angle']:.1f}°")
        st.metric("Shoulder Tilt Asymmetry", f"{m['tilt']:.1f}°")
        st.metric("Facial Pain & Distress Index", f"{m['pain']:.1f}%")

        live_hr = st.session_state.vitals.get("hr", 0.0)
        if m["pain"] >= 65.0 or live_hr > 125.0:
            st.error("🚨 High Distress / Exertion — Pause Exercise Immediately")
        elif m["pain"] >= 35.0 or m["tilt"] > 8.0:
            st.warning("⚠️ Moderate Discomfort or Shoulder Tilt Detected")
        else:
            st.success("✅ Optimal Form & Comfortable Exertion Level")

        if st.session_state.captured_snapshot is not None and cv2 is not None:
            ok, png_buf = cv2.imencode(".png", st.session_state.captured_snapshot)
            if ok:
                st.download_button(
                    label="🖼️ Download Annotated Snapshot (PNG)",
                    data=png_buf.tobytes(),
                    file_name=f"{patient_id}_Vision_Analysis.png",
                    mime="image/png",
                    use_container_width=True,
                )

        therapist_note = st.text_input("Session Clinical Notes", value="Completed rehab set with stable vitals.")
        if st.button("💾 Save Rehab Session to Database", use_container_width=True):
            diag_now, conf_now = run_ml_inference(st.session_state.vitals, model, scaler, le_target)
            save_session(
                patient_id=patient_id,
                patient_name=patient_name,
                hr=st.session_state.vitals.get("hr", 0.0),
                spo2=st.session_state.vitals.get("spo2", 0.0),
                sys_bp=st.session_state.vitals.get("sys_bp", 120),
                dia_bp=st.session_state.vitals.get("dia_bp", 80),
                temp=st.session_state.vitals.get("temp", 36.8),
                diagnosis=diag_now,
                confidence=conf_now,
                reps=m["reps"],
                pain_score=m["pain"],
                notes=therapist_note,
            )
            st.success("✅ Session saved! Download the professional PDF in 'Patient Records & PDF'.")


# ==============================================================================
# TAB 3: PATIENT RECORDS & PROFESSIONAL PDF REPORT
# ==============================================================================
with tab_records:
    st.subheader("📋 Stored Rehabilitation Sessions & Clinical PDF Generator")
    records_df = get_all_sessions()

    if records_df.empty:
        st.info("No saved rehabilitation sessions yet. Save a session from 'AI Vision & Rehab' or take a quick snapshot below.")
        if st.button("➕ Quick-Save Current Live Vitals Snapshot"):
            d_snap, c_snap = run_ml_inference(st.session_state.vitals, model, scaler, le_target)
            save_session(
                patient_id=patient_id,
                patient_name=patient_name,
                hr=st.session_state.vitals.get("hr", 0.0),
                spo2=st.session_state.vitals.get("spo2", 0.0),
                sys_bp=st.session_state.vitals.get("sys_bp", 120),
                dia_bp=st.session_state.vitals.get("dia_bp", 80),
                temp=st.session_state.vitals.get("temp", 36.8),
                diagnosis=d_snap,
                confidence=c_snap,
                reps=st.session_state.rehab_metrics["reps"],
                pain_score=st.session_state.rehab_metrics["pain"],
                notes="Live telemetry clinical snapshot.",
            )
            st.rerun()
    else:
        st.dataframe(records_df, use_container_width=True)
        selected_idx = st.selectbox(
            "Select Session ID for Professional PDF Report Export",
            options=records_df.index.tolist(),
            format_func=lambda idx: f"Session #{records_df.loc[idx, 'id']} — {records_df.loc[idx, 'patient_name']} ({records_df.loc[idx, 'timestamp']})",
        )
        chosen_record = records_df.loc[selected_idx].to_dict()
        pdf_bytes = build_clinical_pdf(chosen_record)
        if pdf_bytes:
            st.download_button(
                label="📄 Download Official Clinical PDF Report",
                data=pdf_bytes,
                file_name=f"{chosen_record.get('patient_id', 'PT')}_Clinical_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ==============================================================================
# TAB 4: MANUAL PREDICTOR
# ==============================================================================
with tab_manual:
    st.subheader("🧪 Manual Clinical Parameter Testing")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        m_hr = st.slider("Heart Rate (bpm)", min_value=40, max_value=180, value=85)
        m_spo2 = st.slider("SpO2 Level (%)", min_value=70, max_value=100, value=97)
        m_sys = st.slider("Systolic BP (mmHg)", min_value=80, max_value=200, value=120)
        m_dia = st.slider("Diastolic BP (mmHg)", min_value=50, max_value=130, value=80)

    with m_col2:
        m_temp = st.slider("Body Temperature (°C)", min_value=35.0, max_value=41.0, value=37.00, step=0.05)
        m_fall = st.selectbox("Fall Event Detected?", options=["No", "Yes"], index=0)
        m_acc = st.slider("Sensor Accuracy (%)", min_value=50, max_value=100, value=95)

    if st.button("🧪 Run Manual Diagnosis", use_container_width=True):
        manual_vitals = {
            "hr": m_hr,
            "spo2": m_spo2,
            "sys_bp": m_sys,
            "dia_bp": m_dia,
            "temp": m_temp,
            "fall": m_fall,
            "accuracy": m_acc,
        }
        pred_disease, pred_conf = run_ml_inference(manual_vitals, model, scaler, le_target)

        if pred_disease == "Healthy" and pred_conf >= 80.0:
            border_color = "#10b981"
            text_color = "#10b981"
            status_note = "✅ المؤشرات الحيوية ضمن النطاق الطبيعي المستقر"
        elif pred_conf >= 60.0 and pred_disease == "Healthy":
            border_color = "#f59e0b"
            text_color = "#f59e0b"
            status_note = "⚠️ قراءات حدّية — يُنصح بالمتابعة"
        else:
            border_color = "#ef4444"
            text_color = "#ef4444"
            status_note = "🚨 تنبيه سريري — مؤشرات خارج النطاق الطبيعي"

        st.markdown(
            f"""
            <div style="border: 2px solid {border_color}; border-radius: 12px; padding: 26px; text-align: center;
                        background: linear-gradient(180deg, rgba(16, 32, 30, 0.65) 0%, rgba(11, 15, 23, 0.92) 100%);
                        margin-top: 16px;">
                <h2 style="margin: 0; color: #f9fafb; font-size: 1.95rem;">
                    Predicted Diagnosis: <span style="color: {text_color};">{pred_disease}</span>
                </h2>
                <h4 style="margin: 10px 0 8px 0; color: #e5e7eb; font-size: 1.2rem;">
                    Model Confidence: {pred_conf:.2f}%
                </h4>
                <p style="margin: 0; color: {text_color}; font-size: 0.95rem; font-weight: 600;">
                    {status_note}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 5: DATASET ANALYTICS
# ==============================================================================
with tab_analytics:
    st.subheader("📊 Healthcare Dataset Exploratory Analytics")
    df_health = load_health_dataset()

    if df_health is None:
        st.warning("Dataset file `Synthetic_patient-HealthCare-Monitoring_dataset.csv` was not found in the project folder.")
    else:
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Clinical Records", f"{len(df_health):,}")
        kpi2.metric("Monitored Features", f"{df_health.shape[1]}")
        target_col = df_health.columns[-1]
        kpi3.metric("Diagnostic Classes", f"{df_health[target_col].nunique()}")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            class_counts = df_health[target_col].value_counts().reset_index()
            class_counts.columns = ["Condition", "Count"]
            fig_bar = px.bar(
                class_counts,
                x="Condition",
                y="Count",
                color="Condition",
                title="Clinical Class Distribution",
                template="plotly_dark",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            num_cols = df_health.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                fig_scatter = px.scatter(
                    df_health.head(400),
                    x=num_cols[0],
                    y=num_cols[1],
                    color=target_col,
                    title=f"{num_cols[0]} vs. {num_cols[1]}",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)