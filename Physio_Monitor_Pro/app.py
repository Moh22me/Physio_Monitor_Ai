# app.py
from collections import deque
from datetime import datetime
import os
import cv2
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_lottie import st_lottie

from hardware_stream import (
    advance_ecg_buffer,
    connect_arduino,
    get_available_ports,
    read_arduino_vitals,
    simulate_physiological_vitals,
)
from session_manager import (
    delete_session_by_id,
    fetch_all_sessions,
    generate_session_pdf,
    init_db,
    save_clinical_record,
)
from vision_engine import (
    evaluate_multimodal_clinical_status,
    init_vision_models,
    process_vision_frame,
)

# 1. Page Configuration & Database Init
st.set_page_config(page_title="Physio Monitor AI", page_icon="🫀", layout="wide")
init_db()

# 2. Custom CSS (Neon ICU & Rehab Aesthetics)
st.markdown("""
    <style>
    .glow-text {
        font-size: 36px; color: #00ffcc; text-align: center; font-weight: bold;
        text-shadow: 0 0 10px #00ffcc, 0 0 20px #00ffcc; margin-bottom: 15px;
    }
    .metric-card {
        background-color: #11151c; border-radius: 10px; padding: 15px;
        border: 1px solid #1f2937; text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #0d2b2a 0%, #11151c 100%);
        border: 2px solid #00ffcc; border-radius: 12px; padding: 18px; text-align: center;
    }
    .status-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 13px; font-weight: bold; margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# 3. Cached Resource Loaders
@st.cache_data
def load_data():
    file_path = "data/Synthetic_patient-HealthCare-Monitoring_dataset.csv"
    return pd.read_csv(file_path) if os.path.exists(file_path) else None


@st.cache_resource
def load_ai_models():
    model_path = "models/physio_monitor_rf_optimized.pkl"
    scaler_path = "models/feature_scaler.pkl"
    target_path = "models/le_target.pkl"
    if all(os.path.exists(p) for p in [model_path, scaler_path, target_path]):
        return joblib.load(model_path), joblib.load(scaler_path), joblib.load(target_path)
    return None, None, None


@st.cache_resource
def get_cached_vision_models():
    return init_vision_models()


@st.cache_data
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def run_ml_inference(vitals: dict, model, scaler, le_target):
    """Run real-time Random Forest prediction on incoming vitals."""
    if model is None:
        return "Model Missing", 0.0

    hr = vitals["hr"]
    spo2 = vitals["spo2"]
    sys_bp = vitals["sys_bp"]
    dia_bp = vitals["dia_bp"]
    temp = vitals["temp"]
    fall_enc = 1 if vitals["fall"] == "Yes" else 0
    acc = vitals["accuracy"]

    hr_alert = 1 if (60 <= hr <= 100) else 0
    spo2_alert = 1 if spo2 >= 95 else 0
    bp_alert = 1 if (sys_bp <= 120 and dia_bp <= 80) else 0
    temp_alert = 1 if (36.5 <= temp <= 37.5) else 0

    features = [[hr, spo2, sys_bp, dia_bp, temp, fall_enc, acc, hr_alert, spo2_alert, bp_alert, temp_alert]]
    scaled = scaler.transform(features)
    pred_idx = model.predict(scaled)[0]
    confidence = float(np.max(model.predict_proba(scaled)[0])) * 100
    disease = le_target.inverse_transform([pred_idx])[0]
    return disease, confidence


# 4. Initialize Session State
if "ecg_buffer" not in st.session_state:
    st.session_state.ecg_buffer = deque([0.0] * 300, maxlen=300)
if "ecg_phase" not in st.session_state:
    st.session_state.ecg_phase = 0.0
if "sim_tick" not in st.session_state:
    st.session_state.sim_tick = 0
if "serial_conn" not in st.session_state:
    st.session_state.serial_conn = None
if "live_vitals" not in st.session_state:
    st.session_state.live_vitals = {
        "hr": 78.0, "spo2": 98.0, "sys_bp": 120, "dia_bp": 80,
        "temp": 36.8, "fall": "No", "accuracy": 98
    }
if "rep_state" not in st.session_state:
    st.session_state.rep_state = {"count": 0, "stage": "READY"}
if "latest_vision" not in st.session_state:
    st.session_state.latest_vision = {
        "joint_angle": 0.0,
        "shoulder_tilt": 0.0,
        "posture_status": "Not Assessed",
        "pain_score": 0.0,
        "pain_status": "Not Assessed",
        "reps": 0,
        "stage": "READY",
    }
if "active_exercise" not in st.session_state:
    st.session_state.active_exercise = "Right Elbow Flexion"

# Load Static Assets & Models
df = load_data()
model, scaler, le_target = load_ai_models()
pose_model, face_model = get_cached_vision_models()
lottie_heart = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_5n8kx1yq.json")

# 5. Sidebar: Patient Profile & Hardware Control
with st.sidebar:
    st.header("🧑‍⚕️ Active Patient Profile")
    patient_id_in = st.text_input("Patient ID", value="PT-101")
    patient_name_in = st.text_input("Patient Name (English for PDF)", value="Ahmed Hassan")

    st.divider()
    st.header("🔌 Data Source Control")
    source_mode = st.radio(
        "Select Input Mode:",
        ["🟢 Live Physiological Simulator", "🔌 Arduino Nano (USB Serial)"],
        index=0
    )

    if source_mode == "🔌 Arduino Nano (USB Serial)":
        ports = get_available_ports()
        selected_port = st.selectbox("Select COM Port", ports if ports else ["No Ports Detected"])
        baud_rate = st.selectbox("Baud Rate", [9600, 115200], index=0)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Connect", use_container_width=True) and ports:
                ser, err = connect_arduino(selected_port, baud_rate)
                if ser:
                    st.session_state.serial_conn = ser
                    st.success(f"Connected: {selected_port}")
                else:
                    st.error(f"Failed: {err}")
        with col_c2:
            if st.button("Disconnect", use_container_width=True):
                if st.session_state.serial_conn:
                    st.session_state.serial_conn.close()
                    st.session_state.serial_conn = None
                    st.info("Disconnected")

        st.divider()
        st.caption("Baseline Parameters (if not sent by Arduino):")
        st.session_state.live_vitals["sys_bp"] = st.slider("Baseline Systolic BP", 90, 190, 120)
        st.session_state.live_vitals["dia_bp"] = st.slider("Baseline Diastolic BP", 60, 120, 80)
        st.session_state.live_vitals["fall"] = st.selectbox("Fall Status", ["No", "Yes"])
    else:
        sim_scenario = st.selectbox(
            "Simulation Scenario:",
            [
                "Normal Rest (حالة راحة طبيعية)",
                "Active Rehab Exercise (مجهود تأهيلي)",
                "Respiratory Distress (أزمة تنفسية ونقص أكسجين)",
                "Critical Hypertensive & Fall (حالة طوارئ وسقوط)",
            ]
        )

    stream_active = st.toggle("▶️ Run Real-Time ECG Stream", value=True)

# Header
st.markdown("<div class='glow-text'>Physio Monitor AI — Clinical Platform</div>", unsafe_allow_html=True)

# 6. Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🫀 Dynamic Live Monitor",
    "👁️ AI Vision & Rehab",
    "📋 Patient Records & PDF",
    "🧪 Manual Predictor",
    "📊 Dataset Analytics",
])

# ==========================================
# TAB 1: DYNAMIC SCROLLING MONITOR (FRAGMENT)
# ==========================================
with tab1:
    @st.fragment(run_every=0.25 if stream_active else None)
    def render_live_telemetry():
        if source_mode == "🔌 Arduino Nano (USB Serial)":
            if st.session_state.serial_conn and st.session_state.serial_conn.is_open:
                st.session_state.live_vitals = read_arduino_vitals(
                    st.session_state.serial_conn, st.session_state.live_vitals
                )
                badge_html = "<span class='status-badge' style='background:#004d40; color:#00ffcc;'>● ARDUINO NANO STREAMING LIVE</span>"
            else:
                badge_html = "<span class='status-badge' style='background:#4a1515; color:#ff6b6b;'>● ARDUINO DISCONNECTED — WAITING FOR PORT</span>"
        else:
            if stream_active:
                st.session_state.sim_tick += 1
                st.session_state.live_vitals = simulate_physiological_vitals(
                    sim_scenario, st.session_state.sim_tick
                )
            badge_html = f"<span class='status-badge' style='background:#102a43; color:#38bdf8;'>● SIMULATION ACTIVE: {sim_scenario}</span>"

        vitals = st.session_state.live_vitals

        if stream_active:
            st.session_state.ecg_phase = advance_ecg_buffer(
                st.session_state.ecg_buffer,
                st.session_state.ecg_phase,
                vitals["hr"],
                dt=0.25,
                fs=100
            )

        disease, confidence = run_ml_inference(vitals, model, scaler, le_target)
        st.markdown(f"<div style='text-align:center;'>{badge_html}</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 1.4, 1])

        with c1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            hr_status = "NORMAL" if 60 <= vitals["hr"] <= 100 else "ABNORMAL"
            spo2_status = "NORMAL" if vitals["spo2"] >= 95 else "LOW O2"
            st.metric("Heart Rate (Arduino)", f"{vitals['hr']} bpm", hr_status, delta_color="inverse")
            st.metric("SpO2 Level", f"{vitals['spo2']}%", spo2_status, delta_color="inverse")
            st.metric("Body Temp", f"{vitals['temp']} °C")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            if lottie_heart:
                st_lottie(lottie_heart, height=160, key="live_heart_anim")

            if confidence >= 80:
                box_color = "#00ffcc"
                note = "✅ تشخيص مؤكد ومتطابق فسيولوجياً"
            elif confidence >= 60:
                box_color = "#ffcc00"
                note = "ℹ️ مؤشرات متداخلة — يفضل الملاحظة"
            else:
                box_color = "#ff4444"
                note = "⚠️ قراءات غير حاسمة — راجع الطبيب"

            st.markdown(f"""
                <div class='prediction-box' style='border-color:{box_color}; box-shadow: 0 0 12px {box_color}44;'>
                    <div style='font-size:14px; color:#9ca3af;'>LIVE AI DIAGNOSIS</div>
                    <h2 style='color:{box_color}; margin:5px 0;'>{disease}</h2>
                    <div style='font-size:16px;'>Confidence: <b>{confidence:.1f}%</b></div>
                    <div style='font-size:13px; color:{box_color}; margin-top:6px;'>{note}</div>
                </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Systolic BP", f"{vitals['sys_bp']} mmHg")
            st.metric("Diastolic BP", f"{vitals['dia_bp']} mmHg")
            st.metric("Fall Detection", vitals["fall"])
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("📈 Real-Time Scrolling ECG Monitor (Lead II)")
        y_vals = list(st.session_state.ecg_buffer)
        x_vals = list(range(len(y_vals)))

        fig = go.Figure(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color="#00ffcc", width=2.2),
            )
        )
        fig.update_layout(
            plot_bgcolor="#05080d",
            paper_bgcolor="#0e1117",
            uirevision="constant_ecg",
            xaxis=dict(showgrid=True, gridcolor="#132226", zeroline=False, showticklabels=False, range=[0, 300]),
            yaxis=dict(showgrid=True, gridcolor="#132226", zeroline=False, showticklabels=False, range=[-0.7, 2.0]),
            margin=dict(l=0, r=0, t=5, b=0),
            height=250,
        )
        st.plotly_chart(fig, use_container_width=True, key="dynamic_ecg_plot")

    render_live_telemetry()

    st.divider()
    save_col1, save_col2 = st.columns([2.5, 1])
    with save_col1:
        live_notes = st.text_input(
            "ملاحظات الأخصائي على القراءة الحية (اختياري):",
            placeholder="e.g., Patient vitals stable during resting ECG monitoring.",
            key="tab1_notes",
        )
    with save_col2:
        st.write("")
        if st.button("💾 Save Vitals Snapshot to DB", use_container_width=True, key="save_tab1"):
            d_now, c_now = run_ml_inference(st.session_state.live_vitals, model, scaler, le_target)
            rec_id = save_clinical_record(
                patient_id=patient_id_in,
                patient_name=patient_name_in,
                exercise_type="Vitals & ECG Monitoring",
                vitals=st.session_state.live_vitals,
                ai_diagnosis=d_now,
                ai_confidence=c_now,
                vision_data=st.session_state.latest_vision,
                clinical_status="Vitals Logged",
                therapist_notes=live_notes,
            )
            st.success(f"✅ تم حفظ السجل رقم #{rec_id} للمريض {patient_name_in} في قاعدة البيانات!")

# ==========================================
# TAB 2: AI VISION & BIOMECHANICS ASSESSMENT
# ==========================================
with tab2:
    st.subheader("👁️ Computer Vision Rehabilitation & Facial Distress Analyzer")
    st.write("تحليل حركة المفاصل، اتزان القوام، وتعبيرات الألم من الوجه مع ربطها لحظياً بنبض القلب وحفظها في سجل المريض.")

    v_col1, v_col2, v_col3 = st.columns([1.2, 1.2, 0.8])
    with v_col1:
        vision_mode = st.radio(
            "اختيار وضع الكاميرا:",
            ["📸 Snapshot / Photo Capture (تصوير المريض)", "🎥 Live Webcam Stream (بث مباشر)"],
            horizontal=True,
        )
    with v_col2:
        exercise_choice = st.selectbox(
            "التمرين التأهيلي المستهدف (Target Joint):",
            [
                "Right Elbow Flexion (ثني الكوع الأيمن)",
                "Right Shoulder Abduction (رفع الكتف الأيمن)",
                "Right Knee Extension (تأهيل الركبة اليمنى)",
            ],
        )
        st.session_state.active_exercise = exercise_choice.split(" (")[0]
    with v_col3:
        st.write("")
        if st.button("🔄 Reset Rep Counter", use_container_width=True):
            st.session_state.rep_state = {"count": 0, "stage": "READY"}

    st.divider()

    if vision_mode == "📸 Snapshot / Photo Capture (تصوير المريض)":
        snap_col1, snap_col2 = st.columns([1, 1])
        with snap_col1:
            img_file = st.camera_input("التقط صورة للمريض أمام الكاميرا")
            uploaded_file = st.file_uploader("أو ارفع صورة جاهزة للمريض (JPG / PNG)", type=["jpg", "jpeg", "png"])
            active_file = img_file if img_file is not None else uploaded_file

        with snap_col2:
            if active_file is not None:
                file_bytes = np.asarray(bytearray(active_file.read()), dtype=np.uint8)
                frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if st.session_state.serial_conn and st.session_state.serial_conn.is_open:
                    st.session_state.live_vitals = read_arduino_vitals(
                        st.session_state.serial_conn, st.session_state.live_vitals
                    )

                annotated_bgr, v_data = process_vision_frame(
                    frame_bgr, pose_model, face_model, exercise_choice, st.session_state.rep_state
                )
                st.session_state.latest_vision = v_data

                st.image(
                    cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                    caption="AI Skeletal & Facial Analysis",
                    use_container_width=True,
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Joint Angle", f"{v_data['joint_angle']}°")
                m2.metric("Shoulder Tilt", f"{v_data['shoulder_tilt']}°", v_data["posture_status"])
                m3.metric("Facial Pain Index", f"{v_data['pain_score']}%", v_data["pain_status"])

                f_color, f_title, f_desc = evaluate_multimodal_clinical_status(
                    st.session_state.live_vitals, v_data
                )
                st.markdown(f"""
                    <div class='prediction-box' style='border-color:{f_color}; margin-top:12px;'>
                        <h3 style='color:{f_color}; margin:0;'>{f_title}</h3>
                        <p style='margin:8px 0 4px 0; font-size:15px;'>{f_desc}</p>
                        <small style='color:#9ca3af;'>Linked Heart Rate: <b>{st.session_state.live_vitals['hr']} bpm</b> | SpO2: <b>{st.session_state.live_vitals['spo2']}%</b></small>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("قم بالتقاط صورة أو رفع صورة للمريض لبدء تحليل الهيكل الحركي وتعبيرات الوجه.")

    else:
        run_cam = st.checkbox("▶️ تشغيل البث المباشر لكاميرا اللابتوب (Start Live Webcam)")
        cam_col1, cam_col2 = st.columns([1.4, 1])
        frame_placeholder = cam_col1.empty()
        metrics_placeholder = cam_col2.empty()

        if run_cam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("⚠️ تعذر الوصول إلى الكاميرا. تأكد من عدم استخدامها في برنامج آخر أو استخدم وضع Snapshot.")
            else:
                while run_cam:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if st.session_state.serial_conn and st.session_state.serial_conn.is_open:
                        st.session_state.live_vitals = read_arduino_vitals(
                            st.session_state.serial_conn, st.session_state.live_vitals
                        )

                    annotated_bgr, v_data = process_vision_frame(
                        frame, pose_model, face_model, exercise_choice, st.session_state.rep_state
                    )
                    st.session_state.latest_vision = v_data

                    frame_placeholder.image(
                        cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )

                    f_color, f_title, f_desc = evaluate_multimodal_clinical_status(
                        st.session_state.live_vitals, v_data
                    )
                    with metrics_placeholder.container():
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Completed Reps (العدّات)", f"{v_data['reps']} Reps", f"Stage: {v_data['stage']}")
                        st.metric("Live Joint Angle", f"{v_data['joint_angle']}°")
                        st.metric("Posture Tilt", f"{v_data['shoulder_tilt']}°", v_data["posture_status"])
                        st.metric("Facial Pain Score", f"{v_data['pain_score']}%", v_data["pain_status"])
                        st.markdown("</div>", unsafe_allow_html=True)

                        st.markdown(f"""
                            <div class='prediction-box' style='border-color:{f_color}; margin-top:10px;'>
                                <h4 style='color:{f_color}; margin:0;'>{f_title}</h4>
                                <p style='font-size:13px; margin-top:6px;'>{f_desc}</p>
                                <small style='color:#9ca3af;'>Live HR: {st.session_state.live_vitals['hr']} bpm</small>
                            </div>
                        """, unsafe_allow_html=True)
                cap.release()

    st.divider()
    v_save_col1, v_save_col2 = st.columns([2.5, 1])
    with v_save_col1:
        rehab_notes = st.text_input(
            "ملاحظات الجلسة التأهيلية (تظهر في التقرير الطبي PDF):",
            placeholder="e.g., Completed 10 reps with good range of motion and minimal discomfort.",
            key="tab2_notes",
        )
    with v_save_col2:
        st.write("")
        if st.button("💾 Save Full Rehab Session to DB", use_container_width=True, key="save_tab2"):
            d_now, c_now = run_ml_inference(st.session_state.live_vitals, model, scaler, le_target)
            _, status_title, _ = evaluate_multimodal_clinical_status(
                st.session_state.live_vitals, st.session_state.latest_vision
            )
            clean_status = "Critical Alert" if "إنذار" in status_title else ("Caution / Strain" if "تنبيه" in status_title else "Safe & Stable")
            rec_id = save_clinical_record(
                patient_id=patient_id_in,
                patient_name=patient_name_in,
                exercise_type=st.session_state.active_exercise,
                vitals=st.session_state.live_vitals,
                ai_diagnosis=d_now,
                ai_confidence=c_now,
                vision_data=st.session_state.latest_vision,
                clinical_status=clean_status,
                therapist_notes=rehab_notes,
            )
            st.success(f"✅ تم حفظ الجلسة التأهيلية الشاملة رقم #{rec_id} بنجاح!")

# ==========================================
# TAB 3: PATIENT RECORDS, PROGRESS & PDF EXPORT
# ==========================================
with tab3:
    st.subheader("📋 Patient Clinical History & PDF Report Generator")
    sessions_df = fetch_all_sessions()

    if sessions_df.empty:
        st.info("لا توجد جلسات محفوظة بعد. استخدم زر الحفظ في تبويب المراقبة الحية أو تبويب الرؤية الحاسوبية لتسجيل أول جلسة.")
    else:
        filter_col1, filter_col2 = st.columns([1.5, 2.5])
        with filter_col1:
            unique_patients = ["All Patients"] + sorted(sessions_df["Patient ID"].unique().tolist())
            selected_pt = st.selectbox("تصفية السجلات برقم المريض (Filter by Patient ID):", unique_patients)

        filtered_sessions = (
            sessions_df if selected_pt == "All Patients" else sessions_df[sessions_df["Patient ID"] == selected_pt]
        )

        # Longitudinal Progress Chart when a specific patient is selected
        if selected_pt != "All Patients" and len(filtered_sessions) >= 1:
            st.markdown(f"#### 📈 منحنى تطور الحالة للمريض: **{filtered_sessions.iloc[0]['Patient Name']} ({selected_pt})**")
            chart_df = filtered_sessions.sort_values("Session ID")
            prog_fig = go.Figure()
            prog_fig.add_trace(
                go.Scatter(
                    x=chart_df["Date & Time"],
                    y=chart_df["HR (bpm)"],
                    mode="lines+markers",
                    name="Heart Rate (bpm)",
                    line=dict(color="#00ffcc", width=2.5),
                )
            )
            prog_fig.add_trace(
                go.Scatter(
                    x=chart_df["Date & Time"],
                    y=chart_df["Pain Index (%)"],
                    mode="lines+markers",
                    name="Facial Pain Index (%)",
                    line=dict(color="#ff4444", width=2.5),
                )
            )
            prog_fig.add_trace(
                go.Scatter(
                    x=chart_df["Date & Time"],
                    y=chart_df["Joint Angle (deg)"],
                    mode="lines+markers",
                    name="Joint Angle (deg)",
                    line=dict(color="#ffcc00", width=2.5),
                )
            )
            prog_fig.update_layout(
                plot_bgcolor="#05080d",
                paper_bgcolor="#0e1117",
                height=280,
                margin=dict(l=10, r=10, t=25, b=10),
            )
            st.plotly_chart(prog_fig, use_container_width=True)

        st.dataframe(filtered_sessions, use_container_width=True)

        st.divider()
        st.markdown("#### 📄 تصدير التقرير الطبي الرسمي (Clinical PDF Export)")
        exp_col1, exp_col2, exp_col3 = st.columns([2, 1.2, 0.8])

        with exp_col1:
            session_options = {
                f"Session #{row['Session ID']} | {row['Patient Name']} ({row['Patient ID']}) - {row['Date & Time']}": row
                for _, row in filtered_sessions.iterrows()
            }
            chosen_label = st.selectbox("اختر الجلسة المطلوب طباعة تقريرها:", list(session_options.keys()))
            chosen_row = session_options[chosen_label].to_dict()

        with exp_col2:
            st.write("")
            pdf_bytes = generate_session_pdf(chosen_row)
            file_name_pdf = f"Physio_Report_{chosen_row['Patient ID']}_Session_{chosen_row['Session ID']}.pdf"
            st.download_button(
                label="⬇️ Download Clinical PDF Report",
                data=pdf_bytes,
                file_name=file_name_pdf,
                mime="application/pdf",
                use_container_width=True,
            )

        with exp_col3:
            st.write("")
            if st.button("🗑️ Delete Record", use_container_width=True):
                if delete_session_by_id(int(chosen_row["Session ID"])):
                    st.warning(f"تم حذف السجل #{chosen_row['Session ID']}")
                    st.rerun()

# ==========================================
# TAB 4: MANUAL PREDICTOR (PRESERVED)
# ==========================================
with tab4:
    st.subheader("🧪 Manual Clinical Parameter Testing")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        hr_in = st.slider("Heart Rate (bpm)", 40, 180, 85)
        spo2_in = st.slider("SpO2 Level (%)", 70, 100, 97)
        sys_bp_in = st.slider("Systolic BP (mmHg)", 80, 200, 120)
        dia_bp_in = st.slider("Diastolic BP (mmHg)", 50, 130, 80)
    with col_in2:
        temp_in = st.slider("Body Temperature (°C)", 35.0, 41.0, 37.0)
        fall_in = st.selectbox("Fall Event Detected?", ["No", "Yes"])
        acc_in = st.slider("Sensor Accuracy (%)", 80, 100, 95)

    if st.button("🚀 Run Manual Diagnosis", use_container_width=True):
        manual_vitals = {
            "hr": hr_in, "spo2": spo2_in, "sys_bp": sys_bp_in,
            "dia_bp": dia_bp_in, "temp": temp_in, "fall": fall_in, "accuracy": acc_in
        }
        disease_m, conf_m = run_ml_inference(manual_vitals, model, scaler, le_target)
        color_m = "#00ffcc" if conf_m >= 80 else ("#ffcc00" if conf_m >= 60 else "#ff4444")
        st.markdown(f"""
            <div class='prediction-box' style='border-color:{color_m}; margin-top:15px;'>
                <h2>Predicted Diagnosis: <span style='color:{color_m};'>{disease_m}</span></h2>
                <h4>Model Confidence: <span style='color:{color_m};'>{conf_m:.2f}%</span></h4>
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 5: DATASET ANALYTICS
# ==========================================
with tab5:
    if df is not None:
        st.subheader("📊 Training Dataset Overview")
        st.dataframe(df.head(100), use_container_width=True)
        ca, cb = st.columns(2)
        with ca:
            st.bar_chart(df["Predicted Disease"].value_counts())
        with cb:
            st.bar_chart(df["Fall Detection"].value_counts())
    else:
        st.warning("Dataset CSV not found in 'data/' folder.")