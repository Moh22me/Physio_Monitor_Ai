# session_manager.py
from datetime import datetime
import os
from fpdf import FPDF
import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

os.makedirs("data", exist_ok=True)
DATABASE_URL = "sqlite:///data/physio_sessions.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ClinicalSession(Base):
    """SQLAlchemy ORM model representing a multi-modal clinical rehab session."""

    __tablename__ = "clinical_sessions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    patient_id = Column(String(50), index=True, nullable=False)
    patient_name = Column(String(100), nullable=False)
    exercise_type = Column(String(100), default="General Assessment")

    # Physiological & ML Metrics
    heart_rate = Column(Float, default=0.0)
    spo2 = Column(Float, default=0.0)
    sys_bp = Column(Integer, default=120)
    dia_bp = Column(Integer, default=80)
    body_temp = Column(Float, default=37.0)
    ai_diagnosis = Column(String(100), default="N/A")
    ai_confidence = Column(Float, default=0.0)

    # Computer Vision Biomechanics & Pain Metrics
    joint_angle = Column(Float, default=0.0)
    reps_completed = Column(Integer, default=0)
    shoulder_tilt = Column(Float, default=0.0)
    pain_score = Column(Float, default=0.0)
    clinical_status = Column(String(150), default="Stable")
    therapist_notes = Column(String(500), default="")


def init_db():
    """Create database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def save_clinical_record(
    patient_id: str,
    patient_name: str,
    exercise_type: str,
    vitals: dict,
    ai_diagnosis: str,
    ai_confidence: float,
    vision_data: dict,
    clinical_status: str,
    therapist_notes: str = "",
) -> int:
    """Persist a combined hardware + AI + vision session snapshot into SQLite."""
    db = SessionLocal()
    try:
        record = ClinicalSession(
            timestamp=datetime.now(),
            patient_id=patient_id.strip() or "PT-001",
            patient_name=patient_name.strip() or "Anonymous Patient",
            exercise_type=exercise_type,
            heart_rate=float(vitals.get("hr", 0.0)),
            spo2=float(vitals.get("spo2", 0.0)),
            sys_bp=int(vitals.get("sys_bp", 120)),
            dia_bp=int(vitals.get("dia_bp", 80)),
            body_temp=float(vitals.get("temp", 37.0)),
            ai_diagnosis=str(ai_diagnosis),
            ai_confidence=round(float(ai_confidence), 2),
            joint_angle=float(vision_data.get("joint_angle", 0.0)),
            reps_completed=int(vision_data.get("reps", 0)),
            shoulder_tilt=float(vision_data.get("shoulder_tilt", 0.0)),
            pain_score=float(vision_data.get("pain_score", 0.0)),
            clinical_status=str(clinical_status),
            therapist_notes=therapist_notes.strip(),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id
    finally:
        db.close()


def fetch_all_sessions() -> pd.DataFrame:
    """Retrieve all stored clinical sessions ordered by newest first."""
    db = SessionLocal()
    try:
        rows = db.query(ClinicalSession).order_by(ClinicalSession.timestamp.desc()).all()
        if not rows:
            return pd.DataFrame()
        data = [
            {
                "Session ID": r.id,
                "Date & Time": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Patient ID": r.patient_id,
                "Patient Name": r.patient_name,
                "Exercise": r.exercise_type,
                "HR (bpm)": r.heart_rate,
                "SpO2 (%)": r.spo2,
                "BP (mmHg)": f"{r.sys_bp}/{r.dia_bp}",
                "Temp (C)": r.body_temp,
                "AI Diagnosis": r.ai_diagnosis,
                "Confidence (%)": r.ai_confidence,
                "Joint Angle (deg)": r.joint_angle,
                "Reps": r.reps_completed,
                "Shoulder Tilt (deg)": r.shoulder_tilt,
                "Pain Index (%)": r.pain_score,
                "Clinical Status": r.clinical_status,
                "Notes": r.therapist_notes,
            }
            for r in rows
        ]
        return pd.DataFrame(data)
    finally:
        db.close()


def delete_session_by_id(session_id: int) -> bool:
    """Delete a specific clinical record by ID."""
    db = SessionLocal()
    try:
        record = db.query(ClinicalSession).filter(ClinicalSession.id == session_id).first()
        if record:
            db.delete(record)
            db.commit()
            return True
        return False
    finally:
        db.close()


def _latin_safe(text: str) -> str:
    """Ensure text is compatible with core PDF fonts without encoding errors."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


class ClinicalReportPDF(FPDF):
    """Custom styled hospital-grade PDF report template."""

    def header(self):
        self.set_fill_color(13, 43, 42)
        self.rect(0, 0, 210, 32, "F")
        self.set_text_color(0, 255, 204)
        self.set_font("Helvetica", "B", 18)
        self.set_y(8)
        self.cell(0, 8, "PHYSIO MONITOR AI SYSTEM", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(220, 230, 235)
        self.set_font("Helvetica", "", 10)
        self.cell(
            0,
            6,
            "Multi-Modal Physiological & Biomechanical Rehabilitation Report",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(
            0,
            10,
            f"Generated by Physio Monitor AI Clinical Decision Support | Page {self.page_no()}",
            align="C",
        )


def generate_session_pdf(record_row: dict) -> bytes:
    """Build a formatted PDF report from a session dictionary and return raw bytes."""
    pdf = ClinicalReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # 1. Patient & Session Metadata Block
    pdf.set_text_color(20, 30, 40)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(235, 245, 244)
    pdf.cell(0, 9, "  1. PATIENT & SESSION INFORMATION", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    meta_pairs = [
        ("Session ID:", f"#{record_row.get('Session ID', 'LIVE')}", "Date & Time:", str(record_row.get("Date & Time", datetime.now().strftime("%Y-%m-%d %H:%M")))),
        ("Patient Name:", _latin_safe(record_row.get("Patient Name", "N/A")), "Patient ID:", _latin_safe(record_row.get("Patient ID", "N/A"))),
        ("Target Exercise:", _latin_safe(record_row.get("Exercise", "N/A")), "Clinical Status:", _latin_safe(record_row.get("Clinical Status", "N/A"))),
    ]

    for lbl1, val1, lbl2, val2 in meta_pairs:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(35, 7, lbl1)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(60, 7, val1)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(35, 7, lbl2)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(60, 7, val2, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # 2. Physiological Telemetry & ML Inference
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(235, 245, 244)
    pdf.cell(0, 9, "  2. PHYSIOLOGICAL TELEMETRY & AI DIAGNOSIS", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    vitals_rows = [
        ("Heart Rate (Arduino / Sensor)", f"{record_row.get('HR (bpm)', 0)} bpm"),
        ("Blood Oxygen Saturation (SpO2)", f"{record_row.get('SpO2 (%)', 0)} %"),
        ("Blood Pressure (Systolic/Diastolic)", f"{record_row.get('BP (mmHg)', '120/80')} mmHg"),
        ("Body Temperature", f"{record_row.get('Temp (C)', 37.0)} C"),
        ("AI Predicted Condition", _latin_safe(record_row.get("AI Diagnosis", "N/A"))),
        ("Model Diagnostic Confidence", f"{record_row.get('Confidence (%)', 0)} %"),
    ]

    for metric_name, metric_val in vitals_rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(95, 8, f"  {metric_name}", border=1)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(95, 8, f"  {metric_val}", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # 3. Computer Vision Biomechanics & Pain Assessment
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(235, 245, 244)
    pdf.cell(0, 9, "  3. COMPUTER VISION BIOMECHANICS & DISTRESS ANALYSIS", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    vision_rows = [
        ("Measured Joint Flexion Angle", f"{record_row.get('Joint Angle (deg)', 0)} degrees"),
        ("Completed Rehabilitation Reps", f"{record_row.get('Reps', 0)} repetitions"),
        ("Postural Shoulder Alignment Tilt", f"{record_row.get('Shoulder Tilt (deg)', 0)} degrees"),
        ("Facial Pain & Distress Index", f"{record_row.get('Pain Index (%)', 0)} %"),
    ]

    for v_name, v_val in vision_rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(95, 8, f"  {v_name}", border=1)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(95, 8, f"  {v_val}", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # 4. Clinical Notes Section
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(235, 245, 244)
    pdf.cell(0, 9, "  4. THERAPIST OBSERVATIONS & RECOMMENDATIONS", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    notes_text = _latin_safe(record_row.get("Notes", "")) or "No additional therapist remarks recorded for this session."
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, notes_text, border=1)

    return bytes(pdf.output())