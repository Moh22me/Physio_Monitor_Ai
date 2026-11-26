import os
import sqlite3
from datetime import datetime
import pandas as pd

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

DB_PATH = "physio_sessions.db"


def init_db() -> None:
    """Ensure SQLite sessions table exists."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            patient_id TEXT,
            patient_name TEXT,
            hr REAL,
            spo2 REAL,
            sys_bp INTEGER,
            dia_bp INTEGER,
            temp REAL,
            diagnosis TEXT,
            confidence REAL,
            reps INTEGER,
            pain_score REAL,
            notes TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_session(
    patient_id: str,
    patient_name: str,
    hr: float,
    spo2: float,
    sys_bp: int,
    dia_bp: int,
    temp: float,
    diagnosis: str,
    confidence: float,
    reps: int,
    pain_score: float,
    notes: str,
) -> bool:
    """Insert a new rehabilitation & telemetry record into SQLite."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sessions (
            timestamp, patient_id, patient_name, hr, spo2,
            sys_bp, dia_bp, temp, diagnosis, confidence, reps, pain_score, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            patient_id,
            patient_name,
            float(hr),
            float(spo2),
            int(sys_bp),
            int(dia_bp),
            float(temp),
            str(diagnosis),
            float(confidence),
            int(reps),
            float(pain_score),
            str(notes),
        ),
    )
    conn.commit()
    conn.close()
    return True


def get_all_sessions() -> pd.DataFrame:
    """Return all recorded patient sessions sorted newest first."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM sessions ORDER BY id DESC", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def _clean_latin1(text) -> str:
    """Sanitize text for FPDF core Helvetica font."""
    cleaned = str(text if text is not None else "-").encode("latin-1", "replace").decode("latin-1").strip()
    return cleaned if cleaned else "-"


def build_clinical_pdf(record: dict) -> bytes:
    """Generate a styled, hospital-grade clinical PDF report with structured tables."""
    if FPDF is None:
        return b""

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    usable_w = 180.0  # A4 width (210mm) - 30mm margins

    # --- 1. TOP CLINICAL HEADER BANNER ---
    pdf.set_fill_color(15, 23, 42)  # Dark Navy #0F172A
    pdf.rect(15, 15, usable_w, 26, style="F")
    pdf.set_fill_color(16, 185, 129)  # Emerald Accent Strip #10B981
    pdf.rect(15, 41, usable_w, 2.2, style="F")

    pdf.set_xy(20, 19)
    pdf.set_text_color(16, 185, 129)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(usable_w - 10, 8, "PHYSIO MONITOR AI", new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(20)
    pdf.set_text_color(226, 232, 240)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        usable_w - 10,
        6,
        "Multi-Modal Clinical Telemetry & Biomechanical Rehabilitation Report",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(9)

    # --- 2. PATIENT & SESSION METADATA TABLE ---
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_x(15)
    pdf.cell(usable_w, 7, "1. PATIENT & SESSION IDENTIFICATION", new_x="LMARGIN", new_y="NEXT")

    meta_rows = [
        ("Patient ID:", _clean_latin1(record.get("patient_id", "PT-101")), "Session Date:", _clean_latin1(record.get("timestamp", "-"))),
        ("Patient Name:", _clean_latin1(record.get("patient_name", "Ahmed Hassan")), "Record ID:", f"#{_clean_latin1(record.get('id', '1'))}"),
    ]
    col_widths = [35.0, 55.0, 35.0, 55.0]
    for r in meta_rows:
        pdf.set_x(15)
        pdf.set_fill_color(241, 245, 249)
        pdf.set_draw_color(203, 213, 225)
        for idx, val in enumerate(r):
            is_label = (idx % 2 == 0)
            pdf.set_font("Helvetica", "B" if is_label else "", 9.5)
            pdf.cell(col_widths[idx], 8, f" {val}", border=1, fill=is_label, new_x="RIGHT", new_y="TOP")
        pdf.ln(8)
    pdf.ln(5)

    # --- 3. AI DIAGNOSTIC SUMMARY BOX ---
    diagnosis = _clean_latin1(record.get("diagnosis", "Healthy"))
    conf = float(record.get("confidence", 0.0))
    is_healthy = (diagnosis.lower() == "healthy" and conf >= 80.0)

    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(usable_w, 7, "2. AI CLINICAL TRIAGE ASSESSMENT", new_x="LMARGIN", new_y="NEXT")

    if is_healthy:
        pdf.set_fill_color(236, 253, 245)
        pdf.set_draw_color(16, 185, 129)
        status_rgb = (5, 150, 105)
        rec_text = "Vital signs are stable within normal physiological resting ranges."
    else:
        pdf.set_fill_color(254, 242, 242)
        pdf.set_draw_color(239, 68, 68)
        status_rgb = (220, 38, 38)
        rec_text = "Abnormal physiological parameter detected. Clinical review is advised."

    box_y = pdf.get_y()
    pdf.rect(15, box_y, usable_w, 20, style="DF")
    pdf.set_xy(19, box_y + 3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*status_rgb)
    pdf.cell(usable_w - 8, 7, f"Primary Diagnosis: {diagnosis}   |   Model Confidence: {conf:.1f}%", new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(19)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(usable_w - 8, 6, rec_text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(box_y + 24)

    # --- 4. PHYSIOLOGICAL VITAL SIGNS TABLE ---
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(usable_w, 7, "3. PHYSIOLOGICAL TELEMETRY MEASUREMENTS", new_x="LMARGIN", new_y="NEXT")

    v_widths = [62.0, 40.0, 44.0, 34.0]
    headers = ["Vital Parameter", "Measured Value", "Reference Range", "Status"]
    pdf.set_x(15)
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9.5)
    for w_c, h_txt in zip(v_widths, headers):
        pdf.cell(w_c, 8.5, f" {h_txt}", border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.ln(8.5)

    hr = float(record.get("hr", 0.0))
    spo2 = float(record.get("spo2", 0.0))
    sys_bp = int(record.get("sys_bp", 120))
    dia_bp = int(record.get("dia_bp", 80))
    temp = float(record.get("temp", 36.8))

    vital_rows = [
        ("Heart Rate (Optical PPG)", f"{hr:.1f} bpm", "60.0 - 100.0 bpm", "NORMAL" if 60 <= hr <= 100 else ("NO DATA" if hr <= 0 else "ABNORMAL")),
        ("Blood Oxygen Saturation (SpO2)", f"{spo2:.1f} %", "95.0 - 100.0 %", "NORMAL" if spo2 >= 95 else ("NO DATA" if spo2 <= 0 else "LOW SPO2")),
        ("Systolic Blood Pressure", f"{sys_bp} mmHg", "90 - 129 mmHg", "NORMAL" if 90 <= sys_bp <= 129 else "ELEVATED"),
        ("Diastolic Blood Pressure", f"{dia_bp} mmHg", "60 - 84 mmHg", "NORMAL" if 60 <= dia_bp <= 84 else "ELEVATED"),
        ("Body Temperature", f"{temp:.1f} C", "36.1 - 37.5 C", "NORMAL" if 36.1 <= temp <= 37.5 else "ABNORMAL"),
    ]

    for row_idx, (param, val_str, ref_str, st_str) in enumerate(vital_rows):
        pdf.set_x(15)
        if row_idx % 2 == 0:
            pdf.set_fill_color(248, 250, 252)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_draw_color(203, 213, 225)
        pdf.set_text_color(15, 23, 42)
        pdf.set_font("Helvetica", "", 9.5)

        pdf.cell(v_widths[0], 8, f" {param}", border=1, fill=True, new_x="RIGHT", new_y="TOP")
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(v_widths[1], 8, f" {val_str}", border=1, fill=True, new_x="RIGHT", new_y="TOP")
        pdf.set_font("Helvetica", "", 9.5)
        pdf.cell(v_widths[2], 8, f" {ref_str}", border=1, fill=True, new_x="RIGHT", new_y="TOP")

        if st_str == "NORMAL":
            pdf.set_text_color(5, 150, 105)
        elif st_str == "NO DATA":
            pdf.set_text_color(180, 83, 9)
        else:
            pdf.set_text_color(220, 38, 38)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(v_widths[3], 8, f" {st_str}", border=1, fill=True, new_x="RIGHT", new_y="TOP")
        pdf.ln(8)

    pdf.ln(5)

    # --- 5. REHABILITATION & BIOMECHANICS SUMMARY ---
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(usable_w, 7, "4. REHABILITATION & BIOMECHANICAL VISION SUMMARY", new_x="LMARGIN", new_y="NEXT")

    reps = int(record.get("reps", 0))
    pain = float(record.get("pain_score", 0.0))
    notes = _clean_latin1(record.get("notes", "Completed session safely."))

    pdf.set_x(15)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.cell(55.0, 8, " Completed Repetitions:", border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.cell(35.0, 8, f" {reps} reps", border=1, fill=False, new_x="RIGHT", new_y="TOP")
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.cell(55.0, 8, " Facial Pain Index:", border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.cell(35.0, 8, f" {pain:.1f} %", border=1, fill=False, new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.cell(55.0, 8, " Clinical Notes:", border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.multi_cell(usable_w - 55.0, 8, f" {notes}", border=1, new_x="LMARGIN", new_y="NEXT")

    # --- 6. FOOTER ---
    pdf.ln(8)
    pdf.set_draw_color(203, 213, 225)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(2)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(
        usable_w,
        6,
        "Generated by Physio Monitor AI Clinical Platform | Automated Biometric & Vision Telemetry Summary.",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    raw_output = pdf.output()
    if isinstance(raw_output, (bytes, bytearray)):
        return bytes(raw_output)
    return str(raw_output).encode("latin-1")