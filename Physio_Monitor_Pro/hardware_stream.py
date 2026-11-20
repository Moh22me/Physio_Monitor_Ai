# hardware_stream.py
import time
import math
import random
import numpy as np
import serial
import serial.tools.list_ports


def get_available_ports():
    """Scan and return a list of active Serial COM ports."""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def connect_arduino(port: str, baud_rate: int = 9600):
    """Establish a non-blocking serial connection with Arduino Nano."""
    try:
        ser = serial.Serial(port, baud_rate, timeout=0.05)
        time.sleep(1.5)  # Allow Arduino Nano bootloader reset
        ser.reset_input_buffer()
        return ser, None
    except Exception as e:
        return None, str(e)


def read_arduino_vitals(ser, current_vitals: dict) -> dict:
    """
    Read latest serial line from Arduino Nano.
    Supports either single BPM value (e.g., '85')
    or CSV format: 'BPM,SPO2,TEMP' (e.g., '85,97,37.0').
    """
    if ser is None or not ser.is_open:
        return current_vitals

    try:
        while ser.in_waiting > 0:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                continue

            parts = [p.strip() for p in raw_line.split(",")]
            if len(parts) >= 1 and parts[0].replace(".", "", 1).isdigit():
                hr = float(parts[0])
                if 35 <= hr <= 220:  # Physiological sanity filter
                    current_vitals["hr"] = round(hr, 1)

            if len(parts) >= 2 and parts[1].replace(".", "", 1).isdigit():
                spo2 = float(parts[1])
                if 60 <= spo2 <= 100:
                    current_vitals["spo2"] = round(spo2, 1)

            if len(parts) >= 3 and parts[2].replace(".", "", 1).isdigit():
                temp = float(parts[2])
                if 34.0 <= temp <= 42.5:
                    current_vitals["temp"] = round(temp, 1)

    except Exception:
        pass

    return current_vitals


def simulate_physiological_vitals(scenario: str, tick: int) -> dict:
    """Generate medically correlated vital signs for live simulation."""
    wave = math.sin(tick * 0.25)
    noise = random.uniform(-0.8, 0.8)

    if scenario == "Normal Rest (حالة راحة طبيعية)":
        return {
            "hr": round(76 + wave * 4 + noise, 1),
            "spo2": round(98 + random.uniform(-0.5, 0.5), 1),
            "sys_bp": round(118 + wave * 3),
            "dia_bp": round(78 + wave * 2),
            "temp": round(36.8 + random.uniform(-0.1, 0.1), 1),
            "fall": "No",
            "accuracy": 98,
        }
    elif scenario == "Active Rehab Exercise (مجهود تأهيلي)":
        return {
            "hr": round(112 + wave * 6 + noise, 1),
            "spo2": round(96 + random.uniform(-0.5, 0.5), 1),
            "sys_bp": round(136 + wave * 4),
            "dia_bp": round(84 + wave * 2),
            "temp": round(37.3 + random.uniform(-0.1, 0.1), 1),
            "fall": "No",
            "accuracy": 96,
        }
    elif scenario == "Respiratory Distress (أزمة تنفسية ونقص أكسجين)":
        return {
            "hr": round(128 + wave * 5 + noise, 1),
            "spo2": round(86 + wave * 1.5, 1),
            "sys_bp": round(134 + wave * 4),
            "dia_bp": round(86 + wave * 2),
            "temp": round(37.4 + random.uniform(-0.1, 0.1), 1),
            "fall": "No",
            "accuracy": 95,
        }
    else:  # Critical Hypertensive & Fall Event
        return {
            "hr": round(138 + wave * 6 + noise, 1),
            "spo2": round(93 + wave * 1.0, 1),
            "sys_bp": round(185 + wave * 5),
            "dia_bp": round(118 + wave * 3),
            "temp": round(38.9 + random.uniform(-0.1, 0.1), 1),
            "fall": "Yes",
            "accuracy": 97,
        }


def advance_ecg_buffer(ecg_buffer, phase: float, heart_rate: float, dt: float = 0.25, fs: int = 100):
    """
    Append new P-QRS-T cardiac waveform samples into the scrolling deque buffer
    scaled dynamically to the live heart rate (BPM).
    """
    num_samples = int(dt * fs)
    safe_hr = max(35.0, min(float(heart_rate), 220.0))
    beat_period = 60.0 / safe_hr
    phase_step = 1.0 / (fs * beat_period)

    for _ in range(num_samples):
        phase = (phase + phase_step) % 1.0

        # Realistic P-QRS-T morphology using Gaussian components
        p_wave = 0.15 * math.exp(-((phase - 0.15) ** 2) / (2 * 0.025 ** 2))
        q_wave = -0.20 * math.exp(-((phase - 0.27) ** 2) / (2 * 0.008 ** 2))
        r_wave = 1.65 * math.exp(-((phase - 0.30) ** 2) / (2 * 0.007 ** 2))
        s_wave = -0.35 * math.exp(-((phase - 0.33) ** 2) / (2 * 0.009 ** 2))
        t_wave = 0.30 * math.exp(-((phase - 0.55) ** 2) / (2 * 0.040 ** 2))
        baseline_noise = np.random.normal(0, 0.015)

        voltage = p_wave + q_wave + r_wave + s_wave + t_wave + baseline_noise
        ecg_buffer.append(voltage)

    return phase