import time
import serial
import serial.tools.list_ports


def list_available_ports() -> list[str]:
    """Return available serial COM ports sorted with COM6 prioritized if present."""
    ports = [p.device for p in serial.tools.list_ports.comports()]
    if not ports:
        return ["COM6", "COM4", "COM3"]
    if "COM6" in ports:
        ports.remove("COM6")
        ports.insert(0, "COM6")
    return ports


def open_serial_port(port: str, baud_rate: int, existing_ser=None):
    """Safely close any stale serial handle and open the requested COM port."""
    if existing_ser is not None:
        try:
            if existing_ser.is_open:
                existing_ser.close()
                time.sleep(0.15)
        except Exception:
            pass

    try:
        ser = serial.Serial(
            port=port,
            baudrate=int(baud_rate),
            timeout=0.05,
            write_timeout=0.5,
        )
        time.sleep(0.2)
        ser.reset_input_buffer()
        return ser, ""
    except Exception as exc:
        return None, (
            f"تعذر فتح المنفذ '{port}' (تأكد من إغلاق Serial Monitor في Arduino IDE). "
            f"التفاصيل: {exc}"
        )


def close_serial_port(ser) -> None:
    """Safely close an active serial connection."""
    if ser is not None:
        try:
            if ser.is_open:
                ser.close()
        except Exception:
            pass


def read_arduino_vitals(ser, current_vitals: dict) -> dict:
    """Read real-time ESP32/Arduino vitals and reset immediately when finger is lifted."""
    updated = current_vitals.copy()
    if ser is None or not ser.is_open:
        return updated

    try:
        while ser.in_waiting > 0:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                continue

            if raw_line == "NO_SENSOR":
                updated["sensor_status"] = "NO_SENSOR"
                updated["finger_detected"] = False
                updated["hr"] = 0.0
                updated["spo2"] = 0.0
                continue

            parts = [p.strip() for p in raw_line.split(",")]
            if len(parts) >= 2:
                hr = float(parts[0])
                spo2 = float(parts[1])

                if hr == 0.0 or spo2 == 0.0:
                    updated["sensor_status"] = "WAITING_FINGER"
                    updated["finger_detected"] = False
                    updated["hr"] = 0.0
                    updated["spo2"] = 0.0
                elif 45.0 <= hr <= 190.0 and 75.0 <= spo2 <= 100.0:
                    updated["sensor_status"] = "ACTIVE"
                    updated["finger_detected"] = True
                    updated["hr"] = round(hr, 1)
                    updated["spo2"] = round(spo2, 1)
                    if len(parts) >= 3 and 34.0 <= float(parts[2]) <= 42.0:
                        updated["temp"] = round(float(parts[2]), 1)
    except Exception:
        pass

    return updated