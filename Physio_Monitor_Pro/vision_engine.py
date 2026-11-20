# vision_engine.py
import math
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import requests

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

POSE_MODEL_PATH = os.path.join("models", "pose_landmarker_lite.task")
FACE_MODEL_PATH = os.path.join("models", "face_landmarker.task")

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32),
]


def _ensure_model_file(url: str, dest_path: str) -> bool:
    """Download MediaPipe .task model asset into models/ if not already present."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def init_vision_models():
    """Initialize MediaPipe Tasks PoseLandmarker and FaceLandmarker models."""
    pose_ok = _ensure_model_file(POSE_MODEL_URL, POSE_MODEL_PATH)
    face_ok = _ensure_model_file(FACE_MODEL_URL, FACE_MODEL_PATH)

    pose_landmarker = None
    face_landmarker = None

    if pose_ok:
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    if face_ok:
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

    return pose_landmarker, face_landmarker


def calculate_angle(a, b, c) -> float:
    """Calculate the 2D angle (in degrees) at vertex b given three points a, b, c."""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return round(angle, 1)


def _euclidean(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def analyze_facial_pain(mp_image: mp.Image, w: int, h: int, face_landmarker) -> tuple:
    """
    Estimate a Facial Pain & Distress Score (0-100%) using 478 facial landmarks
    based on brow furrowing, eye narrowing, and mouth tension.
    """
    pain_score = 0.0
    status = "No Face Detected"

    if face_landmarker is None:
        return pain_score, "Face Model Unavailable"

    results = face_landmarker.detect(mp_image)
    if results.face_landmarks:
        landmarks = results.face_landmarks[0]
        pts = [(lm.x * w, lm.y * h) for lm in landmarks]

        # Inter-ocular distance for scale normalization
        eye_span = max(_euclidean(pts[33], pts[263]), 1.0)

        # 1. Inner brow furrow distance (landmarks 55 & 285)
        brow_dist = _euclidean(pts[55], pts[285]) / eye_span
        brow_strain = np.clip((0.36 - brow_dist) / 0.12, 0.0, 1.0)

        # 2. Eye Aspect Ratio (EAR) - squinting under pain
        left_ear = _euclidean(pts[159], pts[145]) / max(_euclidean(pts[33], pts[133]), 1.0)
        right_ear = _euclidean(pts[386], pts[374]) / max(_euclidean(pts[362], pts[263]), 1.0)
        avg_ear = (left_ear + right_ear) / 2.0
        eye_squint = np.clip((0.28 - avg_ear) / 0.12, 0.0, 1.0)

        # 3. Mouth tension / grimace (landmarks 13, 14, 61, 291)
        mouth_open = _euclidean(pts[13], pts[14]) / eye_span
        mouth_width = _euclidean(pts[61], pts[291]) / eye_span
        mouth_strain = np.clip((mouth_open * 1.5 + max(0.0, mouth_width - 0.52)), 0.0, 1.0)

        raw_score = (0.45 * brow_strain + 0.30 * eye_squint + 0.25 * mouth_strain) * 100.0
        pain_score = round(float(np.clip(raw_score, 0.0, 100.0)), 1)

        if pain_score < 30:
            status = "Relaxed / Normal"
        elif pain_score < 60:
            status = "Mild Strain / Effort"
        else:
            status = "High Pain / Distress"

    return pain_score, status


def process_vision_frame(
    frame_bgr: np.ndarray,
    pose_model,
    face_model,
    exercise_type: str,
    rep_state: dict,
) -> tuple:
    """
    Process a single BGR frame for Pose Biomechanics + Facial Pain Index.
    Returns annotated BGR frame and telemetry dictionary.
    """
    h, w, _ = frame_bgr.shape
    image_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    pain_score, pain_status = analyze_facial_pain(mp_image, w, h, face_model)
    annotated_bgr = frame_bgr.copy()

    joint_angle = 0.0
    shoulder_tilt = 0.0
    posture_status = "Body Not Detected"

    if pose_model is not None:
        pose_results = pose_model.detect(mp_image)
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks[0]

            # Draw skeletal connections and joint points
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < len(lm) and end_idx < len(lm):
                    p1 = (int(lm[start_idx].x * w), int(lm[start_idx].y * h))
                    p2 = (int(lm[end_idx].x * w), int(lm[end_idx].y * h))
                    cv2.line(annotated_bgr, p1, p2, (255, 100, 100), 2, cv2.LINE_AA)

            for idx in range(11, min(33, len(lm))):
                pt = (int(lm[idx].x * w), int(lm[idx].y * h))
                cv2.circle(annotated_bgr, pt, 4, (0, 255, 204), -1, cv2.LINE_AA)

            def get_pt(idx):
                return (lm[idx].x * w, lm[idx].y * h)

            # Shoulder alignment / tilt check (Left 11, Right 12)
            l_sh, r_sh = get_pt(11), get_pt(12)
            dx = abs(l_sh[0] - r_sh[0])
            dy = abs(l_sh[1] - r_sh[1])
            shoulder_tilt = round(math.degrees(math.atan2(dy, max(dx, 1.0))), 1)
            posture_status = "Balanced" if shoulder_tilt <= 7.0 else "Asymmetric Tilt"

            # Select joint landmarks based on rehab exercise
            if exercise_type == "Right Elbow Flexion (ثني الكوع الأيمن)":
                a, b, c = get_pt(12), get_pt(14), get_pt(16)  # Shoulder-Elbow-Wrist
                down_thresh, up_thresh = 150.0, 55.0
            elif exercise_type == "Right Shoulder Abduction (رفع الكتف الأيمن)":
                a, b, c = get_pt(24), get_pt(12), get_pt(14)  # Hip-Shoulder-Elbow
                down_thresh, up_thresh = 35.0, 130.0
            else:  # Right Knee Extension / Squat
                a, b, c = get_pt(24), get_pt(26), get_pt(28)  # Hip-Knee-Ankle
                down_thresh, up_thresh = 160.0, 95.0

            joint_angle = calculate_angle(a, b, c)

            # Rep counting state machine
            if "Shoulder" in exercise_type:
                if joint_angle < down_thresh:
                    rep_state["stage"] = "DOWN"
                if joint_angle > up_thresh and rep_state.get("stage") == "DOWN":
                    rep_state["stage"] = "UP"
                    rep_state["count"] = rep_state.get("count", 0) + 1
            else:
                if joint_angle > down_thresh:
                    rep_state["stage"] = "DOWN"
                if joint_angle < up_thresh and rep_state.get("stage") == "DOWN":
                    rep_state["stage"] = "UP"
                    rep_state["count"] = rep_state.get("count", 0) + 1

            # Draw angle label on joint vertex
            cv2.putText(
                annotated_bgr,
                f"{joint_angle} deg",
                (int(b[0]) + 10, int(b[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 204),
                2,
                cv2.LINE_AA,
            )

    # Top HUD Bar on Video Frame
    cv2.rectangle(annotated_bgr, (0, 0), (w, 42), (15, 20, 28), -1)
    hud_text = (
        f"Angle: {joint_angle} deg | Reps: {rep_state.get('count', 0)} | "
        f"Pain: {pain_score}% | Tilt: {shoulder_tilt} deg"
    )
    cv2.putText(
        annotated_bgr,
        hud_text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 204),
        2,
        cv2.LINE_AA,
    )

    telemetry = {
        "joint_angle": joint_angle,
        "shoulder_tilt": shoulder_tilt,
        "posture_status": posture_status,
        "pain_score": pain_score,
        "pain_status": pain_status,
        "reps": rep_state.get("count", 0),
        "stage": rep_state.get("stage", "READY"),
    }
    return annotated_bgr, telemetry


def evaluate_multimodal_clinical_status(vitals: dict, vision_data: dict) -> tuple:
    """
    Fuse Arduino/Live Vitals (HR, SpO2) with Computer Vision (Facial Pain, Posture Tilt)
    to generate a comprehensive clinical rehabilitation alert.
    """
    hr = vitals.get("hr", 80)
    spo2 = vitals.get("spo2", 98)
    pain = vision_data.get("pain_score", 0)
    tilt = vision_data.get("shoulder_tilt", 0)

    if spo2 < 90 or hr > 140 or (pain >= 65 and hr > 115):
        return (
            "#ff4444",
            "🚨 إنذار حرج (إيقاف التمرين فوراً)",
            "رصدت الكاميرا علامات ألم شديدة بالتزامن مع تسارع النبض أو نقص الأكسجين. يجب إراحة المريض فوراً.",
        )
    elif pain >= 45 or tilt > 10.0 or hr > 115:
        return (
            "#ffcc00",
            "⚠️ تنبيه إجهاد / تصحيح وضعية",
            "يوجد إجهاد عضلي ملحوظ أو عدم اتزان في ميلان الكتفين أثناء الحركة. يُنصح بتقليل شدة التمرين وتعديل الوضعية.",
        )
    else:
        return (
            "#00ffcc",
            "✅ أداء تأهيلي آمن ومستقر",
            "المؤشرات الحيوية من الأردوينو وحركة المفاصل وتعبيرات الوجه جميعها في النطاق الآمن والمستقر.",
        )