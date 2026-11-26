import os
import math
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
]

FACE_OVAL_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FACE_BROW_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
FACE_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
FACE_MOUTH_NOSE_IDX = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


def _resolve_model(filename: str) -> str:
    for path in [filename, os.path.join("models", filename), os.path.join("assets", filename)]:
        if os.path.exists(path):
            return path
    return filename


def create_landmarkers():
    """Initialize MediaPipe PoseLandmarker and FaceLandmarker if task files exist."""
    pose_landmarker = None
    face_landmarker = None
    if mp is None or not hasattr(mp, "tasks"):
        return None, None

    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        pose_path = _resolve_model("pose_landmarker_lite.task")
        face_path = _resolve_model("face_landmarker.task")

        if os.path.exists(pose_path):
            pose_opts = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=pose_path),
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)

        if os.path.exists(face_path):
            face_opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_path),
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_opts)
    except Exception:
        pass

    return pose_landmarker, face_landmarker


def compute_angle(a, b, c) -> float:
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    return 360.0 - angle if angle > 180.0 else angle


def get_exercise_config(exercise_type: str, side: str):
    is_left = (side == "Left Side")
    if "Elbow" in exercise_type:
        tri = (11, 13, 15) if is_left else (12, 14, 16)
        return tri, 50.0, 145.0, "flex_down_to_up"
    if "Shoulder Abduction" in exercise_type:
        tri = (23, 11, 13) if is_left else (24, 12, 14)
        return tri, 135.0, 35.0, "raise_low_to_high"
    if "Shoulder Forward" in exercise_type:
        tri = (23, 11, 15) if is_left else (24, 12, 16)
        return tri, 140.0, 35.0, "raise_low_to_high"
    if "Knee Extension" in exercise_type:
        tri = (23, 25, 27) if is_left else (24, 26, 28)
        return tri, 160.0, 95.0, "raise_low_to_high"
    if "Squat" in exercise_type:
        tri = (23, 25, 27) if is_left else (24, 26, 28)
        return tri, 100.0, 160.0, "flex_down_to_up"
    return (11, 0, 12), 85.0, 95.0, "posture_only"


def analyze_clinical_frame(
    frame_bgr,
    pose_landmarker,
    face_landmarker,
    exercise_type: str,
    tracking_side: str,
    draw_body: bool,
    draw_face: bool,
    metrics: dict,
    target_reps: int = 10,
):
    """Detect body skeleton & facial pain landmarks on a single BGR frame (photo or live)."""
    if cv2 is None or frame_bgr is None:
        return frame_bgr, metrics

    annotated = frame_bgr.copy()
    rgb = np.ascontiguousarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    h, w, _ = annotated.shape

    active_tri, up_thresh, down_thresh, rep_mode = get_exercise_config(exercise_type, tracking_side)
    pose_lms = None
    face_lms = None

    # 1. Body Pose Landmarks (33 points)
    if pose_landmarker is not None and mp is not None:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_res = pose_landmarker.detect(mp_img)
        if pose_res.pose_landmarks:
            pose_lms = pose_res.pose_landmarks[0]
            a_i, b_i, c_i = active_tri
            pa = (pose_lms[a_i].x * w, pose_lms[a_i].y * h)
            pb = (pose_lms[b_i].x * w, pose_lms[b_i].y * h)
            pc = (pose_lms[c_i].x * w, pose_lms[c_i].y * h)

            angle = compute_angle(pa, pb, pc)
            p11 = (pose_lms[11].x * w, pose_lms[11].y * h)
            p12 = (pose_lms[12].x * w, pose_lms[12].y * h)
            tilt = abs(math.degrees(math.atan2(p12[1] - p11[1], max(1.0, abs(p12[0] - p11[0])))))

            metrics["angle"] = round(angle, 1)
            metrics["tilt"] = round(tilt, 1)

            if rep_mode == "flex_down_to_up":
                if angle > down_thresh:
                    metrics["stage"] = "DOWN"
                elif angle < up_thresh and metrics["stage"] == "DOWN":
                    metrics["stage"] = "UP"
                    metrics["reps"] += 1
            elif rep_mode == "raise_low_to_high":
                if angle < down_thresh:
                    metrics["stage"] = "DOWN"
                elif angle > up_thresh and metrics["stage"] == "DOWN":
                    metrics["stage"] = "UP"
                    metrics["reps"] += 1

    # 2. Facial Pain & Expression Landmarks (478 points)
    if face_landmarker is not None and mp is not None:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        face_res = face_landmarker.detect(mp_img)
        if face_res.face_landmarks:
            face_lms = face_res.face_landmarks[0]
            eye_span = max(1e-4, math.hypot(face_lms[33].x - face_lms[263].x, face_lms[33].y - face_lms[263].y))
            brow_dist = math.hypot(face_lms[107].x - face_lms[336].x, face_lms[107].y - face_lms[336].y) / eye_span
            eye_open = math.hypot(face_lms[159].x - face_lms[145].x, face_lms[159].y - face_lms[145].y) / eye_span
            mouth_stretch = math.hypot(face_lms[61].x - face_lms[291].x, face_lms[61].y - face_lms[291].y) / eye_span

            brow_strain = np.clip((0.36 - brow_dist) / 0.12, 0.0, 1.0)
            eye_squint = np.clip((0.11 - eye_open) / 0.06, 0.0, 1.0)
            mouth_tension = np.clip((mouth_stretch - 0.52) / 0.18, 0.0, 1.0)
            metrics["pain"] = round(float((0.45 * brow_strain + 0.30 * eye_squint + 0.25 * mouth_tension) * 100.0), 1)

    # 3. Draw Skeleton & Facial Landmarks
    if draw_body and pose_lms is not None:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_lms]
        for idx1, idx2 in POSE_CONNECTIONS:
            if idx1 < len(pts) and idx2 < len(pts):
                cv2.line(annotated, pts[idx1], pts[idx2], (255, 200, 0), 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(annotated, pt, 5, (16, 185, 129), -1, cv2.LINE_AA)
            cv2.circle(annotated, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)

        a_i, b_i, c_i = active_tri
        if max(a_i, b_i, c_i) < len(pts):
            cv2.line(annotated, pts[a_i], pts[b_i], (16, 185, 129), 4, cv2.LINE_AA)
            cv2.line(annotated, pts[b_i], pts[c_i], (16, 185, 129), 4, cv2.LINE_AA)
            for j_i in (a_i, b_i, c_i):
                cv2.circle(annotated, pts[j_i], 8, (0, 230, 255), -1, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"{metrics['angle']:.1f} deg",
                (pts[b_i][0] + 10, pts[b_i][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 200),
                2,
                cv2.LINE_AA,
            )

    if draw_face and face_lms is not None:
        fpts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lms]
        p_val = metrics["pain"]
        p_color = (68, 68, 239) if p_val >= 65.0 else ((11, 158, 245) if p_val >= 35.0 else (129, 185, 16))
        for idx in FACE_OVAL_IDX:
            if idx < len(fpts):
                cv2.circle(annotated, fpts[idx], 2, (200, 220, 255), -1, cv2.LINE_AA)
        for idx in FACE_BROW_IDX:
            if idx < len(fpts):
                cv2.circle(annotated, fpts[idx], 3, p_color, -1, cv2.LINE_AA)
        for idx in FACE_EYE_IDX:
            if idx < len(fpts):
                cv2.circle(annotated, fpts[idx], 2, (0, 255, 255), -1, cv2.LINE_AA)
        for idx in FACE_MOUTH_NOSE_IDX:
            if idx < len(fpts):
                cv2.circle(annotated, fpts[idx], 2, p_color, -1, cv2.LINE_AA)

    # Top HUD Overlay
    cv2.rectangle(annotated, (0, 0), (w, 44), (15, 23, 35), -1)
    hud = (
        f"{exercise_type} ({tracking_side.split()[0]}) | "
        f"Reps: {metrics['reps']}/{target_reps} | Angle: {metrics['angle']:.1f} | Pain: {metrics['pain']:.0f}%"
    )
    cv2.putText(annotated, hud, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (16, 185, 129), 2, cv2.LINE_AA)

    return annotated, metrics