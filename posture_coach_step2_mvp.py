import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# -----------------------------
# 각도 계산 함수
# -----------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle


# -----------------------------
# MediaPipe 초기화
# -----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 창 이름 및 크기 설정 (모서리로 자유 조절 가능)
window_name = "Real-time AI Coach - MVP"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(window_name, 960, 720)  # 필요하면 기본 크기 지정

# -----------------------------
# 자세 기준값 및 기타 설정
# -----------------------------
NECK_ANGLE_THRESHOLD = 50.0   # 거북목: CVA < 50도
SLUMP_ANGLE_THRESHOLD = 90.0  # 굽은 허리: 몸통-허벅지 < 90도

VISIBILITY_THRESHOLD = 0.7
SIDE_VIEW_Z_THRESHOLD = 0.15

neck_angles_deque = deque(maxlen=5)
slumped_angles_deque = deque(maxlen=5)

# NameError 방지용 초기값
avg_neck_angle = 0.0
avg_slumped_angle = 0.0

# -----------------------------
# 메인 루프
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 거울처럼 보이게 좌우 반전
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # 포즈 추론
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True

    # 상태 기본값
    status = "Detecting..."
    status_color = (200, 200, 200)   # 텍스트 색
    border_color = (200, 200, 200)   # 테두리 색

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        LE = mp_pose.PoseLandmark.LEFT_EAR.value
        RE = mp_pose.PoseLandmark.RIGHT_EAR.value
        LH = mp_pose.PoseLandmark.LEFT_HIP.value
        RH = mp_pose.PoseLandmark.RIGHT_HIP.value
        LK = mp_pose.PoseLandmark.LEFT_KNEE.value
        RK = mp_pose.PoseLandmark.RIGHT_KNEE.value

        left_shoulder_z = landmarks[LS].z
        right_shoulder_z = landmarks[RS].z
        is_frontal_view = abs(left_shoulder_z - right_shoulder_z) < SIDE_VIEW_Z_THRESHOLD

        slumped_angles = []
        neck_angles = []

        # ----- 왼쪽 관절 -----
        if landmarks[LS].visibility > VISIBILITY_THRESHOLD:
            if is_frontal_view or left_shoulder_z < right_shoulder_z:
                left_shoulder = (landmarks[LS].x * w, landmarks[LS].y * h)
                left_ear = (landmarks[LE].x * w, landmarks[LE].y * h)
                left_hip = (landmarks[LH].x * w, landmarks[LH].y * h)
                left_knee = (landmarks[LK].x * w, landmarks[LK].y * h)

                # 몸통-허벅지 각도
                slumped_angles.append(
                    calculate_angle(left_shoulder, left_hip, left_knee)
                )
                # 목(CVA) 각도
                neck_angles.append(
                    calculate_angle(
                        (left_shoulder[0] - 100, left_shoulder[1]),
                        left_shoulder,
                        left_ear
                    )
                )

                cv2.line(
                    frame,
                    tuple(np.array(left_shoulder, dtype=int)),
                    tuple(np.array(left_ear, dtype=int)),
                    (0, 255, 0),
                    2
                )

        # ----- 오른쪽 관절 -----
        if landmarks[RS].visibility > VISIBILITY_THRESHOLD:
            if is_frontal_view or right_shoulder_z < left_shoulder_z:
                right_shoulder = (landmarks[RS].x * w, landmarks[RS].y * h)
                right_ear = (landmarks[RE].x * w, landmarks[RE].y * h)
                right_hip = (landmarks[RH].x * w, landmarks[RH].y * h)
                right_knee = (landmarks[RK].x * w, landmarks[RK].y * h)

                slumped_angles.append(
                    calculate_angle(right_shoulder, right_hip, right_knee)
                )
                neck_angles.append(
                    calculate_angle(
                        (right_shoulder[0] - 100, right_shoulder[1]),
                        right_shoulder,
                        right_ear
                    )
                )

                cv2.line(
                    frame,
                    tuple(np.array(right_shoulder, dtype=int)),
                    tuple(np.array(right_ear, dtype=int)),
                    (0, 255, 0),
                    2
                )

        # ----- 스무딩 -----
        if slumped_angles:
            slumped_angles_deque.append(np.mean(slumped_angles))
        if neck_angles:
            neck_angles_deque.append(np.mean(neck_angles))

        avg_slumped_angle = np.mean(slumped_angles_deque) if len(slumped_angles_deque) > 0 else 0.0
        avg_neck_angle = np.mean(neck_angles_deque) if len(neck_angles_deque) > 0 else 0.0

        # ----- 상태 판별 -----
        if avg_neck_angle > 0 and avg_neck_angle < NECK_ANGLE_THRESHOLD:
            status = "NECK WARN!"
            status_color = (0, 0, 255)      # 빨간 텍스트
            border_color = (0, 0, 255)      # 빨간 테두리
        elif avg_slumped_angle > 0 and avg_slumped_angle < SLUMP_ANGLE_THRESHOLD:
            status = "SLUMP WARN!"
            status_color = (0, 0, 255)
            border_color = (0, 0, 255)
        elif avg_neck_angle > 0 or avg_slumped_angle > 0:
            status = "GOOD"
            status_color = (0, 255, 0)      # 초록 텍스트
            border_color = (0, 255, 0)      # 초록 테두리

        # 포즈 뼈대
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # -----------------------------
    # 왼쪽 위: 각도 + 기준값 표시
    # -----------------------------
    cv2.putText(
        frame,
        f"Neck(CVA): {avg_neck_angle:.1f} deg [>=50 good]",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Back(Torso-Thigh): {avg_slumped_angle:.1f} deg [>=90 good]",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # -----------------------------
    # 오른쪽 위: 상태 텍스트
    # -----------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 4

    (text_width, text_height), _ = cv2.getTextSize(status, font, font_scale, thickness)
    margin_right = 20
    x = w - text_width - margin_right
    y = 140  # 각도 텍스트보다 아래쪽

    cv2.putText(
        frame,
        status,
        (x, y),
        font,
        font_scale,
        status_color,   # GOOD=초록, WARN=빨강, Detecting=회색
        thickness,
        cv2.LINE_AA
    )

    # -----------------------------
    # 화면 전체 테두리 그리기 (정상/비정상 색상)
    # -----------------------------
    margin = 5
    border_thickness = 6
    cv2.rectangle(
        frame,
        (margin, margin),
        (w - margin, h - margin),
        border_color,
        border_thickness
    )

    # -----------------------------
    # 창 띄우기
    # -----------------------------
    cv2.imshow(window_name, frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC 키
        break

cap.release()
cv2.destroyAllWindows()