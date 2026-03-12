import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import threading
import csv
import os
from datetime import datetime

# ----- (옵션) 윈도우 소리 알림용 -----
try:
    import winsound
    USE_SOUND = True
except ImportError:
    winsound = None
    USE_SOUND = False

# ----- (옵션) TTS(음성) 알림용 -----
USE_TTS = False
try:
    import pyttsx3
    USE_TTS = True
except ImportError:
    pyttsx3 = None
    USE_TTS = False


def speak_async(text: str):
    """
    OpenCV 메인 루프를 멈추지 않고 TTS 실행.
    매번 새 pyttsx3 엔진을 만들어서 사용 → 내부 run loop 꼬임 방지.
    """
    if not USE_TTS or pyttsx3 is None:
        return

    def _run():
        try:
            engine = pyttsx3.init()

            # 한국어 음성 선택 시도
            voices = engine.getProperty("voices")
            for v in voices:
                vid = getattr(v, "id", "")
                name = getattr(v, "name", "")
                if "korean" in name.lower() or "ko_" in vid.lower():
                    engine.setProperty("voice", v.id)
                    break

            engine.setProperty("rate", 180)  # 말 속도

            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("TTS 오류:", e)

    threading.Thread(target=_run, daemon=True).start()


# -----------------------------
# 유틸: 초 → MM:SS 포맷
# -----------------------------
def format_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


# -----------------------------
# 각도 계산 함수
# -----------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


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
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    raise SystemExit

window_name = "Real-time AI Coach - MVP 3 (Timer + Alert + TTS + Log)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# -----------------------------
# 설정 파라미터
# -----------------------------
NECK_ANGLE_THRESHOLD = 50.0       # CVA 50도 미만이면 나쁜 자세
SLUMP_ANGLE_THRESHOLD = 90.0      # 90도 미만이면 허리 굽음

VISIBILITY_THRESHOLD = 0.5
SIDE_VIEW_Z_THRESHOLD = 0.10      # 어깨 Z 차이로 정면 vs 측면 판단

BAD_POSTURE_ALERT_SEC = 3.0       # 3초 이상 나쁜 자세면 경고
SOUND_ALERT_DELAY_SEC = 2.0       # 경고 진입 후 2초 뒤부터 소리 시작
SOUND_INTERVAL_SEC = 1.0          # 소리 반복 간격

LOG_FILE = "posture_log.csv"      # 사용 로그 저장 파일

start_time = time.time()
last_time = start_time

status = "Detecting..."
status_color = (255, 255, 255)
border_color = (255, 255, 255)

# 각도 스무딩용 히스토리
ANGLE_HISTORY_SIZE = 10
neck_angle_history = deque(maxlen=ANGLE_HISTORY_SIZE)
slumped_angle_history = deque(maxlen=ANGLE_HISTORY_SIZE)

# 나쁜 자세 시간 및 알림 상태
bad_posture_start_time = None     # 나쁜 자세 연속 구간 시작 시각
bad_posture_duration = 0.0        # 현재 연속 나쁜 자세 유지 시간

first_alert_time = None           # 경고 구간(3초 넘은 후) 진입 시각
last_sound_time = None            # 마지막 Beep 시각
tts_spoken = False                # 이번 나쁜 자세 에피소드에서 TTS를 이미 했는지 여부

last_status = "Detecting..."

# -----------------------------
# 세션 통계: 시간/카운트
# -----------------------------
session_good_time = 0.0         # GOOD 상태 누적 시간
session_bad_time = 0.0          # NECK/SLUMP 나쁜 자세 누적 시간
session_noperson_time = 0.0     # NO PERSON / Detecting 등 유효 자세 아님

neck_warn_count = 0
slump_warn_count = 0

# -----------------------------
# 메인 루프
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 좌우 반전 (거울처럼 보이게)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Mediapipe 입력 준비
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True

    # 기본 상태
    status = "Detecting..."
    status_color = (255, 255, 255)
    border_color = (255, 255, 255)
    is_bad_posture = False

    now = time.time()
    elapsed_time = now - start_time
    dt = now - last_time
    if dt < 0:
        dt = 0.0

    avg_neck_angle = 0.0
    avg_slumped_angle = 0.0

    # -----------------------------
    # 관절 및 각도 계산
    # -----------------------------
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

        neck_angles = []
        slumped_angles = []

        # ----- 왼쪽 측 -----
        if landmarks[LS].visibility > VISIBILITY_THRESHOLD:
            if is_frontal_view or left_shoulder_z < right_shoulder_z:
                left_shoulder = (landmarks[LS].x * w, landmarks[LS].y * h)
                left_ear = (landmarks[LE].x * w, landmarks[LE].y * h)
                left_hip = (landmarks[LH].x * w, landmarks[LH].y * h)
                left_knee = (landmarks[LK].x * w, landmarks[LK].y * h)

                slumped_angles.append(
                    calculate_angle(left_shoulder, left_hip, left_knee)
                )
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

        # ----- 오른쪽 측 -----
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
                        (right_shoulder[0] + 100, right_shoulder[1]),
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

        # 각도 스무딩
        if neck_angles:
            neck_angle_history.append(np.mean(neck_angles))
        if slumped_angles:
            slumped_angle_history.append(np.mean(slumped_angles))

        avg_neck_angle = float(np.mean(neck_angle_history)) if neck_angle_history else 0.0
        avg_slumped_angle = float(np.mean(slumped_angle_history)) if slumped_angle_history else 0.0

        # -----------------------------
        # 자세 상태 판별
        # -----------------------------
        if avg_neck_angle > 0 and avg_neck_angle < NECK_ANGLE_THRESHOLD:
            status = "NECK WARN!"
            status_color = (0, 0, 255)
            border_color = (0, 0, 255)
            is_bad_posture = True
        elif avg_slumped_angle > 0 and avg_slumped_angle < SLUMP_ANGLE_THRESHOLD:
            status = "SLUMP WARN!"
            status_color = (0, 0, 255)
            border_color = (0, 0, 255)
            is_bad_posture = True
        elif avg_neck_angle > 0 or avg_slumped_angle > 0:
            status = "GOOD"
            status_color = (0, 255, 0)
            border_color = (0, 255, 0)
            is_bad_posture = False

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    else:
        status = "NO PERSON"
        status_color = (200, 200, 200)
        border_color = (200, 200, 200)
        is_bad_posture = False

    # -----------------------------
    # 세션 시간 누적 (GOOD / BAD / NO PERSON)
    # -----------------------------
    if results.pose_landmarks:
        if is_bad_posture:
            session_bad_time += dt
        elif status == "GOOD":
            session_good_time += dt
        else:
            session_noperson_time += dt
    else:
        session_noperson_time += dt

    # -----------------------------
    # 경고 카운트 (상태 전이 감지)
    # -----------------------------
    if status == "NECK WARN!" and last_status != "NECK WARN!":
        neck_warn_count += 1
    elif status == "SLUMP WARN!" and last_status != "SLUMP WARN!":
        slump_warn_count += 1

    # -----------------------------
    # 나쁜 자세 연속 시간 & 에피소드 상태 관리
    # -----------------------------
    if is_bad_posture:
        if bad_posture_start_time is None:
            # 새로운 나쁜 자세 에피소드 시작
            bad_posture_start_time = now
            tts_spoken = False          # 이 에피소드에서는 아직 TTS 안 함
            first_alert_time = None
            last_sound_time = None

        bad_posture_duration = now - bad_posture_start_time
    else:
        # 좋은 자세(또는 사람 없음) → 연속 나쁜 자세 관련 상태만 리셋
        bad_posture_start_time = None
        bad_posture_duration = 0.0
        tts_spoken = False
        first_alert_time = None
        last_sound_time = None

    # -----------------------------
    # 임계 시간 이상 나쁜 자세일 때 경고 (TTS 1회 + 소리 + 박스)
    # -----------------------------
    if is_bad_posture and bad_posture_duration >= BAD_POSTURE_ALERT_SEC:

        # 1) TTS: 이번 에피소드에서 아직 안 했으면 딱 한 번만
        if not tts_spoken:
            if status == "NECK WARN!":
                msg = "목이 앞으로 나왔습니다. 턱을 당기고 목을 세워 주세요."
            elif status == "SLUMP WARN!":
                msg = "허리가 앞으로 굽었습니다. 허리를 펴 주세요."
            else:
                msg = "나쁜 자세가 계속되고 있습니다. 자세를 고쳐 주세요."
            speak_async(msg)
            tts_spoken = True

        # 2) 경고 구간 진입 시각 기록 (비프 시작 기준)
        if first_alert_time is None:
            first_alert_time = now
            last_sound_time = None

        # 3) 화면 중앙 경고 박스
        overlay = frame.copy()
        alpha = 0.6
        cv2.rectangle(
            overlay,
            (int(w * 0.1), int(h * 0.3)),
            (int(w * 0.9), int(h * 0.7)),
            (0, 0, 255),
            -1
        )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        alert_font_scale = 1.0
        alert_thickness = 3
        line1 = f"Bad posture over {int(BAD_POSTURE_ALERT_SEC)}s!"
        line2 = "Straighten your back!"

        (w1, h1), _ = cv2.getTextSize(line1, font, alert_font_scale, alert_thickness)
        (w2, h2), _ = cv2.getTextSize(line2, font, alert_font_scale, alert_thickness)
        text_w = max(w1, w2)
        line_gap = 15

        x_text = (w - text_w) // 2
        y_text = h // 2 - (h1 + line_gap // 2)

        cv2.putText(
            frame,
            line1,
            (x_text, y_text),
            font,
            alert_font_scale,
            (255, 255, 255),
            alert_thickness,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            line2,
            (x_text, y_text + h1 + line_gap),
            font,
            alert_font_scale,
            (255, 255, 255),
            alert_thickness,
            cv2.LINE_AA
        )

        # 4) 비프 소리: 나쁜 자세 유지되는 동안 일정 간격으로 반복
        if USE_SOUND and winsound is not None:
            if (now - first_alert_time) >= SOUND_ALERT_DELAY_SEC:
                if (last_sound_time is None) or (now - last_sound_time >= SOUND_INTERVAL_SEC):
                    winsound.Beep(1000, 500)  # 1000Hz, 0.5초
                    last_sound_time = now

    # -----------------------------
    # 각도 및 타이머 정보 출력
    # -----------------------------
    cv2.putText(
        frame,
        f"Neck(CVA): {avg_neck_angle:5.1f} deg [>=50 good]",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Back(Torso-Thigh): {avg_slumped_angle:5.1f} deg [>=90 good]",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Bad Posture Time: {bad_posture_duration:5.1f} s (alert>={BAD_POSTURE_ALERT_SEC:.0f}s)",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # 세션 전체 타이머 / 통계
    session_time_str = format_time(elapsed_time)
    good_time_str = format_time(session_good_time)
    bad_time_str = format_time(session_bad_time)
    noperson_time_str = format_time(session_noperson_time)

    cv2.putText(
        frame,
        f"Session: {session_time_str}",
        (10, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 255, 200),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Good: {good_time_str}   Bad: {bad_time_str}",
        (10, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 255, 200),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"No Person: {noperson_time_str}",
        (10, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 255, 200),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"NECK_WARN: {neck_warn_count}   SLUMP_WARN: {slump_warn_count}",
        (10, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # -----------------------------
    # 아래 중앙 상태 텍스트
    # -----------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    status_font_scale = 2.0
    status_thickness = 4

    (text_width, text_height), _ = cv2.getTextSize(
        status, font, status_font_scale, status_thickness
    )
    x_status = (w - text_width) // 2
    margin_bottom = 25
    y_status = h - margin_bottom

    cv2.putText(
        frame,
        status,
        (x_status, y_status),
        font,
        status_font_scale,
        status_color,
        status_thickness,
        cv2.LINE_AA
    )

    # -----------------------------
    # 화면 테두리
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

    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC
        last_time = now
        last_status = status
        break

    last_time = now
    last_status = status

# -----------------------------
# 세션 로그 저장
# -----------------------------
total_session_time = time.time() - start_time
tracked_time = session_good_time + session_bad_time
good_ratio = (session_good_time / tracked_time * 100.0) if tracked_time > 0 else 0.0

try:
    log_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow([
                "datetime",
                "session_time_sec",
                "good_time_sec",
                "bad_time_sec",
                "noperson_time_sec",
                "good_ratio_percent",
                "neck_warn_count",
                "slump_warn_count",
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{total_session_time:.1f}",
            f"{session_good_time:.1f}",
            f"{session_bad_time:.1f}",
            f"{session_noperson_time:.1f}",
            f"{good_ratio:.1f}",
            neck_warn_count,
            slump_warn_count,
        ])
    print(f"세션 로그가 '{LOG_FILE}' 파일에 저장되었습니다.")
except Exception as e:
    print("로그 저장 중 오류:", e)

# -----------------------------
# 세션 요약 오버레이 (새 창)
# -----------------------------
summary_h = 400
summary_w = 800
summary = np.zeros((summary_h, summary_w, 3), dtype=np.uint8)

# 반투명 배경 박스
overlay = summary.copy()
cv2.rectangle(
    overlay,
    (40, 40),
    (summary_w - 40, summary_h - 40),
    (50, 50, 50),
    -1
)
cv2.addWeighted(overlay, 0.8, summary, 0.2, 0, summary)

font = cv2.FONT_HERSHEY_SIMPLEX

title = "Session Summary"
cv2.putText(
    summary,
    title,
    (int(summary_w * 0.23), 90),
    font,
    1.3,
    (255, 255, 255),
    3,
    cv2.LINE_AA
)

line_y = 150
line_gap = 35

cv2.putText(
    summary,
    f"Total Session : {format_time(total_session_time)}",
    (80, line_y),
    font,
    0.9,
    (200, 255, 200),
    2,
    cv2.LINE_AA
)
cv2.putText(
    summary,
    f"Good Posture  : {format_time(session_good_time)}",
    (80, line_y + line_gap),
    font,
    0.9,
    (200, 255, 200),
    2,
    cv2.LINE_AA
)
cv2.putText(
    summary,
    f"Bad Posture   : {format_time(session_bad_time)}",
    (80, line_y + 2 * line_gap),
    font,
    0.9,
    (0, 255, 255),
    2,
    cv2.LINE_AA
)
cv2.putText(
    summary,
    f"No Person     : {format_time(session_noperson_time)}",
    (80, line_y + 3 * line_gap),
    font,
    0.9,
    (180, 180, 255),
    2,
    cv2.LINE_AA
)
cv2.putText(
    summary,
    f"Good Ratio    : {good_ratio:5.1f} %",
    (80, line_y + 4 * line_gap),
    font,
    0.9,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)
cv2.putText(
    summary,
    f"NECK_WARN: {neck_warn_count}   SLUMP_WARN: {slump_warn_count}",
    (80, line_y + 5 * line_gap),
    font,
    0.9,
    (0, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    summary,
    "Press any key to exit",
    (summary_w - 350, summary_h - 30),
    font,
    0.7,
    (200, 200, 200),
    2,
    cv2.LINE_AA
)

cv2.imshow("Session Summary", summary)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()