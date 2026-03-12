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
    """OpenCV 메인 루프를 멈추지 않고 TTS 실행."""
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

# -----------------------------
# 입력 소스 선택: OAK-D Pro(DepthAI) / 일반 웹캠(OpenCV)
#  - OAK-D Pro 사용 시: pip install depthai
# -----------------------------
USE_OAKD_PRO = True  # OAK-D Pro로 사용하려면 True, 일반 웹캠은 False

class OakCapture:
    """cv2.VideoCapture처럼 read()/isOpened()/release()를 제공하는 OAK 캡처 래퍼."""

    def __init__(self, preview_size=(960, 540), fps=30, blocking=True):
        import depthai as dai

        self._dai = dai
        self._opened = False

        pipeline = dai.Pipeline()

        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutRgb = pipeline.createXLinkOut()
        xoutRgb.setStreamName("rgb")

        # Properties (필요하면 해상도/프레임 바꿔도 됨)
        camRgb.setPreviewSize(preview_size[0], preview_size[1])
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        camRgb.setFps(fps)

        camRgb.preview.link(xoutRgb.input)

        # Device start
        self._device = dai.Device(pipeline)
        self._qRgb = self._device.getOutputQueue(name="rgb", maxSize=4, blocking=blocking)
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        """(ret, frame[BGR])"""
        if not self._opened:
            return False, None
        try:
            inRgb = self._qRgb.get()  # 새 프레임 대기(=웹캠 read()와 동일한 느낌)
            frame = inRgb.getCvFrame()  # OpenCV 호환 BGR 프레임
            return True, frame
        except Exception as e:
            print("OAK frame read error:", e)
            self._opened = False
            return False, None

    def release(self):
        self._opened = False
        try:
            self._device.close()
        except Exception:
            pass

if USE_OAKD_PRO:
    try:
        cap = OakCapture(preview_size=(960, 540), fps=30, blocking=True)
    except ImportError:
        print("depthai가 설치되어 있지 않습니다.  (pip install depthai)")
        raise SystemExit
    except Exception as e:
        print("OAK-D Pro를 열 수 없습니다:", e)
        raise SystemExit
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        raise SystemExit

window_cam = "Real-time AI Coach - Camera"
window_hud = "Real-time AI Coach - HUD"
cv2.namedWindow(window_cam, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_hud, cv2.WINDOW_NORMAL)

# 창 위치(원하면 드래그로 옮기면 됨)
try:
    cv2.moveWindow(window_cam, 50, 50)
    cv2.moveWindow(window_hud, 700, 50)
except Exception:
    pass

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


# -----------------------------
# 성공(복귀) 애니메이션 설정
# -----------------------------
SUCCESS_ANIM_SEC = 1.2      # 성공 애니메이션 지속 시간(초)
GOOD_STABLE_SEC = 0.3       # GOOD가 이 시간 이상 유지되면 "복귀"로 인정(깜빡임 방지)

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
bad_posture_start_time = None
bad_posture_duration = 0.0

first_alert_time = None
last_sound_time = None
tts_spoken = False

last_status = "Detecting..."


# -----------------------------
# "경고 상태였다가 GOOD로 복귀" 감지용
# -----------------------------
was_alerting = False        # 실제로 경고 단계(>=BAD_POSTURE_ALERT_SEC)에 들어간 적 있는지
good_start_time = None      # GOOD 유지 시작 시각
success_anim_start = None   # 성공 애니메이션 시작 시각

# -----------------------------
# 세션 통계: 시간/카운트
# -----------------------------
session_good_time = 0.0
session_bad_time = 0.0
session_noperson_time = 0.0

neck_warn_count = 0
slump_warn_count = 0

# -----------------------------
# 메인 루프
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True

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

        # 왼쪽 측
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

        # 오른쪽 측
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

        if neck_angles:
            neck_angle_history.append(np.mean(neck_angles))
        if slumped_angles:
            slumped_angle_history.append(np.mean(slumped_angles))

        avg_neck_angle = float(np.mean(neck_angle_history)) if neck_angle_history else 0.0
        avg_slumped_angle = float(np.mean(slumped_angle_history)) if slumped_angle_history else 0.0

        # 자세 상태 판별
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

        # 카메라 창에는 포즈 점/선만
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
    # 세션 시간 누적
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
    # 경고 카운트 (상태 전이)
    # -----------------------------
    if status == "NECK WARN!" and last_status != "NECK WARN!":
        neck_warn_count += 1
    elif status == "SLUMP WARN!" and last_status != "SLUMP WARN!":
        slump_warn_count += 1

    # -----------------------------
    # 나쁜 자세 연속 시간 & 에피소드 상태
    # -----------------------------
    if is_bad_posture:
        if bad_posture_start_time is None:
            bad_posture_start_time = now
            tts_spoken = False
            first_alert_time = None
            last_sound_time = None

        bad_posture_duration = now - bad_posture_start_time
    else:
        bad_posture_start_time = None
        bad_posture_duration = 0.0
        tts_spoken = False
        first_alert_time = None
        last_sound_time = None

    # -----------------------------
    # 임계 시간 이상 나쁜 자세일 때 (TTS + 비프)
    # -----------------------------
    if is_bad_posture and bad_posture_duration >= BAD_POSTURE_ALERT_SEC:

        if not tts_spoken:
            if status == "NECK WARN!":
                msg = "목이 앞으로 나왔습니다. 턱을 당기고 목을 세워 주세요."
            elif status == "SLUMP WARN!":
                msg = "허리가 앞으로 굽었습니다. 허리를 펴 주세요."
            else:
                msg = "나쁜 자세가 계속되고 있습니다. 자세를 고쳐 주세요."
            speak_async(msg)
            tts_spoken = True

        if first_alert_time is None:
            first_alert_time = now
            last_sound_time = None

        if USE_SOUND and winsound is not None:
            if (now - first_alert_time) >= SOUND_ALERT_DELAY_SEC:
                if (last_sound_time is None) or (now - last_sound_time >= SOUND_INTERVAL_SEC):
                    winsound.Beep(1000, 500)
                    last_sound_time = now


    # -----------------------------
    # SUCCESS 애니메이션 트리거: (경고 상태였다가) GOOD로 안정 복귀하면 1회 출력
    # -----------------------------
    is_alerting_now = (is_bad_posture and bad_posture_duration >= BAD_POSTURE_ALERT_SEC)

    if status == "NO PERSON":
        # 사람 없으면 복귀로 치지 않음(재등장 시 오작동 방지)
        was_alerting = False
        good_start_time = None
    else:
        # 경고 단계에 실제로 들어갔는지 기록
        if is_alerting_now:
            was_alerting = True

        # GOOD 유지 시간 측정(안정화)
        if (status == "GOOD") and (not is_bad_posture):
            if good_start_time is None:
                good_start_time = now
        else:
            good_start_time = None

        # 경고 → GOOD 안정 복귀 시, 성공 애니메이션 시작(1회)
        if was_alerting and (good_start_time is not None) and ((now - good_start_time) >= GOOD_STABLE_SEC):
            success_anim_start = now
            was_alerting = False

    # =============================
    # 카메라 창 (영상 + 상태 한 줄)
    # =============================
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

    margin = 5
    border_thickness = 6
    cv2.rectangle(
        frame,
        (margin, margin),
        (w - margin, h - margin),
        border_color,
        border_thickness
    )

    # =============================
    # HUD 창 (텍스트/타이머/경고)
    # =============================
    hud_h = 400
    hud_w = 800
    # 전체를 회색으로 꽉 채움
    hud = np.full((hud_h, hud_w, 3), (60, 60, 60), dtype=np.uint8)

    font_small = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        hud,
        "Real-time AI Coach HUD",
        (30, 40),
        font_small,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ★ 여기 두 줄만 위로/촘촘하게 조정 ★
    line_y = 80        # 110 → 70 (정보 블록을 위로)
    line_gap = 30      # 30 → 28 (줄 간격 살짝 축소)

    cv2.putText(
        hud,
        f"Neck(CVA): {avg_neck_angle:5.1f} deg  [>=50 good]",
        (40, line_y),
        font_small,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        hud,
        f"Back(Torso-Thigh): {avg_slumped_angle:5.1f} deg  [>=90 good]",
        (40, line_y + line_gap),
        font_small,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        hud,
        f"Bad Posture Time: {bad_posture_duration:5.1f}s (alert>={BAD_POSTURE_ALERT_SEC:.0f}s)",
        (40, line_y + 2 * line_gap),
        font_small,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    session_time_str = format_time(elapsed_time)
    good_time_str = format_time(session_good_time)
    bad_time_str = format_time(session_bad_time)
    noperson_time_str = format_time(session_noperson_time)

    info_y = line_y + 3 * line_gap

    cv2.putText(
        hud,
        f"Session: {session_time_str}",
        (40, info_y),
        font_small,
        0.8,
        (200, 255, 200),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        hud,
        f"Good: {good_time_str}   Bad: {bad_time_str}",
        (40, info_y + line_gap),
        font_small,
        0.8,
        (200, 255, 200),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        hud,
        f"No Person: {noperson_time_str}",
        (40, info_y + 2 * line_gap),
        font_small,
        0.8,
        (230, 180, 180),
        2,
        cv2.LINE_AA
    )

    # NECK_WARN/SLUMP_WARN 줄 y좌표 저장
    neck_line_y = info_y + 3 * line_gap
    cv2.putText(
        hud,
        f"NECK_WARN: {neck_warn_count}   SLUMP_WARN: {slump_warn_count}",
        (40, neck_line_y),
        font_small,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --- 경고 패널: NECK_WARN 줄 아래, HUD 안에서 꽉 차게 ---
    if is_bad_posture and bad_posture_duration >= BAD_POSTURE_ALERT_SEC:
        panel_top = neck_line_y + 15
        panel_bottom = hud_h - 15
        panel_left = 15
        panel_right = hud_w - 15

        cv2.rectangle(
            hud,
            (panel_left, panel_top),
            (panel_right, panel_bottom),
            (0, 0, 255),
            -1
        )

        alert_font_scale = 1.0
        alert_thickness = 3
        line1 = "Bad posture over 3s!"
        line2 = "Straighten your back!"

        (w1, h1), _ = cv2.getTextSize(
            line1, font_small, alert_font_scale, alert_thickness
        )
        (w2, h2), _ = cv2.getTextSize(
            line2, font_small, alert_font_scale, alert_thickness
        )
        text_w = max(w1, w2)
        line_gap2 = 15

        x_text = (hud_w - text_w) // 2
        y_center = (panel_top + panel_bottom) // 2
        y_text = y_center - (h1 + line_gap2 // 2) + 15

        cv2.putText(
            hud,
            line1,
            (x_text, y_text),
            font_small,
            alert_font_scale,
            (255, 255, 255),
            alert_thickness,
            cv2.LINE_AA
        )
        cv2.putText(
            hud,
            line2,
            (x_text, y_text + h1 + line_gap2),
            font_small,
            alert_font_scale,
            (255, 255, 255),
            alert_thickness,
            cv2.LINE_AA
        )


    # -----------------------------
    # 성공 애니메이션 (HUD): 체크 + 확장 원 + 페이드아웃
    # -----------------------------
    if success_anim_start is not None:
        t_anim = now - success_anim_start
        if 0.0 <= t_anim <= SUCCESS_ANIM_SEC:
            p = t_anim / SUCCESS_ANIM_SEC  # 0~1

            overlay = hud.copy()

            cx, cy = hud_w // 2, int(hud_h * 0.78)
            radius = int(20 + 180 * p)

            # 확장 원
            cv2.circle(overlay, (cx, cy), radius, (0, 255, 0), 6, cv2.LINE_AA)

            # 체크 애니메이션(진행률에 따라 선이 늘어남)
            x1, y1 = int(cx - 60), int(cy + 10)
            x2, y2 = int(cx - 15), int(cy + 55)
            x3, y3 = int(cx + 75), int(cy - 35)

            p1 = min(1.0, p * 1.3)
            mx = int(x1 + (x2 - x1) * p1)
            my = int(y1 + (y2 - y1) * p1)

            p2 = max(0.0, min(1.0, (p - 0.3) * 1.5))
            nx = int(x2 + (x3 - x2) * p2)
            ny = int(y2 + (y3 - y2) * p2)

            cv2.line(overlay, (x1, y1), (mx, my), (0, 255, 0), 8, cv2.LINE_AA)
            if p > 0.3:
                cv2.line(overlay, (x2, y2), (nx, ny), (0, 255, 0), 8, cv2.LINE_AA)

            # 텍스트
            msg = "GOOD!"
            text_y = max(40, int(cy - radius - 15))
            cv2.putText(
                overlay,
                msg,
                (cx - 80, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 255, 0),
                4,
                cv2.LINE_AA
            )

            # 페이드아웃(처음 진하게 → 점점 옅게)
            alpha = 0.6 * (1.0 - p) + 0.1
            cv2.addWeighted(overlay, alpha, hud, 1 - alpha, 0, hud)
        else:
            success_anim_start = None

    # 창 출력
    cv2.imshow(window_cam, frame)
    cv2.imshow(window_hud, hud)

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
# 세션 요약 창
# -----------------------------
summary_h = 400
summary_w = 800
summary = np.zeros((summary_h, summary_w, 3), dtype=np.uint8)

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