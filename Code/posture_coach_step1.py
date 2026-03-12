import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# 세 점의 좌표를 이용해 각도를 계산하는 함수
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return angle if angle <= 180 else 360 - angle

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 임계값 설정
VISIBILITY_THRESHOLD = 0.7
SIDE_VIEW_Z_THRESHOLD = 0.15

# --- 스무딩을 위한 deque 초기화 (최근 5개 데이터 저장) ---
neck_angles_deque = deque(maxlen=5)
slumped_angles_deque = deque(maxlen=5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # --- 뷰(정면/측면) 판단 로직 ---
        left_shoulder_z = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
        right_shoulder_z = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
        z_difference = abs(left_shoulder_z - right_shoulder_z)
        is_frontal_view = z_difference < SIDE_VIEW_Z_THRESHOLD
        
        slumped_angles = []
        neck_angles = []

        # --- 왼쪽 관절 처리 ---
        if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > VISIBILITY_THRESHOLD:
            if is_frontal_view or left_shoulder_z < right_shoulder_z:
                left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)
                left_ear = (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h)
                left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h)
                left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h)
                
                slumped_angles.append(calculate_angle(left_shoulder, left_hip, left_knee))
                # --- [수정됨] 수평선 기준으로 CVA 각도 계산 ---
                neck_angles.append(calculate_angle((left_shoulder[0] - 100, left_shoulder[1]), left_shoulder, left_ear))
                
                cv2.line(frame, tuple(np.array(left_shoulder, dtype=int)), tuple(np.array(left_ear, dtype=int)), (0, 255, 0), 2)

        # --- 오른쪽 관절 처리 ---
        if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > VISIBILITY_THRESHOLD:
            if is_frontal_view or right_shoulder_z < left_shoulder_z:
                right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)
                right_ear = (landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h)
                right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)
                right_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h)

                slumped_angles.append(calculate_angle(right_shoulder, right_hip, right_knee))
                # --- [수정됨] 수평선 기준으로 CVA 각도 계산 ---
                neck_angles.append(calculate_angle((right_shoulder[0] - 100, right_shoulder[1]), right_shoulder, right_ear))

                cv2.line(frame, tuple(np.array(right_shoulder, dtype=int)), tuple(np.array(right_ear, dtype=int)), (0, 255, 0), 2)
        
        # --- 최종 각도 계산 및 스무딩 적용 ---
        if slumped_angles:
            slumped_angles_deque.append(np.mean(slumped_angles))
        if neck_angles:
            neck_angles_deque.append(np.mean(neck_angles))

        # --- 화면에 부드러워진 평균값 표시 ---
        if len(slumped_angles_deque) > 0:
            avg_slumped_angle = np.mean(slumped_angles_deque)
            cv2.putText(frame, f"Slumped Angle: {avg_slumped_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        if len(neck_angles_deque) > 0:
            avg_neck_angle = np.mean(neck_angles_deque)
            cv2.putText(frame, f"Neck Angle (CVA): {avg_neck_angle:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Real-time AI Coach', frame)

    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()