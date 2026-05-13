"""
================================================================================
OAK-D Pro 논문 연구용 - 2D vs 3D 자세 각도 비교 시스템 v3
================================================================================

연구 목적:
    - RGB만 사용한 2D 각도 vs RGB+Depth 사용한 3D 각도 비교
    - 동일 프레임에서 동시 측정으로 정확한 비교

측정 항목:
    - 목(Neck): CVA (Craniovertebral Angle) - 수평선/수평면 기준
    - 허리(Back): S-H-K (Shoulder-Hip-Knee) - 세 점 각도

개선 사항 (v2 대비):
    - 중복 코드 제거 (함수 모듈화)
    - 3D CVA 계산 정확성 개선
    - 논문용 통계 추가 (평균, 표준편차)
    - 코드 가독성 향상
    
저자: Claude (연구 보조)
================================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, List
import csv
import time
import os


# ==============================================================================
# 설정 (Configuration)
# ==============================================================================

@dataclass
class Config:
    """연구 설정 파라미터"""
    
    # ⭐ 거리별 실험 설정
    EXPERIMENT_DISTANCE_M: float = 2.0       # 🔴 여기를 수정! (0.7, 1.0, 1.5, 2.0)
    EXPERIMENT_DURATION_SEC: int = 180      # 3분 = 180초 (자동 종료)
    
    # 데이터 수집
    DATA_COLLECTION_INTERVAL: float = 15.0  # ← 15초 간격으로 변경
    ANGLE_HISTORY_SIZE: int = 10            # 스무딩용 히스토리 크기
    
    # MediaPipe 임계값
    VISIBILITY_THRESHOLD: float = 0.5       # 관절 가시성 임계값
    SIDE_VIEW_Z_THRESHOLD: float = 0.10     # 측면/정면 판단 임계값
    
    # Depth 샘플링
    DEPTH_SAMPLE_SIZE: int = 11             # 중앙값 샘플링 커널 크기
    DEPTH_MIN_M: float = 0.2                # 최소 유효 깊이 (m)
    DEPTH_MAX_M: float = 10.0               # 최대 유효 깊이 (m)
    
    # 화면 설정
    HUD_WIDTH: int = 750
    HUD_HEIGHT: int = 650


CONFIG = Config()


# ==============================================================================
# 데이터 클래스 (Data Classes)
# ==============================================================================

@dataclass(frozen=True)
class CameraIntrinsics:
    """카메라 내부 파라미터"""
    fx: float  # X축 초점 거리
    fy: float  # Y축 초점 거리
    cx: float  # 주점 X
    cy: float  # 주점 Y


@dataclass
class JointPixel:
    """관절 픽셀 좌표"""
    shoulder: Tuple[float, float]
    ear: Tuple[float, float]
    hip: Tuple[float, float]
    knee: Tuple[float, float]


@dataclass
class Joint3D:
    """관절 3D 좌표"""
    shoulder: np.ndarray
    ear: np.ndarray
    hip: np.ndarray
    knee: np.ndarray


@dataclass
class AngleResult:
    """각도 측정 결과"""
    neck_2d: float
    neck_3d: float
    back_2d: float
    back_3d: float
    
    @property
    def neck_diff(self) -> float:
        """목 각도 차이"""
        if self.neck_2d > 0 and self.neck_3d > 0:
            return abs(self.neck_3d - self.neck_2d)
        return 0.0
    
    @property
    def back_diff(self) -> float:
        """허리 각도 차이"""
        if self.back_2d > 0 and self.back_3d > 0:
            return abs(self.back_3d - self.back_2d)
        return 0.0
    
    @property
    def is_valid(self) -> bool:
        """유효한 측정인지"""
        return self.neck_2d > 0 and self.neck_3d > 0


# ==============================================================================
# 유틸리티 함수 (Utilities)
# ==============================================================================

def format_time(seconds: float) -> str:
    """초를 MM:SS 형식으로 변환"""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def clamp_coordinate(value: int, max_value: int) -> int:
    """좌표를 유효 범위 내로 제한"""
    return max(0, min(max_value - 1, value))


# ==============================================================================
# 2D 각도 계산 (RGB만 사용)
# ==============================================================================

def calculate_angle_2d_three_points(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],  # 꼭짓점
    point_c: Tuple[float, float]
) -> float:
    """
    2D에서 세 점의 각도 계산 (b가 꼭짓점)
    
    Args:
        point_a: 첫 번째 점 (x, y)
        point_b: 꼭짓점 (x, y)
        point_c: 세 번째 점 (x, y)
    
    Returns:
        각도 (degrees, 0~180)
    """
    a = np.array(point_a)
    b = np.array(point_b)
    c = np.array(point_c)
    
    ba = a - b
    bc = c - b
    
    # arctan2를 사용한 각도 계산
    angle_ba = np.arctan2(ba[1], ba[0])
    angle_bc = np.arctan2(bc[1], bc[0])
    
    angle = np.abs(angle_bc - angle_ba) * 180.0 / np.pi
    
    if angle > 180.0:
        angle = 360.0 - angle
    
    return angle


def calculate_neck_cva_2d(
    shoulder: Tuple[float, float],
    ear: Tuple[float, float],
    is_left_side: bool = True
) -> float:
    """
    2D 목 CVA (Craniovertebral Angle) 계산
    
    정의: 어깨에서 수평선을 긋고, 귀까지의 각도
    
           귀
          /
         / ) CVA
    ----어깨--------  (수평선)
    
    Args:
        shoulder: 어깨 좌표 (x, y)
        ear: 귀 좌표 (x, y)
        is_left_side: 왼쪽 측면인지 (방향 결정)
    
    Returns:
        CVA 각도 (degrees)
    """
    # 수평 기준점 생성 (어깨에서 수평으로 100px)
    if is_left_side:
        horizontal_point = (shoulder[0] - 100, shoulder[1])
    else:
        horizontal_point = (shoulder[0] + 100, shoulder[1])
    
    return calculate_angle_2d_three_points(horizontal_point, shoulder, ear)


def calculate_back_angle_2d(
    shoulder: Tuple[float, float],
    hip: Tuple[float, float],
    knee: Tuple[float, float]
) -> float:
    """
    2D 허리 각도 (S-H-K) 계산
    
    정의: Shoulder-Hip-Knee 세 점에서 Hip이 꼭짓점인 각도
    
    어깨(S)
         \
          \ ) 각도
           Hip --------
          /
         /
      무릎(K)
    
    Args:
        shoulder: 어깨 좌표
        hip: 엉덩이 좌표 (꼭짓점)
        knee: 무릎 좌표
    
    Returns:
        S-H-K 각도 (degrees)
    """
    return calculate_angle_2d_three_points(shoulder, hip, knee)


# ==============================================================================
# 3D 각도 계산 (RGB + Depth)
# ==============================================================================

def sample_depth_median(
    depth_map: np.ndarray,
    u: int,
    v: int,
    kernel_size: int = CONFIG.DEPTH_SAMPLE_SIZE
) -> float:
    """
    깊이맵에서 중앙값 샘플링 (노이즈 제거)
    
    Args:
        depth_map: 깊이맵 (H, W), 단위: meters
        u: X 좌표 (픽셀)
        v: Y 좌표 (픽셀)
        kernel_size: 샘플링 커널 크기
    
    Returns:
        깊이 값 (meters), 실패시 nan
    """
    if depth_map is None:
        return float("nan")
    
    h, w = depth_map.shape[:2]
    
    # 좌표 범위 확인
    if not (0 <= u < w and 0 <= v < h):
        return float("nan")
    
    # 커널 범위 계산
    r = kernel_size // 2
    x0, x1 = max(0, u - r), min(w, u + r + 1)
    y0, y1 = max(0, v - r), min(h, v + r + 1)
    
    # 패치 추출 및 유효값 필터링
    patch = depth_map[y0:y1, x0:x1].astype(np.float32)
    valid_mask = (
        np.isfinite(patch) & 
        (patch > CONFIG.DEPTH_MIN_M) & 
        (patch < CONFIG.DEPTH_MAX_M)
    )
    
    if not np.any(valid_mask):
        # 더 넓은 범위로 재시도
        r2 = kernel_size * 2
        x0, x1 = max(0, u - r2), min(w, u + r2 + 1)
        y0, y1 = max(0, v - r2), min(h, v + r2 + 1)
        patch = depth_map[y0:y1, x0:x1].astype(np.float32)
        valid_mask = (
            np.isfinite(patch) & 
            (patch > CONFIG.DEPTH_MIN_M) & 
            (patch < CONFIG.DEPTH_MAX_M)
        )
        
        if not np.any(valid_mask):
            return float("nan")
    
    return float(np.median(patch[valid_mask]))


def backproject_to_3d(
    u: float,
    v: float,
    z: float,
    intrinsics: CameraIntrinsics
) -> np.ndarray:
    """
    2D 픽셀 좌표 + 깊이 → 3D 카메라 좌표 역투영
    
    공식:
        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        Z = z
    
    Args:
        u: 픽셀 X 좌표
        v: 픽셀 Y 좌표
        z: 깊이 (meters)
        intrinsics: 카메라 내부 파라미터
    
    Returns:
        3D 좌표 [X, Y, Z] (meters)
    """
    if not np.isfinite(z) or z <= 0.0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    
    X = (u - intrinsics.cx) * z / intrinsics.fx
    Y = (v - intrinsics.cy) * z / intrinsics.fy
    Z = z
    
    return np.array([X, Y, Z], dtype=np.float32)


def calculate_neck_cva_3d(
    shoulder_3d: np.ndarray,
    ear_3d: np.ndarray
) -> float:
    """
    3D 목 CVA 계산 (수평면 기준)
    
    2D CVA와 동일한 정의: 어깨에서 수평면을 기준으로 귀까지의 각도
    
    3D에서 수평면 = XZ 평면 (Y가 수직)
    
           귀 (ear_3d)
          /|
         / |
        /  | 수직 성분 (Y)
       /   |
    어깨----+  수평 성분 (XZ 평면)
    
    Args:
        shoulder_3d: 어깨 3D 좌표 [X, Y, Z]
        ear_3d: 귀 3D 좌표 [X, Y, Z]
    
    Returns:
        CVA 각도 (degrees), 실패시 0.0
    """
    # 유효성 검사
    if not (np.all(np.isfinite(shoulder_3d)) and np.all(np.isfinite(ear_3d))):
        return 0.0
    
    # 어깨 → 귀 벡터
    vec = ear_3d - shoulder_3d
    
    # 수평 거리 (XZ 평면에서)
    horizontal_dist = np.sqrt(vec[0]**2 + vec[2]**2)
    
    # 수직 거리 (Y축, 위가 음수이므로 부호 반전)
    vertical_dist = -vec[1]
    
    # 0으로 나누기 방지
    if horizontal_dist < 1e-6:
        return 90.0 if vertical_dist > 0 else 0.0
    
    # 수평면 기준 각도
    angle = np.degrees(np.arctan2(vertical_dist, horizontal_dist))
    
    return max(0.0, angle)


def calculate_back_angle_3d(
    shoulder_3d: np.ndarray,
    hip_3d: np.ndarray,
    knee_3d: np.ndarray
) -> float:
    """
    3D 허리 각도 (S-H-K) 계산
    
    2D와 동일한 정의: Shoulder-Hip-Knee에서 Hip이 꼭짓점
    
    벡터 내적으로 각도 계산:
        cos(θ) = (v1 · v2) / (|v1| * |v2|)
    
    Args:
        shoulder_3d: 어깨 3D 좌표
        hip_3d: 엉덩이 3D 좌표 (꼭짓점)
        knee_3d: 무릎 3D 좌표
    
    Returns:
        S-H-K 각도 (degrees), 실패시 0.0
    """
    # 유효성 검사
    if not all(np.all(np.isfinite(p)) for p in [shoulder_3d, hip_3d, knee_3d]):
        return 0.0
    
    # Hip에서 다른 두 점으로의 벡터
    vec_to_shoulder = shoulder_3d - hip_3d
    vec_to_knee = knee_3d - hip_3d
    
    # 벡터 크기
    norm_s = np.linalg.norm(vec_to_shoulder)
    norm_k = np.linalg.norm(vec_to_knee)
    
    # 0으로 나누기 방지
    if norm_s < 1e-6 or norm_k < 1e-6:
        return 0.0
    
    # 내적으로 cos(θ) 계산
    cos_angle = np.dot(vec_to_shoulder, vec_to_knee) / (norm_s * norm_k)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 수치 안정성
    
    return float(np.degrees(np.arccos(cos_angle)))


# ==============================================================================
# 관절 처리 (Joint Processing)
# ==============================================================================

def extract_joint_pixels(
    landmarks,
    width: int,
    height: int,
    side: str  # 'left' or 'right'
) -> Optional[JointPixel]:
    """
    MediaPipe 랜드마크에서 관절 픽셀 좌표 추출
    
    Args:
        landmarks: MediaPipe 랜드마크
        width: 이미지 너비
        height: 이미지 높이
        side: 'left' 또는 'right'
    
    Returns:
        JointPixel 또는 None (가시성 부족시)
    """
    mp_pose = mp.solutions.pose
    
    if side == 'left':
        shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        ear_idx = mp_pose.PoseLandmark.LEFT_EAR.value
        hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
        knee_idx = mp_pose.PoseLandmark.LEFT_KNEE.value
    else:
        shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        ear_idx = mp_pose.PoseLandmark.RIGHT_EAR.value
        hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value
        knee_idx = mp_pose.PoseLandmark.RIGHT_KNEE.value
    
    # 가시성 확인
    if landmarks[shoulder_idx].visibility < CONFIG.VISIBILITY_THRESHOLD:
        return None
    
    return JointPixel(
        shoulder=(landmarks[shoulder_idx].x * width, landmarks[shoulder_idx].y * height),
        ear=(landmarks[ear_idx].x * width, landmarks[ear_idx].y * height),
        hip=(landmarks[hip_idx].x * width, landmarks[hip_idx].y * height),
        knee=(landmarks[knee_idx].x * width, landmarks[knee_idx].y * height)
    )


def convert_to_3d(
    joints: JointPixel,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    width: int,
    height: int
) -> Joint3D:
    """
    2D 관절 좌표를 3D로 변환
    
    Args:
        joints: 2D 관절 픽셀 좌표
        depth_map: 깊이맵 (meters)
        intrinsics: 카메라 내부 파라미터
        width: 이미지 너비
        height: 이미지 높이
    
    Returns:
        Joint3D 객체
    """
    def to_3d(pixel_coord):
        u = clamp_coordinate(int(pixel_coord[0] + 0.5), width)
        v = clamp_coordinate(int(pixel_coord[1] + 0.5), height)
        z = sample_depth_median(depth_map, u, v)
        return backproject_to_3d(u, v, z, intrinsics)
    
    return Joint3D(
        shoulder=to_3d(joints.shoulder),
        ear=to_3d(joints.ear),
        hip=to_3d(joints.hip),
        knee=to_3d(joints.knee)
    )


def compute_angles_for_side(
    landmarks,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    width: int,
    height: int,
    side: str
) -> Optional[Tuple[float, float, float, float, JointPixel]]:
    """
    한쪽 측면의 2D/3D 각도 계산
    
    Args:
        landmarks: MediaPipe 랜드마크
        depth_map: 깊이맵
        intrinsics: 카메라 내부 파라미터
        width: 이미지 너비
        height: 이미지 높이
        side: 'left' 또는 'right'
    
    Returns:
        (neck_2d, neck_3d, back_2d, back_3d, joints) 또는 None
    """
    # 관절 픽셀 좌표 추출
    joints = extract_joint_pixels(landmarks, width, height, side)
    if joints is None:
        return None
    
    # 2D 각도 계산
    is_left = (side == 'left')
    neck_2d = calculate_neck_cva_2d(joints.shoulder, joints.ear, is_left)
    back_2d = calculate_back_angle_2d(joints.shoulder, joints.hip, joints.knee)
    
    # 3D 변환 및 각도 계산
    joints_3d = convert_to_3d(joints, depth_map, intrinsics, width, height)
    neck_3d = calculate_neck_cva_3d(joints_3d.shoulder, joints_3d.ear)
    back_3d = calculate_back_angle_3d(joints_3d.shoulder, joints_3d.hip, joints_3d.knee)
    
    return (neck_2d, neck_3d, back_2d, back_3d, joints)


# ==============================================================================
# OAK-D 파이프라인 (Pipeline)
# ==============================================================================

def create_oakd_pipeline():
    """
    OAK-D Pro 파이프라인 생성
    
    설정:
        - RGB: 720p (1080p를 2/3 스케일)
        - Depth: 720p 모노 카메라 기반 스테레오
        - Depth를 RGB에 정렬
    """
    import depthai as dai
    
    pipeline = dai.Pipeline()
    
    # ===== 컬러 카메라 (RGB) =====
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setIspScale(2, 3)  # 1080p * 2/3 = 720p
    cam_rgb.setFps(30)
    
    # ===== 모노 카메라 (스테레오용) =====
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_left.setFps(30)
    mono_right.setFps(30)
    
    # ===== 스테레오 Depth =====
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # RGB에 정렬
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.initialConfig.setConfidenceThreshold(200)
    
    # 모노 → 스테레오 연결
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    
    # ===== 출력 스트림 =====
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")
    
    cam_rgb.isp.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)
    
    return pipeline


# ==============================================================================
# HUD 렌더링 (Visualization)
# ==============================================================================

def render_hud(
    result: AngleResult,
    elapsed: float,
    data_count: int,
    time_until_next: float,
    time_since_last: float,
    log_file: str,
    statistics: dict
) -> np.ndarray:
    """
    HUD 패널 렌더링
    
    Args:
        result: 각도 측정 결과
        elapsed: 경과 시간
        data_count: 수집된 데이터 수
        time_until_next: 다음 수집까지 남은 시간
        time_since_last: 마지막 수집 이후 시간
        log_file: 로그 파일명
        statistics: 통계 정보
    
    Returns:
        HUD 이미지 (numpy array)
    """
    hud_w = CONFIG.HUD_WIDTH
    hud_h = CONFIG.HUD_HEIGHT
    
    # 배경 생성
    hud = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
    hud[:] = (30, 30, 30)  # 어두운 회색
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # ===== 타이틀 =====
    cv2.putText(hud, "2D vs 3D Posture Angle Comparison", 
                (20, 40), font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(hud, "Research Mode", 
                (20, 70), font, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    
    # ===== 상태 바 =====
    y_status = 100
    cv2.rectangle(hud, (20, y_status), (hud_w - 20, y_status + 45), (50, 50, 50), -1)
    cv2.putText(hud, f"Data: {data_count}", 
                (30, y_status + 30), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(hud, f"Next: {max(0, time_until_next):.0f}s", 
                (180, y_status + 30), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(hud, f"Session: {format_time(elapsed)}", 
                (350, y_status + 30), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(hud, f"Interval: {CONFIG.DATA_COLLECTION_INTERVAL:.0f}s", 
                (550, y_status + 30), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    
    # ===== 목 (NECK) 섹션 =====
    y_neck = 170
    cv2.rectangle(hud, (20, y_neck), (hud_w - 20, y_neck + 150), (40, 60, 40), -1)
    cv2.putText(hud, "NECK (CVA - Craniovertebral Angle)", 
                (30, y_neck + 30), font, 0.75, (100, 255, 100), 2, cv2.LINE_AA)
    
    # 2D
    cv2.putText(hud, "2D (RGB only):", 
                (50, y_neck + 70), font, 0.65, (150, 150, 255), 2, cv2.LINE_AA)
    cv2.putText(hud, f"{result.neck_2d:6.1f} deg", 
                (250, y_neck + 70), font, 0.8, (150, 150, 255), 2, cv2.LINE_AA)
    
    # 3D
    cv2.putText(hud, "3D (RGB+Depth):", 
                (50, y_neck + 105), font, 0.65, (150, 255, 150), 2, cv2.LINE_AA)
    cv2.putText(hud, f"{result.neck_3d:6.1f} deg", 
                (250, y_neck + 105), font, 0.8, (150, 255, 150), 2, cv2.LINE_AA)
    
    # 차이
    neck_diff_color = get_diff_color(result.neck_diff)
    cv2.putText(hud, "Difference:", 
                (450, y_neck + 70), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(hud, f"{result.neck_diff:5.1f} deg", 
                (570, y_neck + 70), font, 0.85, neck_diff_color, 2, cv2.LINE_AA)
    
    # 통계
    if 'neck' in statistics and statistics['neck']['count'] > 0:
        stats = statistics['neck']
        cv2.putText(hud, f"Avg Diff: {stats['mean']:.1f} +/- {stats['std']:.1f}", 
                    (450, y_neck + 105), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    
    # ===== 허리 (BACK) 섹션 =====
    y_back = 340
    cv2.rectangle(hud, (20, y_back), (hud_w - 20, y_back + 150), (60, 40, 40), -1)
    cv2.putText(hud, "BACK (S-H-K - Shoulder-Hip-Knee)", 
                (30, y_back + 30), font, 0.75, (255, 100, 100), 2, cv2.LINE_AA)
    
    # 2D
    cv2.putText(hud, "2D (RGB only):", 
                (50, y_back + 70), font, 0.65, (150, 150, 255), 2, cv2.LINE_AA)
    cv2.putText(hud, f"{result.back_2d:6.1f} deg", 
                (250, y_back + 70), font, 0.8, (150, 150, 255), 2, cv2.LINE_AA)
    
    # 3D
    cv2.putText(hud, "3D (RGB+Depth):", 
                (50, y_back + 105), font, 0.65, (150, 255, 150), 2, cv2.LINE_AA)
    cv2.putText(hud, f"{result.back_3d:6.1f} deg", 
                (250, y_back + 105), font, 0.8, (150, 255, 150), 2, cv2.LINE_AA)
    
    # 차이
    back_diff_color = get_diff_color(result.back_diff)
    cv2.putText(hud, "Difference:", 
                (450, y_back + 70), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(hud, f"{result.back_diff:5.1f} deg", 
                (570, y_back + 70), font, 0.85, back_diff_color, 2, cv2.LINE_AA)
    
    # 통계
    if 'back' in statistics and statistics['back']['count'] > 0:
        stats = statistics['back']
        cv2.putText(hud, f"Avg Diff: {stats['mean']:.1f} +/- {stats['std']:.1f}", 
                    (450, y_back + 105), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    
    # ===== 프로그레스 바 =====
    y_bar = 520
    cv2.putText(hud, "Next Data Collection:", 
                (20, y_bar), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    
    bar_x, bar_y = 20, y_bar + 15
    bar_w, bar_h = hud_w - 40, 25
    progress = min(1.0, time_since_last / CONFIG.DATA_COLLECTION_INTERVAL)
    
    cv2.rectangle(hud, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    cv2.rectangle(hud, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 180, 0), -1)
    cv2.rectangle(hud, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 2)
    
    # ===== 범례 =====
    y_legend = 590
    cv2.putText(hud, "Diff Color:", (20, y_legend), font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.circle(hud, (120, y_legend - 5), 8, (0, 255, 0), -1)
    cv2.putText(hud, "<5", (135, y_legend), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.circle(hud, (180, y_legend - 5), 8, (0, 255, 255), -1)
    cv2.putText(hud, "5-10", (195, y_legend), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.circle(hud, (260, y_legend - 5), 8, (0, 0, 255), -1)
    cv2.putText(hud, ">10", (275, y_legend), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    
    # ===== 도움말 & 파일명 =====
    cv2.putText(hud, "ESC: Exit  |  S: Manual Save  |  +/-: Interval", 
                (20, hud_h - 25), font, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(hud, f"File: {log_file}", 
                (20, hud_h - 5), font, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
    
    return hud


def get_diff_color(diff: float) -> Tuple[int, int, int]:
    """차이값에 따른 색상 반환 (BGR)"""
    if diff < 5:
        return (0, 255, 0)    # 녹색: 작은 차이
    elif diff < 10:
        return (0, 255, 255)  # 노랑: 중간 차이
    else:
        return (0, 0, 255)    # 빨강: 큰 차이


def draw_joint_markers(
    frame: np.ndarray,
    joints: JointPixel,
    width: int,
    height: int
) -> None:
    """관절 위치에 마커 그리기"""
    def to_int(coord):
        return (
            clamp_coordinate(int(coord[0] + 0.5), width),
            clamp_coordinate(int(coord[1] + 0.5), height)
        )
    
    # 마커 색상 (BGR)
    colors = {
        'shoulder': (255, 0, 255),   # 마젠타
        'ear': (0, 255, 255),        # 시안
        'hip': (0, 255, 0),          # 녹색
        'knee': (0, 165, 255)        # 주황
    }
    
    cv2.circle(frame, to_int(joints.shoulder), 10, colors['shoulder'], -1)
    cv2.circle(frame, to_int(joints.ear), 10, colors['ear'], -1)
    cv2.circle(frame, to_int(joints.hip), 10, colors['hip'], -1)
    cv2.circle(frame, to_int(joints.knee), 10, colors['knee'], -1)


# ==============================================================================
# 통계 계산 (Statistics)
# ==============================================================================

class StatisticsTracker:
    """실시간 통계 추적"""
    
    def __init__(self):
        self.neck_diffs: List[float] = []
        self.back_diffs: List[float] = []
    
    def add(self, neck_diff: float, back_diff: float) -> None:
        """새 데이터 추가"""
        if neck_diff > 0:
            self.neck_diffs.append(neck_diff)
        if back_diff > 0:
            self.back_diffs.append(back_diff)
    
    def get_stats(self) -> dict:
        """통계 계산"""
        def calc(data):
            if len(data) == 0:
                return {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            return {
                'count': len(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            }
        
        return {
            'neck': calc(self.neck_diffs),
            'back': calc(self.back_diffs)
        }
    
    def get_summary_string(self) -> str:
        """요약 문자열 생성"""
        stats = self.get_stats()
        lines = []
        lines.append("=" * 60)
        lines.append("📊 통계 요약 (Statistics Summary)")
        lines.append("=" * 60)
        
        if stats['neck']['count'] > 0:
            n = stats['neck']
            lines.append(f"NECK CVA 차이:")
            lines.append(f"  - 데이터 수: {n['count']}")
            lines.append(f"  - 평균 (Mean): {n['mean']:.2f}°")
            lines.append(f"  - 표준편차 (Std): {n['std']:.2f}°")
            lines.append(f"  - 최소/최대: {n['min']:.2f}° / {n['max']:.2f}°")
        
        if stats['back']['count'] > 0:
            b = stats['back']
            lines.append(f"BACK S-H-K 차이:")
            lines.append(f"  - 데이터 수: {b['count']}")
            lines.append(f"  - 평균 (Mean): {b['mean']:.2f}°")
            lines.append(f"  - 표준편차 (Std): {b['std']:.2f}°")
            lines.append(f"  - 최소/최대: {b['min']:.2f}° / {b['max']:.2f}°")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ==============================================================================
# 메인 함수 (Main)
# ==============================================================================

def main():
    """메인 실행 함수"""
    
    # DepthAI 확인
    try:
        import depthai as dai
    except ImportError:
        print("❌ DepthAI 라이브러리가 필요합니다.")
        print("   설치: pip install depthai")
        return
    
    # 파이프라인 생성 및 장치 연결
    pipeline = create_oakd_pipeline()
    
    print("=" * 60)
    print("🔬 OAK-D Pro 논문 연구 시스템 v3")
    print("   2D vs 3D 자세 각도 비교")
    print("=" * 60)
    print("\n🔌 OAK-D Pro 연결 중...")
    
    try:
        device = dai.Device(pipeline)
    except Exception as e:
        print(f"❌ OAK-D 연결 실패: {e}")
        return
    
    print("✅ OAK-D Pro 연결 완료!")
    
    # 출력 큐 설정
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
    
    # 해상도 감지
    print("\n📐 해상도 감지 중...")
    rgb_width, rgb_height = None, None
    depth_width, depth_height = None, None
    
    while rgb_width is None or depth_width is None:
        in_rgb = q_rgb.tryGet()
        in_depth = q_depth.tryGet()
        
        if in_rgb is not None and rgb_width is None:
            frame = in_rgb.getCvFrame()
            rgb_height, rgb_width = frame.shape[:2]
            print(f"   RGB: {rgb_width} x {rgb_height}")
        
        if in_depth is not None and depth_width is None:
            depth_frame = in_depth.getFrame()
            depth_height, depth_width = depth_frame.shape[:2]
            print(f"   Depth: {depth_width} x {depth_height}")
    
    # 카메라 Intrinsics
    calib = device.readCalibration()
    intrinsics_data = calib.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A,
        rgb_width, rgb_height
    )
    intrinsics = CameraIntrinsics(
        fx=intrinsics_data[0][0],
        fy=intrinsics_data[1][1],
        cx=intrinsics_data[0][2],
        cy=intrinsics_data[1][2]
    )
    print(f"📷 Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, "
          f"cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}")
    
    # CSV 파일 초기화
    distance_str = f"{CONFIG.EXPERIMENT_DISTANCE_M:.1f}m"
    log_file = f"posture_2d_vs_3d_{distance_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(log_file, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'timestamp', 'elapsed_sec',
        'neck_2d', 'neck_3d', 'neck_diff',
        'back_2d', 'back_3d', 'back_diff',
        'note'
    ])
    
    print(f"\n📝 데이터 저장: {log_file}")
    print(f"⏱️  수집 간격: {CONFIG.DATA_COLLECTION_INTERVAL}초")
    
    # MediaPipe 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # 윈도우 설정
    window_cam = "Research - Camera"
    window_hud = "Research - 2D vs 3D Comparison"
    window_depth = "Research - Depth Map"
    
    cv2.namedWindow(window_cam, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_hud, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_depth, cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow(window_cam, 640, 480)
    cv2.resizeWindow(window_hud, CONFIG.HUD_WIDTH, CONFIG.HUD_HEIGHT)
    cv2.resizeWindow(window_depth, 640, 480)
    
    try:
        cv2.moveWindow(window_cam, 50, 50)
        cv2.moveWindow(window_hud, 720, 50)
        cv2.moveWindow(window_depth, 50, 580)
    except:
        pass
    
    # 상태 변수 초기화
    start_time = time.time()
    last_collection_time = start_time
    
    neck_2d_history = deque(maxlen=CONFIG.ANGLE_HISTORY_SIZE)
    neck_3d_history = deque(maxlen=CONFIG.ANGLE_HISTORY_SIZE)
    back_2d_history = deque(maxlen=CONFIG.ANGLE_HISTORY_SIZE)
    back_3d_history = deque(maxlen=CONFIG.ANGLE_HISTORY_SIZE)
    
    data_count = 0
    stats_tracker = StatisticsTracker()
    
    print("\n" + "=" * 80)
    print("🎬 거리별 자세 측정 실험 시작!")
    print(f"   📏 거리: {CONFIG.EXPERIMENT_DISTANCE_M}m")
    print(f"   ⏱️  실험 시간: {CONFIG.EXPERIMENT_DURATION_SEC}초 ({CONFIG.EXPERIMENT_DURATION_SEC//60}분)")
    print(f"   📊 수집 간격: {CONFIG.DATA_COLLECTION_INTERVAL:.0f}초")
    print(f"   📈 예상 샘플: {CONFIG.EXPERIMENT_DURATION_SEC // int(CONFIG.DATA_COLLECTION_INTERVAL)}개")
    print("   ⌨️  ESC: 수동 종료 | S: 수동 저장")
    print("=" * 80 + "\n")
    
    # ===== 메인 루프 =====
    while True:
        in_rgb = q_rgb.tryGet()
        in_depth = q_depth.tryGet()
        
        if in_rgb is None or in_depth is None:
            continue
        
        # 시간 계산 (자동 종료 조건 확인)
        now = time.time()
        elapsed = now - start_time
        
        # 【자동 종료 조건】실험 시간 도달
        if elapsed >= CONFIG.EXPERIMENT_DURATION_SEC:
            print(f"\n{'='*80}")
            print(f"✅ 실험 완료! ({CONFIG.EXPERIMENT_DURATION_SEC}초 도달)")
            print(f"   📊 총 샘플: {data_count}개")
            print(f"   📁 파일: {log_file}")
            print(f"   📏 거리: {CONFIG.EXPERIMENT_DISTANCE_M}m")
            print(f"{'='*80}\n")
            break
        
        # 프레임 획득
        frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame()
        
        h, w = frame.shape[:2]
        dh, dw = depth_frame.shape[:2]
        
        # Depth 리사이즈 (RGB와 동일 크기로)
        if (dw, dh) != (w, h):
            depth_frame_resized = cv2.resize(
                depth_frame, (w, h), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            depth_frame_resized = depth_frame
        
        # 깊이맵 미터 변환
        depth_m = depth_frame_resized.astype(np.float32) / 1000.0
        
        # MediaPipe 포즈 추정
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # 시간 계산
        now = time.time()
        elapsed = now - start_time
        time_since_last = now - last_collection_time
        time_until_next = CONFIG.DATA_COLLECTION_INTERVAL - time_since_last
        
        # 각도 초기화
        current_result = AngleResult(0.0, 0.0, 0.0, 0.0)
        
        # ===== 포즈 감지 시 각도 계산 =====
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 측면 판단
            ls_z = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
            rs_z = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
            is_frontal = abs(ls_z - rs_z) < CONFIG.SIDE_VIEW_Z_THRESHOLD
            
            neck_2d_list, neck_3d_list = [], []
            back_2d_list, back_3d_list = [], []
            
            # 왼쪽 측 처리
            if is_frontal or ls_z < rs_z:
                result_left = compute_angles_for_side(
                    landmarks, depth_m, intrinsics, w, h, 'left'
                )
                if result_left:
                    n2d, n3d, b2d, b3d, joints = result_left
                    neck_2d_list.append(n2d)
                    back_2d_list.append(b2d)
                    if n3d > 0:
                        neck_3d_list.append(n3d)
                    if b3d > 0:
                        back_3d_list.append(b3d)
                    draw_joint_markers(frame, joints, w, h)
            
            # 오른쪽 측 처리
            if is_frontal or rs_z < ls_z:
                result_right = compute_angles_for_side(
                    landmarks, depth_m, intrinsics, w, h, 'right'
                )
                if result_right:
                    n2d, n3d, b2d, b3d, joints = result_right
                    neck_2d_list.append(n2d)
                    back_2d_list.append(b2d)
                    if n3d > 0:
                        neck_3d_list.append(n3d)
                    if b3d > 0:
                        back_3d_list.append(b3d)
                    draw_joint_markers(frame, joints, w, h)
            
            # 히스토리 업데이트
            if neck_2d_list:
                neck_2d_history.append(np.mean(neck_2d_list))
            if neck_3d_list:
                neck_3d_history.append(np.mean(neck_3d_list))
            if back_2d_list:
                back_2d_history.append(np.mean(back_2d_list))
            if back_3d_list:
                back_3d_history.append(np.mean(back_3d_list))
            
            # 포즈 스켈레톤 그리기
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
        
        # 스무딩된 결과
        current_result = AngleResult(
            neck_2d=float(np.mean(neck_2d_history)) if neck_2d_history else 0.0,
            neck_3d=float(np.mean(neck_3d_history)) if neck_3d_history else 0.0,
            back_2d=float(np.mean(back_2d_history)) if back_2d_history else 0.0,
            back_3d=float(np.mean(back_3d_history)) if back_3d_history else 0.0
        )
        
        # ===== 자동 데이터 수집 =====
        if time_since_last >= CONFIG.DATA_COLLECTION_INTERVAL and current_result.is_valid:
            data_count += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            csv_writer.writerow([
                timestamp, f"{elapsed:.1f}",
                f"{current_result.neck_2d:.2f}", f"{current_result.neck_3d:.2f}", 
                f"{current_result.neck_diff:.2f}",
                f"{current_result.back_2d:.2f}", f"{current_result.back_3d:.2f}", 
                f"{current_result.back_diff:.2f}",
                "auto"
            ])
            csv_file.flush()
            
            stats_tracker.add(current_result.neck_diff, current_result.back_diff)
            
            print(f"📊 [{data_count:3d}] {format_time(elapsed)} | "
                  f"Neck: 2D={current_result.neck_2d:5.1f}° 3D={current_result.neck_3d:5.1f}° "
                  f"(Δ{current_result.neck_diff:4.1f}°) | "
                  f"Back: 2D={current_result.back_2d:5.1f}° 3D={current_result.back_3d:5.1f}° "
                  f"(Δ{current_result.back_diff:4.1f}°)")
            
            last_collection_time = now
        
        # ===== 화면 표시 =====
        frame_display = cv2.flip(frame, 1)
        cv2.rectangle(frame_display, (0, 0), (w-1, h-1), (0, 255, 0), 4)
        
        # Depth 컬러맵
        depth_display = cv2.convertScaleAbs(depth_frame_resized, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        depth_colormap = cv2.flip(depth_colormap, 1)
        
        # HUD 렌더링
        hud = render_hud(
            current_result, elapsed, data_count,
            time_until_next, time_since_last, log_file,
            stats_tracker.get_stats()
        )
        
        # 창 출력
        cv2.imshow(window_cam, frame_display)
        cv2.imshow(window_hud, hud)
        cv2.imshow(window_depth, depth_colormap)
        
        # ===== 키 입력 처리 =====
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('s') or key == ord('S'):  # 수동 저장
            if current_result.is_valid:
                data_count += 1
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                csv_writer.writerow([
                    timestamp, f"{elapsed:.1f}",
                    f"{current_result.neck_2d:.2f}", f"{current_result.neck_3d:.2f}", 
                    f"{current_result.neck_diff:.2f}",
                    f"{current_result.back_2d:.2f}", f"{current_result.back_3d:.2f}", 
                    f"{current_result.back_diff:.2f}",
                    "manual"
                ])
                csv_file.flush()
                stats_tracker.add(current_result.neck_diff, current_result.back_diff)
                print(f"💾 [수동저장] Neck Δ={current_result.neck_diff:.1f}° | "
                      f"Back Δ={current_result.back_diff:.1f}°")
        elif key == ord('+') or key == ord('='):  # 간격 증가
            CONFIG.DATA_COLLECTION_INTERVAL = min(120.0, CONFIG.DATA_COLLECTION_INTERVAL + 5.0)
            print(f"⏱️  수집 간격: {CONFIG.DATA_COLLECTION_INTERVAL:.0f}초")
        elif key == ord('-') or key == ord('_'):  # 간격 감소
            CONFIG.DATA_COLLECTION_INTERVAL = max(5.0, CONFIG.DATA_COLLECTION_INTERVAL - 5.0)
            print(f"⏱️  수집 간격: {CONFIG.DATA_COLLECTION_INTERVAL:.0f}초")
    
    # ===== 종료 처리 =====
    csv_file.close()
    device.close()
    cv2.destroyAllWindows()
    
    # 통계 출력
    print("\n" + stats_tracker.get_summary_string())
    print(f"\n✅ 연구 데이터 수집 완료!")
    print(f"   총 데이터: {data_count}개")
    print(f"   저장 파일: {log_file}")
    print("=" * 60)


# ==============================================================================
# 엔트리 포인트
# ==============================================================================

if __name__ == "__main__":
    main()