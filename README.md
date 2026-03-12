# AI Posture Coach Capstone Project

실시간 영상 기반 자세 교정 캡스톤 프로젝트입니다.
웹캠 또는 OAK-D Pro 입력을 받아 사용자의 자세를 분석하고, **거북목(CVA)** 및 **굽은 허리(Torso-Thigh angle)** 상태를 감지하여 **시각 경고**, **TTS 음성 안내**, **비프음 알림**, **HUD 표시**, **세션 로그 저장**, **성공 복귀 애니메이션**을 제공합니다.

---

## 1. Project Overview

장시간 앉아서 공부하거나 작업하는 사용자는 무의식적으로 거북목이나 굽은 자세를 유지하기 쉽습니다.
이 프로젝트는 실시간 카메라 영상을 바탕으로 상체 자세를 추정하고, 사용자의 목과 허리 상태를 지속적으로 분석하여 잘못된 자세가 일정 시간 이상 유지될 경우 즉각적인 피드백을 제공하는 것을 목표로 합니다.

핵심적으로 다음 두 가지 지표를 사용합니다.

* **Neck Angle (CVA, Craniovertebral Angle)**

  * 어깨와 귀 좌표를 기반으로 거북목 여부를 판단
* **Back Angle (Torso-Thigh Angle)**

  * 어깨, 엉덩이, 무릎 좌표를 기반으로 상체 숙임/허리 굽음 여부를 판단

이 프로젝트는 초기 프로토타입에서 시작하여, 단계적으로 기능을 확장하며 완성도를 높인 형태의 캡스톤 프로젝트입니다.

---

## 2. Main Features

### Core Features

* 실시간 카메라 입력 처리
* MediaPipe Pose 기반 인체 랜드마크 추출
* Neck(CVA), Back(Torso-Thigh) 각도 계산
* 최근 프레임 평균 기반 스무딩 처리
* 자세 상태 분류

  * `GOOD`
  * `NECK WARN!`
  * `SLUMP WARN!`
  * `NO PERSON`

### Alert Features

* 나쁜 자세가 일정 시간 이상 지속될 때만 경고 발생
* 화면 중앙 또는 HUD 경고 패널 표시
* TTS 음성 안내
* 비프음 알림
* 자세 에피소드 단위 경고 제어

### Extended Features

* 카메라 창과 HUD 창 분리
* 세션 시간 / 좋은 자세 시간 / 나쁜 자세 시간 / 사람 미검출 시간 표시
* 경고 횟수 집계
* CSV 로그 저장
* 세션 종료 후 요약 화면 출력
* 좋은 자세 복귀 성공 애니메이션 표시
* OAK-D Pro 카메라 입력 지원

---

## 3. Tech Stack

* **Python**
* **OpenCV**
* **MediaPipe Pose**
* **NumPy**
* **pyttsx3** *(optional, TTS)*
* **winsound** *(optional, Windows beep sound)*
* **DepthAI** *(optional, OAK-D Pro support)*

---

## 4. Project Structure

```text
Ai-posture-coach_capstone_project/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ posture_coach_step1.py
├─ posture_coach_step2_mvp.py
├─ posture_coach_step3_tts_episode.py
├─ posture_coach_step4_hud_tts_log_version1.py
├─ posture_coach_step4_hud_tts_log_version2.py
├─ posture_coach_step4_hud_tts_log_success_anim.py
└─ posture_coach_step4_hud_tts_log_success_anim_oakd.py
```

> 현재 로컬 파일 중 `posture_coach_step4_hud_tts_log_success_anim.py.py` 는 GitHub에 업로드할 때
> `posture_coach_step4_hud_tts_log_success_anim.py` 로 이름을 정리하는 것을 권장합니다.

---

## 5. Development Flow

이 프로젝트는 다음과 같은 순서로 발전했습니다.

1. **기본 자세 각도 측정 프로토타입 구현**
2. **MVP 수준의 자세 상태 분류 추가**
3. **TTS / 비프음 / 경고 에피소드 처리 추가**
4. **세션 타이머 / 로그 저장 / 요약 화면 추가**
5. **카메라 창과 HUD 창 분리**
6. **좋은 자세 복귀 시 성공 애니메이션 추가**
7. **OAK-D Pro 입력 지원**

즉, 단순한 자세 측정 도구에서 시작하여, 사용자 피드백과 기록 기능을 갖춘 실시간 자세 코칭 시스템으로 발전한 프로젝트입니다.

---

## 6. File Description

### `posture_coach_step1.py`

초기 프로토타입 버전입니다.

* 웹캠 입력 + MediaPipe Pose 기반 자세 추정
* `calculate_angle()` 함수로 각도 계산
* 목 각도와 허리/상체 각도 표시
* 최근 프레임 평균을 이용한 스무딩 처리
* 기초적인 실시간 자세 측정 기능 구현

### `posture_coach_step2_mvp.py`

MVP 버전입니다.

* `GOOD`, `NECK WARN!`, `SLUMP WARN!`, `NO PERSON` 상태 분류
* 임계값 기반 자세 판별 로직 적용
* 화면 하단 상태 텍스트 표시
* 상태에 따른 테두리/텍스트 색상 변경
* 사용자 피드백 구조 정리

### `posture_coach_step3_tts_episode.py`

경고 시스템이 추가된 버전입니다.

* 나쁜 자세 지속 시간 추적
* 일정 시간 이상 나쁜 자세가 유지될 때만 경고 발생
* TTS 음성 안내 추가
* 비프음 알림 추가
* 경고 에피소드 단위 제어
* 화면 중앙 경고 박스 표시

### `posture_coach_step4_hud_tts_log_version1.py`

기능 확장 1차 버전입니다.

* 세션 시간 추적
* GOOD / BAD / NO PERSON 시간 누적
* `neck_warn_count`, `slump_warn_count` 집계
* `posture_log.csv` 로그 저장
* 세션 종료 후 요약 화면 출력

### `posture_coach_step4_hud_tts_log_version2.py`

UI 개선 버전입니다.

* 카메라 창과 HUD 창 분리
* HUD에 각도, 경고 지속 시간, 세션 시간, GOOD/BAD/NO PERSON 시간 표시
* 경고 횟수 표시
* HUD 내부 경고 패널 추가
* 전체 사용자 인터페이스 정리

### `posture_coach_step4_hud_tts_log_success_anim.py`

복귀 성공 피드백이 추가된 버전입니다.

* 나쁜 자세 이후 좋은 자세를 안정적으로 유지하면 성공 애니메이션 표시
* 초록 체크, 원형 효과, `GOOD!` 텍스트 등 긍정적 피드백 제공
* HUD 기반 피드백 강화

### `posture_coach_step4_hud_tts_log_success_anim_oakd.py`

OAK-D Pro 지원 버전입니다.

* DepthAI 기반 OAK-D Pro 입력 처리
* 기존 HUD / 로그 / 성공 애니메이션 기능 유지
* 일반 웹캠 대신 OAK-D Pro 장비 사용 가능

---

## 7. Posture Decision Logic

### Neck Posture

* **Neck(CVA) < 50°** → 거북목 경고
* **Neck(CVA) >= 50°** → 정상 범위

### Back Posture

* **Torso-Thigh Angle < 90°** → 굽은 자세 경고
* **Torso-Thigh Angle >= 90°** → 정상 범위

### Alert Condition

* 나쁜 자세가 **3초 이상** 지속될 경우 경고 발생
* 자세가 다시 정상으로 돌아오면 경고 상태를 초기화
* 일부 버전에서는 좋은 자세 복귀 후 성공 애니메이션 표시

---

## 8. Installation

### Basic Installation

```bash
pip install opencv-python mediapipe numpy pyttsx3
```

### For OAK-D Pro Support

```bash
pip install depthai
```

---

## 9. How to Run

### Run basic versions

```bash
python posture_coach_step1.py
python posture_coach_step2_mvp.py
python posture_coach_step3_tts_episode.py
python posture_coach_step4_hud_tts_log_version1.py
python posture_coach_step4_hud_tts_log_version2.py
python posture_coach_step4_hud_tts_log_success_anim.py
```

### Run OAK-D Pro version

```bash
python posture_coach_step4_hud_tts_log_success_anim_oakd.py
```

---

## 10. Log File

Step 4 이상 버전에서는 세션 종료 후 `posture_log.csv` 파일이 생성될 수 있으며, 일반적으로 다음과 같은 정보가 저장됩니다.

* 실행 시각
* 총 세션 시간
* 좋은 자세 시간
* 나쁜 자세 시간
* 사람 미검출 시간
* 좋은 자세 비율
* 목 경고 횟수
* 허리 경고 횟수

이를 통해 단순 실시간 피드백뿐 아니라, 세션 단위의 자세 관리 기록도 확인할 수 있습니다.

---

## 11. Expected Applications

* 장시간 공부하는 대학생 자세 관리
* 장시간 컴퓨터 작업 환경에서의 자세 교정
* 개인용 자세 모니터링 시스템
* 실시간 컴퓨터 비전 기반 헬스케어/웰니스 프로젝트
* OAK-D Pro 기반 스마트 자세 코칭 시스템 확장

---

## 12. Limitations

* MediaPipe landmark 추정 정확도는 카메라 각도와 조명 환경에 영향을 받음
* 측면 자세 인식 품질은 사용자 위치와 회전 정도에 따라 달라질 수 있음
* 단일 카메라 기반이므로 깊이 정보가 제한적일 수 있음
* 현재 코드는 단계별 실험/확장 구조라 모듈화가 충분하지 않음

---

## 13. Future Improvements

* 코드 모듈화 및 리팩토링
* `config.py` 등 설정값 분리
* GUI 앱 형태로 패키징
* 로그 분석 및 통계 시각화 대시보드 추가
* 사용자별 맞춤 피드백 기능 추가
* 자세 정확도 검증 실험 및 평가 지표 정리
* 다중 센서 또는 깊이 정보 기반 자세 분석 확장

---

## 14. Conclusion

AI Posture Coach는 단순한 자세 측정에서 출발해, 실시간 자세 상태 판단, 경고 시스템, 로그 저장, HUD, 성공 피드백, 하드웨어 확장까지 단계적으로 발전한 캡스톤 프로젝트입니다.

특히 이 프로젝트의 강점은 **단계별 기능 확장 과정이 명확하게 남아 있다는 점**입니다.
따라서 GitHub에는 단순 결과물만 올리기보다, **step 기반 개발 흐름 자체를 프로젝트의 핵심 스토리로 정리하는 방식**이 가장 적절합니다.

---

## 15. Author

* **박은성**
* 제주한라대학교 인공지능학과
* Capstone Project: **AI Posture Coach**
