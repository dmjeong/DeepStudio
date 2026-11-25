# 11.83 — LibreYOLO Grad-CAM / 원본 비교

## 원인

- upstream `.pt` 로더는 Grad-CAM 객체를 만들지 않았고, 모델 검사 UI도 모든 LibreYOLO 모델을 일괄 비활성화했다.
- upstream 추론은 원본 비교용 `_original_preview_rgb`를 채우지 않았다. 특히 in-process worker의 캐시에 원본이 누락되어 원본 파일에 의존했다. 별도 프로세스 경로는 원본을 다시 읽는 우회 처리가 이미 있었다.
- 원본 비교 체크박스는 상세 화면의 원본 패널만 켰다. 격자 화면 자체를 전환하지 않아 비교가 보이지 않았다. 원본 파일 읽기 실패 시 이전 원본 픽셀이 남는 경로도 있었다.

## 변경

- 기본 제공 LibreYOLO MobileNetV4 분류 `.pt`에 기존 GradCAM 연결. 공간 특징은 `blocks`, 점수는 softmax 이전 `classifier` 출력, 타겟은 native predict의 예측 클래스다. 풀링 후 1×1인 `conv_head`는 사용하지 않는다.
- pinned LibreYOLO 1.5.0의 실제 eval transform을 재사용. Resize 결과 크기를 기록하고 중앙 잘림을 원본 좌표로 복원한다. 보지 않은 가장자리는 원본 그대로 표시한다.
- Grad-CAM은 켠 경우에만 별도로 실행/측정한다. 판정·확률은 기존 native predict 결과를 유지한다. CAM 실패도 판정 결과를 지우지 않는다.
- 모델 검사 프로세스가 지원 여부를 UI에 전달한다. 탐지 모델에는 기존 박스를 표시하며 분류 CAM을 적용하지 않는다.
- LibreYOLO 분류/탐지 원본을 캐시에 보존. 격자에서 비교를 켜면 현재 선택 이미지의 상세 비교로 전환한다. 원본/결과 크기와 확대·이동을 맞추고 없는 원본의 이전 픽셀을 지운다.

## 사용

앱을 재시작하고 기본 LibreYOLO MobileNetV4 `.pt`를 다시 로드한다. `Grad-CAM`을 체크하고 추론을 실행한다. `원본 비교`를 켜면 왼쪽 원본, 오른쪽 결과를 본다. 이전 실행에는 CAM이 저장되어 있지 않으므로 다시 추론해야 한다. 재학습은 필요 없다. CAM 계산 시간은 추론 시간과 별도로 표시한다.

## 검증

macOS Apple M4 / Python 3.12 / LibreYOLO 1.5.0 / CPU / Qt offscreen에서 관련 테스트 **130개 통과**. 실행 로그: 로컬 `/tmp/dvs-librecam-release-tests.log`.

- 실제 LibreMobileNetV4 scratch 모델 생성 → native save → 앱 로드 → CAM 생성. 가로/세로/정사각형 원본 검사.
- native 전처리 텐서와 CAM 입력 동일, 분류 확률 동일, 모든 가중치/BN 버퍼 불변, 파라미터 gradient 미누적, 훅 해제 확인.
- CAM 끔/실패, 중앙 잘림 바깥 픽셀 보존, 작업 결과 NPZ 및 지원 여부 메타데이터 검사.
- 분류/탐지 원본 파일을 지운 후 캐시 비교, 격자 전환, 비교 화면 확대/이동 및 기존 Heatmap/Segmentation/PatchCore/Web worker 회귀 검사.
- 실제 `모델 로드` 버튼 → 검사 자식 프로세스 → Grad-CAM 체크박스 마우스 클릭 → 추론 자식 프로세스 → 히트맵·원본 비교 표시 종단 테스트 통과. 이 과정에서 기존 체크포인트 바이트 불변을 확인했다.
- 실제 Qt 화면 캡처 확인: `runs/libreyolo-gradcam-20261110/comparison.png` (합성 이미지와 scratch 가중치; 검출 품질 증거가 아님).
- 기존 Qt `invalidateFilter` deprecation 경고는 남아 있다. Windows/GPU/사용자 체크포인트는 이 환경에서 실행하지 않았다.
- CI에 pinned LibreYOLO를 설치한 뒤 실제 native 테스트를 별도 실행하는 단계 추가. 원격 CI는 아직 실행하지 않았다.

이 소스에는 앞서 검증한 11.82 C++ 구간 측정 기능도 포함된다. 적용 시 SDK와 호출 프로젝트를 함께 재빌드해야 한다. Windows 실행파일을 새로 빌드한 결과는 포함하지 않는다.
