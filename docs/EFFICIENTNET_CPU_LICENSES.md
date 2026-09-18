# EfficientNet CPU 경로의 라이선스 범위

확인일: 2026-09-17. 이번 변경의 독립 CPU 벤치마크/추론 경로에 대한 기술적 점검이다.
전체 기존 저장소, 고객 데이터, 고객 가중치에 대해 법적 무위험을 보증하는 문서가 아니다.

## 코드와 배포 라이브러리

| 항목 | 확인한 조건 | 배포 시 처리 |
|---|---|---|
| 로컬 EfficientNet 구조 | torchvision v0.23.0의 BSD-3-Clause 기반 | `python/EFFICIENTNET_NOTICE.txt` 원문 유지, 바이너리 배포물에도 포함 |
| ONNX Runtime | MIT | 설치 패키지의 `LICENSE`와 `ThirdPartyNotices.txt` 보존 |
| OpenVINO | Apache-2.0 및 포함된 제3자 구성요소 조건 | 설치 패키지의 LICENSE 및 licensing 하위 고지 함께 보존 |
| NumPy | BSD 계열 및 번들 구성요소 조건 | 실제 설치 wheel의 licenses 디렉터리 보존 |
| OpenCV | 본체 Apache-2.0; wheel에는 제3자 바이너리 포함 | 실제 wheel의 LICENSE와 LICENSE-3RD-PARTY 확인/보존 |
| PyTorch / ONNX | 각각 BSD 계열 / Apache-2.0 및 제3자 조건 | 벤치마크/내보내기 환경에만 필요; 실제 배포 wheel 조건 확인 |

공식 근거:
- [torchvision v0.23.0 LICENSE](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE)
- [ONNX Runtime LICENSE](https://github.com/microsoft/onnxruntime/blob/main/LICENSE)
- [OpenVINO LICENSE](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE)
- [OpenCV Python Licensing](https://github.com/opencv/opencv-python#licensing)
- [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- [ONNX LICENSE](https://github.com/onnx/onnx/blob/main/LICENSE)

이 라이선스들은 조건을 준수하는 상업적 사용을 허용한다. 단, OpenCV Python 공식 설명은
wheel에 LGPLv2.1 FFmpeg가 포함된다고 명시한다. **headless 설치만으로 LGPL 구성요소가
없어지는 것은 아니다.** 폐쇄형 제품의 바이너리 배포에서는 실제 포함 라이브러리의 고지,
소스 제공 및 재링크/교체 관련 의무 등 적용 조건을 확인해야 한다. 고지 파일 복사만으로
모든 조건을 충족했다고 단정하면 안 된다. LGPL 번들 자체를 제외해야 하는 제품은
FFmpeg/Qt 없이 빌드한 OpenCV와 그 빌드의 의존성을 별도로 검증한다.

## 검토 범위

사용자 요청에 따라 가중치와 학습 데이터의 권한 검토는 이번 작업에서 제외했다.
이 문서는 코드와 실행 라이브러리 조건만 다룬다.

## 현재 코드 범위

사용하지 않는 외부 학습·다운로드·전처리 어댑터는 현재 소스에서 제거했습니다.
EfficientNet 고지와 실제 배포 라이브러리의 원문 라이선스는 유지합니다. 소스 이름 정리가
전체 저장소 또는 제3자 바이너리의 법적 검토를 대신하지는 않습니다.
