# 7.19 — LibreYOLO ONNX 배포 이름 일치 수정

## 문제와 수정 범위

LibreYOLO 내보내기는 ONNX 입력 `images`로 출력 검증을 수행했지만,
배포 JSON에는 기본값 `input_image`를 저장했다. ONNX와 JSON의 이름을
비교하는 Python·C++ 로더에서는 검증을 통과한 모델도 열리지 않았다.

ONNX의 실제 입력·출력 이름을 읽어 배포 JSON에 기록하고, 검증 실행도
실제 입력 이름을 사용한다. 지원하는 단일 입력 및 모델군별 출력 개수를
확인한 뒤 산출물을 배포한다. 분류 출력은 LibreYOLO 1.5.0의 raw logits
계약을 명시하고 로더에서 softmax를 한 번 적용한다.

## 검증 계획

- 실제 `export_checkpoint()` 경로로 만든 ONNX와 JSON의 이름을 그대로
  사용해 Python ONNX Runtime에서 열고 추론·softmax 확률을 비교한다.
- 이름을 바꾼 exporter도 처리하고, 검증 생략 시에도 이름을 기록한다.
- 지원하지 않는 입출력 개수는 기존 배포 파일을 덮어쓰기 전에 거부한다.
- 같은 export 산출물을 eVision C++ `Classifier`에서 열어 BW8 추론을
  실행한다. 테스트를 위해 JSON 이름을 수동으로 보정하지 않는다.

## 검증 결과

- 7.19 푸시 전 재확인: export·upstream·버전·Windows 패키징 계약 테스트 67개 통과.
  현재 코드로 export 산출물을 다시 생성하고 C++17 Release를 재빌드해 CTest 3/3 통과.
- Python 관련 회귀 검사 53개 통과: export 계약, LibreYOLO 어댑터·추론,
  ORT 설정, EfficientNet fallback·BN 검증, 버전 정책.
- Windows 패키징 계약 검사 31개 통과.
- C++17 Release / AppleClang / ONNX Runtime 1.29.0: CTest 3/3 통과.
  수정 전 동일 생성기로 만든 산출물은 `Tensor names do not match JSON`으로
  실패했고, 수정 후 재생성한 산출물은 이름 보정 없이 통과했다.
- LibreYOLO 1.5.0 실제 MobileNetV4-S 구조·무작위 5클래스 가중치와 pinned
  exporter에서 raw logits임을 확인했다. PyTorch/ORT 최대 절대 오차는
  1.86e-9였고, 그래프에 Softmax 노드가 없었다.
- Windows VS2022/VS2017 v141 CTest에 동일 회귀 검사를 연결했다.
  Windows 실행은 아직 확인하지 않았다. 대상 i7 처리시간 검증은 포함하지 않는다.

## 별도 확인이 필요한 범위

회귀용 작은 모델은 생산 export와 배포 로더의 연결을 검사한다. 실제 사용자
가중치의 정확도나 원래 LibreYOLO 이미지 전처리와의 동등성을 인증하지 않는다.
LibreYOLO MobileNetV4-S의 기본 검증 전처리는 bicubic resize와 center crop을
사용하며, Studio의 기존 bilinear resize 배포 설정과 다르다. 이 전처리 차이는
이번 이름 수정에 포함하지 않았으며 실제 이미지 기준의 별도 검증이 필요하다.
