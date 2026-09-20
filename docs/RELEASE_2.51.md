# 2.51 — LibreYOLO ONNX 내보내기 수정

LibreYOLO MobileNetV4 Small, LibreYOLO9 Tiny, Re-DETR v4 체크포인트가 기존 DVS ONNX exporter에서 미지원 포맷으로 거부되던 오류를 수정했다.

내보내기는 LibreYOLO의 공개 exporter를 사용한다. 파일을 만들기 전에 seeded 입력과 zero 입력에서 native PyTorch 출력과 ONNX Runtime 출력을 비교한다. 검증을 통과한 ONNX와 JSON만 최종 경로로 교체한다.

C++ SDK는 LibreYOLO 분류의 logits, LibreYOLO9의 `pixel_xyxy + class probability` 출력, Re-DETR v4의 `pred_logits + pred_boxes` 출력을 각각 읽도록 계약을 추가했다.
