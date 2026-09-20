# 3.52 — 라벨 지우기·작은 화면과 SAM2 프롬프트 ONNX 검증 수정

분할 편집기의 지우개는 foreground 객체를 추가하는 도구가 아니다. 브라우저 편집기는
배경 subtraction stroke를 저장하지만 객체 목록에는 표시하지 않으며, 실행 취소·다시 실행과
저장 후 다시 열기에서 같은 마스크 결과를 재현한다. Shift를 누른 브러시도 같은 지우개로
동작하고 `[`·`]`로 브러시 크기를 조절할 수 있다.

검출 편집기의 직접 조작 도구는 900 px 폭에서 창을 확대하지 않고 세 줄 그리드로 배치한다.
100%, 125%, 150% Qt 배율에서 버튼 겹침, 캔버스 크기, 박스 생성 동작을 검사했다.

SAM2 exporter는 공식 Hiera Tiny checkpoint를 다시 export하면서 다음 prompt 계약 각각을
PyTorch FP32와 ONNX Runtime CPU 출력으로 비교한다: 빈 프롬프트, positive/negative point,
box를 표현하는 label 2/3, 8개 mixed point, 이전 low-resolution mask refinement. 이 검증은
SAM2 encoder/decoder 두 ONNX와 `sam2.json` 배포 묶음에 적용된다. 다른 Hiera 변형과 실제
Windows C++ 실행은 별도 실기 검증 대상이며 완료로 표시하지 않는다.
