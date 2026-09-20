# 3.51 — SAM2.1 사전학습·실모델 ONNX 및 라벨 편집 개선

SAM2 Hiera Tiny, Small, Base+, Large는 이제 Meta의 공식 SAM2.1 저장소에서 각각의
사전학습 checkpoint를 내려받을 수 있다. Windows `build.bat`은 고정한 Apache-2.0
SAM-2 소스 revision을 설치본에 포함하며, 가중치는 사용자가 Settings에서 선택한 변형만
사용자 모델 캐시에 내려받는다.

SAM2 프로젝트에서 해당 checkpoint를 선택해 ONNX Export를 실행하면, 공식 SAM2.1
image path를 encoder와 prompt decoder 두 그래프로 export한다. `sam2_encoder.onnx`,
`sam2_decoder.onnx`, `sam2.json`은 하나의 배포 단위이며, point/box/mask prompt와
1·2·3·8개 point의 PyTorch 대 ONNX Runtime 비교를 통과한 뒤에만 생성된다.

분할 라벨 편집기는 `Shift`를 누른 브러시를 임시 지우개로 사용하고, `[`와 `]`로 브러시
크기를 조절한다. 지우개는 foreground 영역 목록에 표시되지 않으며 Undo/Redo에 포함된다.

이 릴리스가 SAM2 전체 fine-tuning이나 video memory state의 Windows/C++ 실기 검증을
뜻하지는 않는다. 해당 항목은 완료 기준을 충족하기 전까지 release-ready로 표시하지 않는다.
