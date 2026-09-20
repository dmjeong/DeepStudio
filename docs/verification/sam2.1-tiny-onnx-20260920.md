# SAM2.1 Hiera Tiny 공식 checkpoint ONNX 검증

검증일: 2026-09-20

실행 환경: macOS ARM64, Python 3.11.15, PyTorch 2.14.0, ONNX 1.22.0,
ONNX Runtime 1.29.0, Meta SAM-2 source revision `2b90b9f`.

공식 `facebook/sam2.1-hiera-tiny` checkpoint를 `sam2_assets`로 내려받아 확인했다.
파일 크기는 156,008,466 bytes였다. `load_sam2_pretrained`가 공식 `build_sam2`를 통해
`SAM2Base`, `image_size=1024`, `sam_prompt_embed_dim=256` 모델을 생성했고, 입력
`[1,3,1024,1024]`의 실제 image encoder를 실행했다.

`export_official_sam2_checkpoint(..., verify=True)`로 생성한 산출물은 다음과 같다.

| 파일 | 크기 |
| --- | ---: |
| `sam2_encoder.onnx` | 약 104 MB |
| `sam2_decoder.onnx` | 약 16 MB |
| `sam2.json` | 약 2.2 KB |

exporter는 `onnx.checker`를 실행하고, encoder와 decoder의 raw FP32 출력을 ONNX Runtime
CPU Execution Provider와 비교했다. decoder는 1·2·3·8개의 prompt point와 mask prompt
미사용 경로를 검사했고 검증 결과는 `passed`였다. 이 실행은 공식 Hiera Tiny의 실제
checkpoint를 사용했다.

고정 1024×1024 encoder graph를 export했기 때문에 PyTorch tracer가 input shape 분기에
대한 경고를 낸다. 이는 SAM2의 고정 1024 deployment contract와 일치한다. Windows C++17
SDK가 이 **실제 Hiera Tiny export**를 읽어 prompt 추론한 실기 결과와, Small/Base+/Large
각 변형의 실기 결과는 아직 이 문서에 없다. fixture 기반 C++ 계약 테스트를 그 결과로
대체하지 않는다.
