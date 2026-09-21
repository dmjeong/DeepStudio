# Deep Vision Studio 5.95

EfficientNet CPU 최적화와 GPU 학습 환경을 분리해 보장한다. CPU 벤치마크와 ONNX Runtime 설정은 배포 추론용이다. 학습 화면에서 `자동 (GPU 우선)` 또는 GPU 이름을 선택하면 모델과 배치를 CUDA로 옮기고 지원 GPU에서 AMP를 사용한다.

`gui\build.bat`은 NVIDIA GPU를 감지하면 드라이버에 맞는 공식 CUDA PyTorch를 자동 설치한다. 설치 직후 실제 GPU 합성곱과 역전파를 검사하며 실패하면 CPU로 조용히 전환하지 않고 빌드를 중단한다. `build.bat gpu`는 CUDA를 필수로 요구하고 `build.bat cpu`는 CPU 전용 빌드를 명시한다.

Windows 설치파일 검증본은 CUDA 13.0 PyTorch를 동결하고 `torch_cuda.dll`, `c10_cuda.dll` 포함 여부를 검사한다. GPU가 없는 GitHub 러너에서는 CUDA DLL 포함과 CPU 실행을 검증하며 실제 GPU 학습 완료 검증은 대상 Windows GPU PC에서 학습 시작 시 수행한다.
