# Deep Vision Studio Desktop 7.2

GPU 학습을 복구하고 CUDA 포함 배포 파일의 GitHub 크기 제한 문제를 수정합니다.

- CUDA 13.0 PyTorch 포함. NVIDIA 드라이버 580 이상이 필요합니다.
- GPU 요청 시 CPU로 몰래 전환하지 않고 실제 GPU 합성곱과 역전파를 먼저 검사합니다.
- 대용량 ZIP 대신 자동 압축 해제 EXE로 배포합니다. 사용자의 PC에는 Python, pip, 7-Zip을 따로 설치할 필요가 없습니다.

실행 방법: Deep-Vision-Studio-Desktop-v7.2.exe를 실행해 새 폴더에 압축을 푼 뒤, DeepVisionStudio 폴더의 DeepVisionStudio.exe를 실행합니다. 학습 장치는 자동 (GPU 우선) 또는 GPU 이름을 선택합니다.

배포 검사는 실제 자동 압축 해제 EXE 실행, CUDA DLL 포함 여부, 앱 실행, 모델 로딩, 추론과 결과 복원을 확인합니다. 검증 서버에 GPU가 없어 실제 CUDA 학습 완료 여부는 해당 서버에서 검사하지 않습니다. GPU PC에서는 학습 시작 시 실제 CUDA 연산 검사가 자동 실행됩니다.
