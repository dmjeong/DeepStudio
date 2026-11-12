# 17.90 — 프로젝트 클래스 생성과 Windows 배포 확인

신규 태스크별 데이터 생성 기능이므로 16.90에서 +1.0입니다.

- [프로젝트 클래스 생성 사용법](PROJECT_CLASS_GENERATION.md): 사용자 등록 클래스 → 생성 방법 선택 → 원본/정답 검수 → 새 이미지가 포함된 독립 프로젝트.
- [패치 학습 설명](PATCH_TRAINING_EXPLAINED.md): 그림과 코드로 입력 크기·패치 배율을 구분합니다. 학습 화면 입력 크기 도움말도 수정했습니다.
- PyInstaller의 `python/` 모델 검색 경로를 명시했습니다. 모델 코드를 데이터 파일로만 넣고 hidden import가 누락되던 경로를 수정했습니다.
- C++ 예제의 이동 실행 검증은 EXE와 실제 DLL을 함께 배치합니다. PATH만 사용하면 Windows System32의 다른 ONNX Runtime을 읽을 수 있습니다.

## 설치 파일 만들기

Windows x64 빌드 PC에서 `gui` 폴더의 명령 프롬프트를 열고 실행합니다.

```bat
build.bat installer gpu
```

Python 패키지·CUDA PyTorch·사전학습 가중치를 준비하고 EXE와 설치 파일을 만듭니다. **installer 모드는 기본 가중치가 하나라도 누락되면 중단합니다.** 오류를 무시하고 배포한 뒤 대상 PC에서 다시 다운로드하게 되는 것을 막습니다. 받은 가중치는 재사용하므로 실패한 다운로드만 복구 후 다시 실행할 수 있습니다. 단순 개발용 `build.bat`은 기존처럼 누락 가중치를 경고하고 EXE를 만들 수 있습니다.

배포할 것은 생성된 `DeepVisionStudio-Setup-17.90.0-win-x64.exe`입니다. EXE 파일 하나만 `dist/DeepVisionStudio` 폴더 밖으로 빼면 DLL/리소스가 누락됩니다. 설치 프로그램은 app과 C++/C# Examples 및 고지 파일을 포함합니다.

빌드 PC에는 Python, 설치 파일 제작용 .NET/WiX가 필요합니다. 앱을 쓰는 PC에는 Python을 별도로 설치할 필요가 없습니다. GPU 학습은 호환 NVIDIA GPU와 드라이버가 필요하며, CUDA DLL 포함과 실제 GPU 학습 가능 여부는 다릅니다. CPU 추론은 GPU 없이 가능합니다. 모델별 메모리·디스크 요구량은 데이터 크기와 사전학습 가중치 구성에 따라 달라집니다.

## 검증 범위

Windows Actions에 실제 frozen EXE 실행과 MSI 설치 → 설치된 EXE 실행 → 제거 검사를 추가했습니다. 소스 Python 실행으로 대신하지 않습니다. 한글 설치 경로, 모델 모듈 import, 별도 작업 프로세스 추론, 원본 비교 자료 복원, 프로젝트 클래스 생성·저장을 확인합니다.

CI의 설치 테스트용 패키지는 최소 자산으로 패키징 동작을 확인하며 모든 모델 가중치가 포함된 정식 배포물과 구분합니다. 실제 오프라인 배포는 위 `installer` 명령의 전체 가중치 검사를 통과해야 합니다. 검사 데이터의 실제 정확도, 모든 GPU/드라이버 조합은 CI 결과만으로 보장하지 않습니다.

실행 결과와 남은 제한은 [검증 기록](03-analysis/project-generation-release.md)에 기록합니다.
