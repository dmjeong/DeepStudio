# 17.90 배포·프로젝트 클래스 생성 결과

## 제공 기능

- Data Gen의 첫 탭에서 프로젝트에 등록한 클래스를 읽고, 클래스별 생성 방식·장수·크기·강도를 지정한다.
- 분류, 세그멘테이션, 일반 검출, 회전 검출, 이상검출에 맞는 이미지·정답을 저장한다.
- 원본 비교와 정답 검수 후 승인한 PNG를 별도 프로젝트로 저장한다. 원본 프로젝트는 유지한다.
- SEG의 배경/기존 정답/ignore 영역, 검출의 기존 박스, 중앙 크롭 판정 영역을 보존한다.
- 분류·SEG·검출의 합성 결과는 새 Train에만 추가한다. 이상검출 합성 불량은 정상 학습에 섞지 않고 별도 합성 평가 Test에 저장한다.
- [패치 학습 그림과 코드](../PATCH_TRAINING_EXPLAINED.md), [생성 기능 사용법](../PROJECT_CLASS_GENERATION.md), [배포 방법](../RELEASE_17.90.md)을 제공한다.

## 수정한 배포 문제

모델 hidden import 검색 경로, timm 등 신규 모델 의존성, Windows 가중치 무결성 캐시, C++ 예제 DLL 배치, PowerShell 인자 전달, WiX 확장 준비와 버전, 큰 설치본의 ZIP64 패키징을 수정했다. 설치 검사 명령의 공백 경로 인용 오류도 실제 Windows MSI 검사로 수정했다. 자세한 원인과 증거는 [검증 기록](../03-analysis/project-generation-release.md)에 있다.

## 실제 검사

| 검사 | 결과 |
|---|---|
| macOS CPU 전체 회귀 | 1,656개 통과 / 78개 조건부 생략 |
| Windows 전체 회귀 | 1,657개 통과 / 77개 조건부 생략 / 236개 하위 검사 |
| 5개 태스크 생성 UI | 별도 작업 프로세스 생성 → 원본/정답 표시 → 승인 → 저장 → 재열기 통과 |
| Windows C++/C# | VS2017 C++17 Release·Debug 공개 예제, C# SDK/예제 통과 |
| modern dense 모델 배포 | Windows FP32·INT8·암호화·C++ 출력 비교 35개 통과 |
| Windows EXE | 모델 모듈 import, CPU 추론, Grad-CAM, 결과 복원, 기존/신규 생성 저장 통과 |
| 실제 전체 앱 MSI | 한글 경로 설치 → 설치된 EXE 실행 → 제거 통과. 설치/제거 반환값 0 |
| 작은 두 Setup 형식 | 단일 Setup과 외부 CAB/ZIP Setup 각각 설치·제거 통과 |
| 별도 Windows 환경의 설치본 재검사 | 앱 학습 라이브러리를 pip 설치하지 않은 새 runner에서 설치·CPU 추론·생성·제거 통과. 실제 Windows 플러그인과 한글 표시 확인 |

Windows 설치본 실행에서는 개발용 Python·Qt·SDK 환경 변수와 PATH를 제거했다. 설치 로그에서 MSVC 런타임 DLL도 앱 내부에 배포되는 것을 확인했다. 실행 보고서는 `frozen=true`, `inference_status=ok`, `gradcam_status=completed`, `persistent_results=true`, `project_class_generation_saved=true`를 기록했다.

- [전체 CI](https://github.com/dmjeong/DeepStudio/actions/runs/43200440826)
- [Windows EXE·MSI·SDK 검사](https://github.com/dmjeong/DeepStudio/actions/runs/43206601258)
- [별도 Windows 설치본 재검사](https://github.com/dmjeong/DeepStudio/actions/runs/43211949375)

## 배포 범위

Windows 빌드 PC의 `gui` 폴더에서 `build.bat installer gpu`를 실행한다. 기본 가중치가 하나라도 빠지면 배포용 설치파일 생성을 중단한다. 작은 구성은 Setup.exe 하나, 큰 구성은 Setup/MSI/CAB를 담은 오프라인 ZIP 하나를 배포한다. 대상 PC에 Python을 별도로 설치할 필요는 없다.

CI의 설치본은 **최소 가중치로 패키징을 검증한 시험 구성**이다. 모든 기본 가중치를 포함한 정식 오프라인 배포물로 소개하지 않는다. GPU 실물 학습, 사용자 영상의 정확도와 지연시간은 이 검사로 확인한 범위가 아니다. 생성 기법 역시 클래스 이름만으로 실제 불량 의미를 학습하지 않으므로 사용자 검수가 필요하다.
