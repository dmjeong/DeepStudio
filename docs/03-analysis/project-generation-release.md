# 17.90 검증 기록

## 확인된 문제와 변경

- PyInstaller 분석 프로세스에 python/ 경로가 전달되지 않아 신규 모델 hidden import가 누락됐다. 명시적 --paths와 frozen 모듈 import 검사를 추가했다.
- Windows 한글 폴더 C++ 테스트가 EXE만 복사하고 DLL은 PATH에 의존했다. 실행 파일 옆의 동일 SDK DLL을 함께 배포하도록 테스트를 수정했다.
- Windows 정답 연결 UI 회귀는 저장 작업이 끝나기 전 테스트의 5초 제한과 창 제거가 겹쳤다. 실제 작업 종료를 기다리고 모달 오류를 로그로 드러내도록 변경했다. 이후 Windows 전체 회귀에서 통과했다.
- 프로젝트 클래스/생성 기법/태스크 정답 출력이 일관되게 연결되지 않았다. 새 공통 생성 모드와 5개 태스크의 저장 경로를 구현했다.
- 중앙 Crop 설정이 있으면 검사 영역 안에만 생성하고 원본 해상도·좌표로 저장한다. 영역 안에 기존 정답이 가득 찬 이미지는 건너뛴다.
- 기존 EXE 내부 검증 코드가 개편 전 생성 화면 메서드를 호출하고 있었다. 실제 legacy 위젯과 새 프로젝트 클래스 위젯을 각각 실행하도록 고치고 같은 경로를 소스 모드에서도 검증한다.
- Windows 전체 검사에서 같은 크기로 빠르게 덮어쓴 가중치의 수정 시각이 같아져 무결성 캐시가 잘못 재사용되는 결함을 확인했다. Windows는 파일 시각 캐시 대신 실제 SHA-256을 재검사한다. 사전학습 가중치를 찾는 경로에만 적용하며 이미지별 추론 경로는 바뀌지 않는다. POSIX 캐시에는 파일 식별자와 변경 시각도 포함한다.
- Windows PyInstaller 로그에서 미사용 Qt3D/QML 모듈의 분석으로 빌드가 지연되는 것을 확인했다. 실제 사용하는 QtCore/QtGui/QtWidgets를 분석하고 PySide6의 DLL·리소스는 계속 수집한다. 미사용 Python 모듈까지 강제로 import하던 `collect-all`만 분리했다.
- 설치 스크립트가 PowerShell 배열을 이름 있는 옵션으로 전달한다고 가정해 실제로는 위치 인자로 잘못 전달했다. 해시테이블 splatting으로 고쳤다. 초기에 실패해 산출물 경로가 비어 있어도 정리 코드가 원래 오류를 덮어쓰지 않게 했다. 일반 PC에서 GITHUB_SHA가 없어도 빌드할 수 있도록 선택적 메타데이터 인자도 정리했다.
- WiX CLI 설치만으로는 확장 모듈이 설치되지 않는다. 사용 버전과 같은 확장을 준비하고, CLI에서 이름이 변경된 BootstrapperApplications 확장을 사용하도록 수정했다. 긴 GUI 빌드와 별도로 작은 실제 MSI/Setup 컴파일을 먼저 검사한다. 근거: [WiX 확장 준비](https://docs.firegiant.com/wix/tools/wixext/), [Bootstrapper 패키지 이름 변경](https://docs.firegiant.com/wix/whatsnew/oopbas/).
- 실제 작은 Windows 빌드에서 WiX 7의 OSMF EULA 미수락 오류를 확인했다. 유료 조건을 자동 수락하지 않고 MS-RL로 배포된 WiX 5.0.2로 빌드 도구를 고정했다. 라이선스 사본과 원본 소스 링크를 고지에 포함한다.
- 기본 가중치·CUDA를 모두 포함할 수 있도록 원본 payload 예산과 단일 서명 EXE 용량 제한을 분리했다. 1.5 GiB 초과는 외부 CAB + MSI + Setup의 ZIP64 오프라인 묶음이다. 작은 배포는 단일 Setup을 유지한다. 파일 해시 검증을 생략하거나 가중치를 제외하지 않는다.
- 자동 MSI 설치 검사에서 `Popen`에 목록을 넘기면 공백 경로가 `"INSTALLFOLDER=경로"`로 인용되어 msiexec가 도움말 대화상자에서 대기했다. 설치 로그가 생기기 전 900초 제한에 걸렸다. [Microsoft의 속성 인자 문법](https://learn.microsoft.com/en-us/windows/win32/msi/command-line-options)대로 `INSTALLFOLDER="경로"`를 전달하도록 변경했다. 셸은 사용하지 않는다. 한글·공백이 포함된 미디어/설치/로그 경로로 실제 작은 MSI 설치·파일 해시 확인·제거 회귀를 추가했고 Windows에서 2.92초에 통과했다.

## 로컬 확인

- 생성 엔진: 모든 태스크의 한글 클래스, 기존 정답 보존, 새 픽셀 변화, 재현 시드, 취소, 원본 불변, 파일 변조·클래스 변경 거부, 이동 후 프로젝트 재읽기와 중앙 Crop 검사.
- 실제 Qt 작업 프로세스: 5개 태스크 모두 생성 → 원본 비교 → 정답 표시 → 승인 → 새 프로젝트 저장 → 재열기 통과.
- 기존 Data Gen/SEG 참조 생성/정답 연결과 함께 11개 통과.
- 9f42cfd 전체 회귀 **1,650개 통과 / 78개 조건부 생략**, 247.85초, 정상 종료. 소스 패키지 실행 검사와 중앙 Crop 회귀를 포함한다. 이 결과는 macOS CPU이며 Windows 설치 성공을 뜻하지 않는다.
- d8bacc5 전체 회귀 **1,654개 통과 / 78개 조건부 생략**, 248.54초, 정상 종료. 실제 PowerShell 호출 검사 3개를 포함한다.
- d56f669 전체 회귀 **1,656개 통과 / 78개 조건부 생략**, 251.37초. 대용량 ZIP64 구성과 원본 payload 예산 분리 검사 포함. 실제 Windows 설치 도구의 빈 캐시 반환값(2)도 확인해 후속 수정했다.
- 소스 패키지 실행·5개 태스크 Qt 생성·중앙 Crop·설치 계약을 함께 실행한 집중 검사 **56개 통과**.
- 추가된 오프라인 가중치 누락 차단, 설치 payload README 및 모델 검색 경로 검사 통과.
- 저장소 CI 규칙 Ruff E/F/W (E501/E402 제외) 통과. 실제 UI 스크린샷 확인 및 작은 화면 스크롤 보완.

## Windows 검증

macOS 소스 실행 결과를 Windows EXE 성공으로 대신하지 않았다. CI의 최소 자산 설치 검사는 패키징 계약 검증이며, 모든 사전학습 가중치를 포함한 정식 오프라인 설치본과 구분한다.

- 9f42cfd에서 실제 frozen EXE 실행 통과: version 17.90, CUDA 13.0 DLL 포함, CPU 추론 ok, Grad-CAM completed, 결과 복원, 기존 생성·새 클래스 생성 저장 확인.
- 같은 Windows 실행에서 modern dense FP32/INT8/암호화/C++ parity 35개 통과(316.09초). 이후 installer 인자 전달 오류를 발견해 위와 같이 수정.
- PowerShell 자체로 이름 있는 인자 전달·원래 오류 보존·GitHub 환경 없는 manifest 수집/검증을 실행하는 검사 3개 추가. WiX 컴파일만 대역을 사용하며 실제 MSI 검증은 별도 Windows job에서 수행한다. 기존 설치 계약과 합계 36개 통과.
- [43df2b9 전체 CI](https://github.com/dmjeong/DeepStudio/actions/runs/43194536794) 성공. Windows 전체 회귀 1,652개 통과 / 77개 조건부 생략 / 236개 하위 검사. Linux·Windows C++ 및 기타 job도 성공.
- 3f587c7의 Qt 수집 범위를 바꾼 최종 EXE도 Windows 실제 실행을 통과했다. d8bacc5의 Windows payload 계약 107개도 통과했다. MSI 설치·실행·제거 검증은 별도로 진행한다.
- [e5febcf 전체 CI](https://github.com/dmjeong/DeepStudio/actions/runs/43200440826) 성공. Windows 전체 회귀 1,657개 통과 / 77개 조건부 생략 / 236개 하위 검사. 같은 소스의 작은 실제 Setup과 외부 CAB/ZIP Setup은 [Windows SDK 검사](https://github.com/dmjeong/DeepStudio/actions/runs/43200143779)에서 각각 설치·제거까지 통과했다. ZIP은 다른 폴더로 옮긴 뒤 원래 빌드 폴더를 삭제하고 검사했다. 이 실행의 큰 MSI 자동 설치 호출은 위 인자 인용 문제로 중단됐으며, 설치 성공으로 계산하지 않는다.
- 설치본 실행 검사의 PATH에 빌드 PC의 Python/native SDK 디렉터리가 남는 것을 제거했다. 환경 변수로 지정된 Python·Qt·모델/SDK 경로도 제외한다. 완성한 MSI/CAB를 새 runner에 가져와 학습 라이브러리를 pip 설치하지 않고 설치·추론·생성·제거를 반복하는 재검증 경로를 추가했다. 원본 installer run ID를 결과에 기록한다.
- [2113cf4 Windows 재검사](https://github.com/dmjeong/DeepStudio/actions/runs/43206601258) 전체 성공. 작은 MSI의 한글·공백 경로 검사와 두 Setup 형식의 설치·제거, 실제 전체 앱 MSI의 설치·EXE 실행·제거 모두 통과했다. 설치와 제거의 MSI 반환값은 각각 0이다. 전체 EXE는 17.90 / CUDA 런타임 13.0 / CPU 추론 ok / Grad-CAM completed / 영구 결과 복원 / 기존 및 신규 생성 저장을 확인했다. modern dense 모델 FP32·INT8·암호화·C++ 출력 비교 35개도 통과했다.
- 설치 로그에서 `app/_internal` 내부의 msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll 배치를 확인했다. 실패하더라도 만들어진 설치 미디어를 남겨 재검사에 불필요한 전체 빌드를 반복하지 않게 했다. 최초 검사는 Qt offscreen 모드였으므로 별도 설치본 재검사에서는 실제 Windows 플랫폼 플러그인을 사용한다.
- [8841cd1 별도 Windows 재검사](https://github.com/dmjeong/DeepStudio/actions/runs/43211949375) 성공. 새 runner에는 검사 도구용 psutil만 설치하고 앱의 torch/PySide6/timm 등은 pip 설치하지 않았다. 위에서 만든 MSI/CAB를 다운로드해 설치·실행·제거했고, Windows 시스템 디렉터리만 PATH에 둔 상태로 모든 실행 보고 항목이 통과했다. `installer_source_run=43206601258`, `isolated_system_path=true`를 기록했다. Qt `windows` 플랫폼으로 한글이 표시되는 실행 화면도 직접 확인했다. 검사 창은 기능 검사 코드의 기본 스타일이며 일반 앱의 다크 테마 화면과 구분한다.

## 원격 C++ 검증

- [9f42cfd CI 실행](https://github.com/dmjeong/DeepStudio/actions/runs/43191808224)의 Linux C++ job 통과.
- native CTest 7개, PatchCore export 16개, LibreYOLO/native patch 21개(2개 생략), segmentation candidate 5개, pretrained encoder 18개, modern dense FP32/INT8/암호화/native parity 35개 통과.
- C++ 공개 예제 CTest 3개 통과. 실제 사용자 검사 영상의 정확도·지연시간 측정과는 구분한다.

## 한계

생성 방식은 선택한 절차적 표면 합성이다. 클래스 이름만으로 실제 결함 의미를 학습하지 않는다. 실제 형상 결함은 기존 참조/학습형 생성 모드와 사용자 검수가 필요하다. 합성 시험 결과를 실제 검사 정확도로 주장하지 않는다. GPU 실물 학습과 사용자 PC 드라이버 조합은 별도 검증 대상이다.
