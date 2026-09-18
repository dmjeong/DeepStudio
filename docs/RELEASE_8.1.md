# Deep Vision Studio React 8.1

v8.0 Windows 시작 배치에서 명령 일부가 잘려 `0`, `nish` 등을 실행하려는 오류를 수정한다.

이전 Windows 검사는 Git 체크아웃 후의 배치 파일을 실행했지만, 배포 ZIP은 Linux 파일의 LF 줄바꿈을 그대로 담았다. 검사한 파일과 배포한 파일이 다른 바이트였던 검증 누락이다.

- 시작 배치를 BOM 없는 ASCII와 CRLF 줄바꿈으로 고정한다. Git 속성과 ZIP 패키징 단계에서 각각 적용한다. 한글 사용자 안내는 Python 실행기에 유지한다.
- 실행 종료 코드는 Python 실행 직후 보관한다.
- 배포 예정 ZIP을 Windows에서 그대로 풀어 CP949와 UTF-8 명령창에서 실행한다. 한글, 공백, 괄호가 포함된 경로도 사용한다.
- 같은 배포 배치로 자동 설치와 재실행을 검사하고, 실제 서버의 상태 API, 첫 페이지, React 정적 파일의 HTTP 응답까지 확인한다.
- Windows 검증을 통과한 ZIP을 재포장 없이 공개한다. 별도 시작 파일은 검증된 ZIP에서 추출한 동일 바이트다.

이미 v8.0을 받았다면 이 릴리스의 `start_web.bat`만 기존 `DeepVisionStudio` 폴더에 덮어쓸 수 있다. 기존 가상환경과 데이터는 그대로 사용할 수 있다. 전체 업데이트는 `Deep-Vision-Studio-React-v8.1.zip`을 사용한다.
