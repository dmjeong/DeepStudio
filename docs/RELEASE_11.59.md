# Deep Vision Studio Desktop 11.59

영어 Windows에서 C++ 회귀용 ONNX를 생성하던 중 한글 로그를 CP1252 콘솔로 출력하지 못해 중단되는 문제가 확인되었습니다. 기본 `print` 로그에 한해 출력할 수 없는 문자만 이스케이프합니다. CP949에서 경로에 이모지가 있는 경우도 처리합니다.

모델 수치 검증·ONNX·JSON 생성은 그대로 수행하며, 진단 파일의 원본 Unicode와 앱이 전달한 사용자 지정 로그 콜백은 보존합니다. CP1252/CP949 스트림에서 실제 ONNX export를 실행하는 회귀 테스트를 추가했습니다. Windows SDK CI에는 UTF-8 로그 환경도 명시했습니다.
