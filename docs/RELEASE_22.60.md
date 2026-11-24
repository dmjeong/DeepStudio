# 22.60 Windows 실행 파일의 Python DLL 검사

Windows EXE 빌드와 간단 설치본 생성 단계에서 EXE 옆의 `_internal`에 Python DLL이 있는지 확인합니다. 필요한 파일이 빠졌으면 해당 경로를 표시하고 빌드를 중단합니다.

기존 배포물에서 실행 오류가 나면 빌드 PC의 `gui\dist\DeepVisionStudio` 폴더 안에서 실행하세요. 다른 PC로 전달할 때는 그 폴더를 통째로 옮기거나 간단 설치본으로 설치합니다. 오프라인 ZIP을 받았다면 전체를 압축 해제한 뒤 Setup을 실행합니다.
