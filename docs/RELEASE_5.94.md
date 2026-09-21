# Deep Vision Studio 5.94

LibreYOLO 분류 학습에서 빈 검증 클래스 폴더를 자동으로 처리한다.

Deep Vision Studio의 새 분류 프로젝트는 `train`, `val`, `test` 아래에 클래스 폴더를 미리 만든다. 기존 기본 모델은 `val`이 비어 있으면 `train`을 자동 분할했지만 LibreYOLO 1.5.0은 빈 클래스 폴더를 `torchvision.ImageFolder`에 전달해 `Found no valid file for the classes ...` 오류로 중단했다.

5.94는 LibreYOLO 학습 직전에 전용 임시 ImageFolder 보기를 만든다. 명시적인 검증 이미지가 있는 클래스는 그대로 사용하고, 비어 있는 클래스만 프로젝트의 검증 비율에 따라 결정론적으로 분할한다. 같은 드라이브에서는 하드 링크를 사용하며 지원되지 않는 파일시스템에서만 복사한다. 학습이 끝나면 임시 데이터는 삭제되고 원본 이미지와 폴더는 변경되지 않는다.

자동 분할이 필요한 클래스에는 학습 이미지가 최소 2장 있어야 한다. 부족하면 해당 클래스 이름과 폴더를 포함한 오류를 학습 전에 표시한다.
