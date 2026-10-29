# 설계 및 참고

Cognex VisionPro Deep Learning Red Analyze의 Review Markings 공식 설명은 defect outline, heatmap, 표시 체크박스를 구분한다:
https://docs.cognex.com/deep-learning_320/web/EN/deep-learning/Content/deep-learning-Topics/training-red-foc-sup/marking.htm
동일 제품 그래픽/코드를 복제하지 않고 원본과 검출 영역 구분이라는 동작을 참고한다.

- 배경 ID 0은 원본 바이트 그대로 둔다. 전경 영역 안쪽만 약한 클래스색 혼합, 경계도 영역 안쪽에 그린다. 원본 배열은 불변.
- 연결된 전경 영역별 클래스명/면적/원본 좌표 bbox를 기록. 배경 객체나 허위 confidence는 만들지 않는다. 마스크 자체는 모두 표시하고 화면 이름만 큰 영역 256개로 제한해 작은 잡음에 UI가 멈추지 않게 한다. 전체 개수와 표시 제한은 명시한다.
- Qt 이미지 레이어에서 클래스 이름을 그려 OS 한글 폰트와 확대/이동을 지원한다. 래스터 원본에 글씨를 굽지 않는다. 영역 이름 체크박스와 기존 원본 비교를 제공.
- 분할 모델 로드시 Grad-CAM을 기본 해제한다. 선택하면 기존의 별도 반응 맵을 계산/표시한다. 분류 기본값과 모델 출력은 변경하지 않는다.
- 예측 영역 표시는 GT 없이도 항상 작동하며 IoU 평가 상태와 독립이다. 캐시와 프로세스 결과 JSON에 영역 좌표를 보존한다.
