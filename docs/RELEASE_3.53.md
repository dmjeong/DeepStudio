# 3.53 — 저장한 지우개 라벨 목록 회귀 수정

분할 지우개는 PNG 편집 메타데이터에 background subtraction stroke로 저장된다. 다시 연
편집기의 객체 목록은 foreground 도형과 브러시만 표시해야 한다. 브라우저 end-to-end 검증도
이 계약을 검사하도록 고쳤다: 저장 API의 `shapes`에는 erase stroke가 남고, 다시 연 목록은
foreground 두 행만 보여야 한다.
