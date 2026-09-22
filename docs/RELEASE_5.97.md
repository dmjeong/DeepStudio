# Deep Vision Studio 5.97

검증 데이터에 정답 샘플이 0개인 클래스는 그 모델이 잘하는지 평가할 근거가 없다. 이제 분류 macro Precision·Recall·F1은 정답 샘플이 있는 클래스만 평균한다. 학습 대시보드, Best epoch 선정 이력, `best_result.csv`, 학습 종료 평가가 같은 계산 기준을 사용한다.

분할은 정답 픽셀이 있는 클래스만 mIoU·Dice에 포함하고, 검출은 정답 객체가 있는 클래스만 mAP에 포함한다. 클래스별 표와 로그에서 평가할 수 없는 값은 0점이 아닌 `N/A`로 표시한다.

Accuracy와 Pixel Accuracy는 실제 정답 샘플·픽셀 전체의 정답률이므로 계산식을 유지한다. 빈 클래스로 잘못 예측한 결과는 전체 정답률과 실제 클래스의 Recall·F1·IoU·Dice에 오류로 반영된다.
