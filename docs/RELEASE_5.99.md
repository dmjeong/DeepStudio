# Deep Vision Studio 5.99

학습 결과 CSV만으로도 Best 모델의 설정을 확인하고 재현할 수 있도록 저장 계약을 보강했다. `results.csv`의 모든 에폭 행과 `best_result.csv`의 Best 행에 동일한 `hp_...` 열을 저장한다.

기록 범위는 다음과 같다.

- 학습: epochs, batch size, input size, channels, learning rate, weight decay, optimizer, scheduler, warmup, early stopping, Best 선정 지표, label smoothing, class weights, 장치, AMP, 학습 모드
- 증강: horizontal/vertical flip, rotation, color jitter, scale range, MixUp, mosaic
- 모델: model ID, 사전학습 가중치, backbone freeze, backbone learning-rate multiplier, dropout과 구조 설정
- 데이터: validation split, 클래스 수와 클래스 이름
- 실제 적용값: 엔진, 태스크, 모델, 실제 device, AMP, worker, pinned memory 등 엔진별 적용 설정

중첩 설정은 `hp_training_augmentation_horizontal_flip`처럼 평탄화하고, 목록은 JSON 문자열로 기록한다. 같은 원본 구조는 `best_selection.json` 안의 `hyperparameters`에도 보존한다.
