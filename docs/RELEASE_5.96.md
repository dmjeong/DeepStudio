# Deep Vision Studio 5.96

GPU 모델 연산보다 데이터 공급이 느린 학습 경로를 수정했다. CUDA 학습은 DataLoader의 pinned memory를 켜고 CPU→GPU 복사를 비동기로 수행한다. worker가 있는 경우 에폭이 바뀌어도 프로세스를 유지하고 다음 배치 두 개를 미리 준비한다.

분류 데이터셋은 각 worker에서 JPEG/PNG 디코딩 결과를 32MB 한도로 재사용한다. 대형 원본 이미지나 큰 데이터셋에서도 캐시가 제한 없이 증가하지 않는다. 첫 에폭에는 worker 시작과 캐시 준비 비용이 포함되지만, 두 번째 에폭부터 같은 이미지를 다시 디코딩하거나 Windows worker를 다시 생성하지 않는다.

학습 로그에는 `workers`, `pin_memory`, worker 유지 및 prefetch 설정을 표시한다. GPU를 선택했는데 `pin_memory=False`로 나타나면 CUDA 장치 선택이 적용되지 않은 실행이므로 학습 장치 로그를 함께 확인해야 한다.
