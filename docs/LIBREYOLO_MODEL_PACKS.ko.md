# 기본 LibreYOLO와 추가 Docker 모델

LibreYOLO MobileNetV4 Small(분류)과 LibreYOLO9 Tiny(객체 탐지)는 Deep Vision Studio의
**기본 제공 모델**이다. 설치된 앱에서 별도 `.dvmodel`을 찾거나 Docker를 설치할 필요가 없다.

데스크톱 앱에는 같은 계열의 상위 변형도 있다. 학습·추론·ONNX 내보내기·C++ 경로는 Tiny/Small과 같다.

| 태스크 | 모델 | 매개변수 | 고르는 기준 |
| --- | --- | --- | --- |
| 객체 탐지 | LibreYOLO9 Tiny | 1.9M | 가장 빠름. CPU 추론 |
| 객체 탐지 | LibreYOLO9 Small | 7.2M | Tiny로 놓치는 결함이 있을 때 먼저 올려 볼 단계 |
| 객체 탐지 | LibreYOLO9 Medium | 20.1M | 정확도 우선. GPU 학습 권장 |
| 객체 탐지 | LibreYOLO9 Compact | 25.4M | 계열 중 가장 정확하고 가장 무겁다 |
| 분류 | MobileNetV4 Small | — | 가장 빠름 |
| 분류 | MobileNetV4 Medium | 8.4M | Small보다 정확 |
| 분류 | MobileNetV4 Large | 31.3M | 계열 중 가장 정확. 기본 입력 256 |

## 검출 패치 학습 (LibreYOLO9)

큰 영상의 작은 결함은 영상 전체를 640으로 줄이면 사라진다. 학습 화면에서 **패치 학습**을 켜면 줄이지 않고
640px 타일로 잘라 학습한다.

```
학습   영상 ─▶ (패치 배율) ─▶ 겹치는 640px 타일 ─▶ 물체가 있는 타일 전부 + 배경 타일 일부로 학습
추론   데스크톱: 같은 배율의 타일로 나눠 추론하고 합친다
       C++    : 판정 영역 × 배율 크기의 고정 입력 ONNX로 내보내 한 번에 추론한다 (Detect 호출은 그대로)
```

- 타일은 학습 동안만 쓰는 임시 데이터다. 원본 영상과 라벨은 바꾸지 않는다.
- 내보내기 화면의 **판정 영역**에는 설비에서 넘기는 영상 크기를 넣는다. 학습 영상 크기가 기본으로 채워진다.
- C++ SDK는 넘겨받은 영상을 모델 입력 크기에 맞춘다. 학습과 같은 물체 크기가 되려면 판정 영역 크기의 영상을 넘긴다.
- RT-DETR v4는 입력 크기가 고정된 구조라 지원하지 않는다.

## 증강 직접 설정

LibreYOLO 모델은 기본으로 모델에 내장된 증강을 쓴다. 학습 화면 **데이터 증강 → 직접 설정**을 켜면 화면 값을 적용한다.

| | 모델 기본 | 직접 설정 |
| --- | --- | --- |
| LibreYOLO9 | 모자이크 · 수평 뒤집기 0.5 · 색상 변화 | 수평 뒤집기 · 회전 · 색상 변화(0이면 끔) · 모자이크 켬/끔 |
| RT-DETR v4 | 수평 뒤집기 0.5 | 수평 뒤집기 · 모자이크 켬/끔 (색상 변화는 쓰지 않는다) |
| MobileNetV4 분류 | 무작위 잘라내기(면적 50~100%) · 수평 뒤집기 0.5 | 검증과 같은 구도 + 수평 뒤집기 · 회전 · 색상 변화 |

- 검출의 회전은 모자이크로 합친 영상에만 적용된다.
- 분류의 직접 설정은 무작위 잘라내기를 쓰지 않는다. 가장자리 결함이 영상 밖으로 잘려 나가지 않는다.

기본 모델은 native Windows 학습 worker, 추론, ONNX export, C++17/C# 검증을 모두 통과한 뒤에만
`release_ready` 설치본으로 출고한다. 현재 개발 카탈로그에서 `requested`로 보이면 모델 팩이
누락된 것이 아니라 해당 native 구현·인수가 아직 끝나지 않았다는 뜻이다. 임의의 Custom CSP로
대체 실행하지 않는다.

## Docker `.dvmodel`을 쓰는 경우

Docker 모델은 기본 목록 밖의 새 모델을 사용자가 추가할 때만 쓴다.

1. **Settings → 모델 관리 → 모델 추가 (.dvmodel)**을 연다.
2. 배포자가 제공한 서명된 `.dvmodel`을 선택한다.
3. 설치가 끝나면 해당 태스크의 학습 화면에서 추가 모델을 선택한다.
4. 추가 모델에서만 **팩 학습 / 팩 추론 / 팩 ONNX export**가 표시된다.

`.dvmodel`은 가중치 파일 확장자를 바꾼 것이 아니다. 학습·추론 코드, 고정 Docker 이미지,
입출력 계약, 자산 해시, 라이선스 고지와 서명을 포함한 오프라인 추가 모델 형식이다.

## 추가 모델 제작

`packaging/model-pack-template`을 복사해 모델 ID·입출력·클래스 계약과 `prepare`, `train`,
`infer`, `export` worker를 구현한다. 팩은 네트워크 없이 실행되고, C++17/C# 배포에 필요한 ONNX
전처리·후처리 계약을 명시해야 한다. 개발용 팩은 아래처럼 만들 수 있다.

```powershell
python tools\build_model_pack.py work\my-model-pack work\my-model.dvmodel --allow-unsigned
python tools\install_model_pack.py work\my-model.dvmodel --allow-unsigned
```

정식 배포 팩은 원본 라이선스·NOTICE, 자산 출처/해시와 Ed25519 서명을 포함해야 한다.
