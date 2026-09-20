# 기본 LibreYOLO와 추가 Docker 모델

LibreYOLO MobileNetV4 Small(분류)과 LibreYOLO9 Tiny(객체 탐지)는 Deep Vision Studio의
**기본 제공 모델**이다. 설치된 앱에서 별도 `.dvmodel`을 찾거나 Docker를 설치할 필요가 없다.

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
