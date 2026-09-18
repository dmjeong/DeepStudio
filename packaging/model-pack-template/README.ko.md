# Docker 모델 팩 템플릿

이 디렉터리는 새 모델을 Deep Vision Studio에 추가할 때 복사해서 쓰는 최소 팩이다.
`worker.py`는 호스트 Python이나 프로젝트 모듈을 import하지 않으며, `DVW1` 프레임만
stdin/stdout으로 주고받는다. 실제 모델 의존성·체크포인트·ONNX exporter는 Docker 이미지에
넣고, 호스트에는 `.dvmodel` 팩과 검증된 manifest만 설치한다.

## 추가 순서

1. 디렉터리를 새 모델 이름으로 복사한다.
2. `manifest.json`의 `model_id`, `family`, `variant`, `task`, 입력 계약과 capability를 바꾼다.
   이미지 참조는 태그가 아니라 `@sha256:<64자리 digest>`여야 한다.
3. `worker.py`의 `prepare`, `train`, `infer`, `export`를 구현한다. 큰 이미지·체크포인트는
   프레임 payload에 넣지 말고 `/data` 입력과 `/work` 출력 경로를 사용한다.
4. Dockerfile을 수정하고 이미지를 빌드한 뒤 digest를 manifest와 맞춘다.
5. 개발 중에는 다음처럼 서명 없는 팩을 만든다.

   ```sh
   python tools/build_model_pack.py packaging/model-pack-template \
     work/example_classifier.dvmodel --allow-unsigned
   python tools/install_model_pack.py work/example_classifier.dvmodel \
     --root "$HOME/.local/share/DeepVisionStudio/models" --allow-unsigned
   ```

6. 앱의 모델 관리 화면에서 설치된 절대 팩 경로를 선택하고 `팩 학습`, `팩 추론`,
   `팩 ONNX export`를 실행한다. 앱은 팩 코드를 호스트에서 import하지 않는다.

출시 팩은 외부 서명과 `THIRD_PARTY_NOTICES.md`, `licenses/`를 포함해야 한다. 템플릿의
`release_status`는 `requested`로 남겨 두며, 실제 Windows·ONNX·SDK 검증이 끝난 뒤에만
`release_ready`로 올린다. 템플릿의 placeholder worker는 제품 모델 구현이 아니다.
