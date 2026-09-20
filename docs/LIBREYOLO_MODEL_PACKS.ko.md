# LibreYOLO와 .dvmodel 추가 방법

현재 LibreYOLO 분류·검출은 **카탈로그와 컨테이너 연결 규약만 있고, 즉시 사용할 완성 팩은
저장소·기본 설치본에 포함되지 않는다.** 일반 사용자가 받을 `.dvmodel` 다운로드도 제공하지 않는다.
모델 이름이 보이는 것과 실제 학습 구현이 설치된 것은 다르다. 템플릿을 그대로 설치해도
학습·추론·ONNX export는 `not_implemented`로 실패한다.

`.dvmodel`은 Studio 모델 팩 설치 파일이다. LibreYOLO의 `.pt` 가중치 또는 `.onnx`를
확장자만 바꾼 파일이 아니다. 따라서 현재 안내를 보고 사용자가 가중치 설정을 잘못했다고
판단할 필요가 없다. 먼저 실제 LibreYOLO 어댑터와 완성 팩을 개발·검증해야 한다.

## 완성 팩을 제공받은 경우

1. 제작자로부터 모델별 `.dvmodel`과 서명 확인용 공개키 JSON을 받는다.
2. Windows PowerShell에서 신뢰 저장소를 지정하고 앱을 실행한다. 공개키 파일의 경로는 절대 경로다.

   ```powershell
   $env:DEEPVISION_MODEL_PACK_TRUST_STORE = 'C:\DeepVision\trusted-model-keys.json'
   python gui\main.py
   ```

3. Qt 학습 화면의 **모델 팩 가져오기 (.dvmodel)**에서 파일을 선택한다.
   해당 태스크의 팩이면 프로젝트 모델로 선택된다. CLI 설치도 가능하다.

   ```powershell
   python tools\install_model_pack.py C:\Models\libreyolo.dvmodel --trust-store C:\DeepVision\trusted-model-keys.json
   ```

4. 컨테이너 실행 환경을 준비하고 **팩 학습 / 팩 추론 / 팩 ONNX export**를 실행한다.
   `.dvmodel`을 가져오는 작업은 Docker/WSL을 설치하지 않는다. Windows 설치본에 컨테이너 런타임을
   묶는 배포 설계와 현재 실제 준비된 설치 환경을 구분해야 한다.

GUI는 서명된 팩만 설치한다. `--allow-unsigned`는 개발 중 직접 만든 팩을 시험하는 CLI 옵션이며
일반 배포 절차로 사용하지 않는다. 신뢰 저장소 형식과 서명 코드는
[`pack_signing.py`](../model_runtime/pack_signing.py)에 있다.

## 팩 개발에 필요한 작업

- [`packaging/model-pack-template`](../packaging/model-pack-template/README.ko.md)을 복사한다.
  LibreYOLO 분류용과 검출용 모델 ID·입출력·클래스 목록을 각각 정의한다.
- Docker 이미지 안에 고정 버전의 LibreYOLO, 모델 정의, 의존성, 가중치를 넣는다.
  팩 실행은 네트워크 없이 동작하므로 최초 실행 중 가중치 다운로드에 의존하면 안 된다.
- `worker.py`의 `prepare`, `train`, `infer`, `export`를 실제로 구현한다. DVW1 프레임으로
  이벤트·결과를 전달하고 `/data` 입력과 `/work` 출력을 사용한다. 취소 처리도 필요하다.
- Studio 데이터 폴더·클래스 순서·Best 지표를 LibreYOLO 학습 입력으로 변환한다.
  ONNX 출력은 Studio의 전처리·후처리 JSON 계약과 맞추고 C++17/C# 예제로 검증한다.
  LibreYOLO가 ONNX를 만들 수 있다는 사실만으로 Studio SDK와 자동 호환되지는 않는다.
- 이미지 digest, 오프라인 이미지 자산, 체크섬, 출처·라이선스 고지와 서명을 넣는다.
  실제 Windows 학습·추론·export·SDK 검증을 끝내기 전에는 출시 완료로 표시하지 않는다.

어댑터 구현과 이미지 준비를 **마친 다음** 개발 팩을 만들고 로컬로 설치하는 명령은 다음과 같다.

```powershell
python tools\build_model_pack.py work\libreyolo-pack work\libreyolo.dvmodel --allow-unsigned
python tools\install_model_pack.py work\libreyolo.dvmodel --allow-unsigned
```

공식 API 참고: [분류 학습](https://www.libreyolo.com/docs/tasks/image-classification),
[YOLOv9](https://www.libreyolo.com/docs/models/yolov9),
[ONNX export](https://www.libreyolo.com/docs/export/onnx).
이 문서는 LibreYOLO 완성 팩을 제공하거나 그 성능·호환성을 검증했다는 뜻이 아니다.
