# 오프라인 WSL payload 준비

이 폴더에는 WSL·Docker 바이너리를 저장하지 않는다. 배포 담당자는 재배포가 허용된
Windows x64 payload를 별도 보안 저장소에서 받아 `DEEPVISION_WSL_PAYLOAD_ROOT`로 지정한다.

소스 디렉터리는 다음 구조를 갖춰야 한다.

```text
wsl-offline.msi
owned-distro.tar
licenses/
  manifest.json
  wsl.txt
  docker.txt
  distro.txt
bootstrap_wsl.ps1
```

`owned-distro.tar`에는 앱 전용 Docker Engine/Moby userspace가 들어 있어야 한다. `manifest.json`은
다음 계약을 사용한다. 각 `artifact`와 `notice`는 payload 루트 기준 상대 경로이고, `sha256`은
실제 artifact의 SHA-256이다.

```json
{
  "schema_version": 1,
  "platform": "windows-x64",
  "components": [
    {"id": "wsl", "artifact": "runtime/wsl/wsl-offline.msi", "sha256": "<64 hex>", "license": "<license>", "notice": "runtime/wsl/licenses/wsl.txt"},
    {"id": "owned_distro", "artifact": "runtime/wsl/owned-distro.tar", "sha256": "<64 hex>", "license": "<license>", "notice": "runtime/wsl/licenses/distro.txt"},
    {"id": "docker_engine", "artifact": "runtime/wsl/owned-distro.tar", "sha256": "<64 hex>", "license": "<license>", "notice": "runtime/wsl/licenses/docker.txt"}
  ]
}
```

Setup 빌드 전에 공급자의 재배포 조건, 고지, 취약점 공지를 검토하고 내용을
`THIRD_PARTY_NOTICES.md`에 합친다. 저장소에는 해당 binary나 checkpoint를 커밋하지 않는다.
설치 단계는 `bootstrap_wsl.ps1 -PayloadRoot <runtime/wsl> -DistroName <app-owned-name>`을
실제 사용자 세션에서 실행한다. 기존 distro가 있으면 marker가 없거나 소유자가 다를 때 중단하고,
새 distro는 로컬 tar만 import한 뒤 `docker info`가 성공할 때 `owned-distro.json`을 원자적으로 기록한다.
