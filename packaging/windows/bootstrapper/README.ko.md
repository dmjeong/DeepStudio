# WiX 7 오프라인 Setup 빌드

`build_release.ps1`는 이미 수집·해시 검증된 payload를 MSI에 넣고, 그 MSI 하나를
Burn bundle EXE에 첨부한다. 실행 중에 WiX extension, Python, 모델, WSL 또는 Docker를
다운로드하지 않는다. WiX 7과 `WixToolset.Bal.wixext`는 Windows 빌드 이미지에
미리 설치하고 버전을 잠근다.

```powershell
pwsh packaging/windows/build_release.ps1 `
  -PayloadRoot .\release-payload `
  -Version 1.0.0 `
  -OutputDirectory .\release-out `
  -RequireOfflineWsl
```

`-RequireOfflineWsl`은 production Setup에서 반드시 사용한다. payload에는 다음 파일이
필요하다.

- `runtime/wsl/wsl-offline.msi`: Windows WSL 오프라인 설치 패키지
- `runtime/wsl/owned-distro.tar`: Docker Engine/Moby가 들어 있는 앱 전용 distro
- `runtime/wsl/licenses/manifest.json`: WSL·Docker·distro 구성요소의 버전, SHA-256, 라이선스/고지 목록

이 파일들은 인터넷에서 Setup 실행 중 내려받지 않는다. 각 공급자의 재배포 허가와 고지는
`THIRD_PARTY_NOTICES.md`에도 포함해야 한다. 세 파일이 없는 unsigned 개발 계약 빌드는
`-RequireOfflineWsl`을 생략할 수 있지만, 그 결과는 Docker 확장 포함 배포본으로 표시하지 않는다.
GitHub Actions의 production 단계는 `DEEPVISION_WSL_PAYLOAD_ROOT`가 가리키는 사전 검증된
로컬 payload를 `runtime/wsl`로 복사한다. 이 경로를 제공하지 않으면 workflow가 Setup을 만들지 않는다.
production Burn은 WSL MSI를 먼저 설치하고 제거 때 보존한 뒤 앱 MSI를 설치한다. 앱 MSI는
실제 사용자 세션에서 `bootstrap_wsl.ps1`을 실행해 owned distro를 import하고 Docker `info`
smoke가 성공한 뒤에만 설치를 완료한다. 새 import가 smoke 또는 marker 기록에서 실패하면
bootstrap이 그 import만 unregister하고 임시 상태를 지워 다음 설치 시 재시도할 수 있다.

서명 릴리스는 인증서와 `signtool.exe`가 준비된 Windows 빌드에서 다음처럼 실행한다.
`-RequireSignature`를 사용하면 인증서가 없거나 MSI/Burn EXE의 Authenticode 검증이 실패할 때
즉시 중단한다. 인증서 옵션을 생략한 개발용 실행은 서명하지 않은 계약 빌드다.

```powershell
pwsh packaging/windows/build_release.ps1 `
  -PayloadRoot .\release-payload `
  -Version 1.0.0 `
  -OutputDirectory .\release-out `
  -CertificatePath .\release-signing.pfx `
  -CertificatePassword $env:DEEP_VISION_SIGNING_PASSWORD `
  -RequireOfflineWsl `
  -RequireSignature
```

결과는 `DeepVisionStudio-Setup-1.0.0-win-x64.exe`와 payload manifest다. 스크립트는
먼저 symlink·파일 hash·총 용량·3.5 GiB 예산을 검사하고, MSI와 Burn 빌드가 실패하면
출력 파일을 남기지 않는다. 서명 옵션을 생략한 결과는 계약용 unsigned 산출물이다.
Authenticode 서명과 Windows VM 오프라인 설치/제거 검사는 서명 인증서와 Windows runner에서
실행해야 하며, macOS 개발 환경에서 성공으로 표시하지 않는다.
