# WiX v4 오프라인 Setup 빌드

`build_release.ps1`는 이미 수집·해시 검증된 payload를 MSI에 넣고, 그 MSI 하나를
Burn bundle EXE에 첨부한다. 실행 중에 WiX extension, Python, 모델, WSL 또는 Docker를
다운로드하지 않는다. WiX v4와 `WixToolset.Bal.wixext`는 Windows 빌드 이미지에
미리 설치하고 버전을 잠근다.

```powershell
pwsh packaging/windows/build_release.ps1 `
  -PayloadRoot .\release-payload `
  -Version 1.0.0 `
  -OutputDirectory .\release-out
```

결과는 `DeepVisionStudio-Setup-1.0.0-win-x64.exe`와 payload manifest다. 스크립트는
먼저 symlink·파일 hash·총 용량·3.5 GiB 예산을 검사하고, MSI와 Burn 빌드가 실패하면
출력 파일을 남기지 않는다. Authenticode 서명과 Windows VM 오프라인 설치/제거 검사는
빌드 후 단계이며, macOS 개발 환경에서 성공으로 표시하지 않는다.

