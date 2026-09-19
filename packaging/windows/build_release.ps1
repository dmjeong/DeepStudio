[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $PayloadRoot,
    [Parameter(Mandatory = $true)] [string] $Version,
    [Parameter(Mandatory = $true)] [string] $OutputDirectory,
    [string] $WixVersion = "7.0.0",
    [string] $CertificatePath = "",
    [string] $CertificatePassword = "",
    [string] $SignToolPath = "signtool.exe",
    [string] $TimestampUrl = "http://timestamp.digicert.com",
    [string] $ModelPackTrustStore = "",
    [switch] $RequireSignature,
    [switch] $RequireOfflineWsl,
    [switch] $RequireReleaseReadyModels
)

$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") { throw "Windows is required for the WiX build." }
if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$') { throw "Version must be major.minor.patch." }
if ($WixVersion -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$') { throw "WixVersion must be major.minor.patch." }
if ($RequireSignature -and [string]::IsNullOrWhiteSpace($CertificatePath)) {
    throw "-RequireSignature requires -CertificatePath."
}
if ($RequireReleaseReadyModels -and [string]::IsNullOrWhiteSpace($ModelPackTrustStore)) {
    throw "-RequireReleaseReadyModels requires -ModelPackTrustStore."
}
if (-not [string]::IsNullOrWhiteSpace($ModelPackTrustStore)) {
    $modelTrustStore = (Resolve-Path $ModelPackTrustStore -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath $modelTrustStore -PathType Leaf)) {
        throw "Model pack trust store does not exist: $ModelPackTrustStore"
    }
}
if (-not [string]::IsNullOrWhiteSpace($CertificatePath)) {
    $certificate = (Resolve-Path $CertificatePath -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath $certificate -PathType Leaf)) {
        throw "Signing certificate does not exist: $CertificatePath"
    }
    $signTool = Get-Command $SignToolPath -ErrorAction SilentlyContinue
    if (-not $signTool) { throw "signtool is required when signing the release." }
}
$root = (Resolve-Path $PayloadRoot).Path
$output = [IO.Path]::GetFullPath($OutputDirectory)
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$notice = Join-Path $root "THIRD_PARTY_NOTICES.md"
if (-not (Test-Path -LiteralPath $notice -PathType Leaf)) {
    throw "Payload must include THIRD_PARTY_NOTICES.md before signing."
}
$manifest = Join-Path $output "release-manifest.json"
$msi = Join-Path $output "DeepVisionStudio-$Version.msi"
$setup = Join-Path $output "DeepVisionStudio-Setup-$Version-win-x64.exe"
$releaseSucceeded = $false
trap {
    $failure = $_
    if (-not $releaseSucceeded) {
        # WiX and signtool can leave a partial MSI/EXE behind.  Never let a
        # failed build be mistaken for a usable installer artifact.
        foreach ($artifact in @($msi, $setup)) {
            if (Test-Path -LiteralPath $artifact -PathType Leaf) {
                Remove-Item -LiteralPath $artifact -Force -ErrorAction SilentlyContinue
            }
        }
    }
    throw $failure
}
$wix = Get-Command wix -ErrorAction SilentlyContinue
if (-not $wix) { throw "WiX 7 'wix' command is required on the locked Windows build image." }
$wixVersionOutput = & $wix.Source --version 2>&1
$wixExitCode = $LASTEXITCODE
$wixVersionText = ($wixVersionOutput -join "`n").Trim()
$expectedWixVersion = [regex]::Escape($WixVersion)
if ($wixExitCode -ne 0 -or $wixVersionText -notmatch "(?<!\d)$expectedWixVersion(?!\d)") {
    throw "WiX version mismatch. Expected $WixVersion, detected: $wixVersionText"
}
New-Item -ItemType Directory -Force -Path $output | Out-Null

$collector = Join-Path $scriptRoot "collect_payloads.py"
$validator = Join-Path $scriptRoot "validate_payloads.py"
python $collector $root --manifest $manifest --version $Version --commit $env:GITHUB_SHA
if ($LASTEXITCODE -ne 0) { throw "Payload manifest collection failed." }
python $validator $root --manifest $manifest `
    --require "app\DeepVisionStudio.exe" `
    --require "models\default-model-catalog.json" `
    --require "sdk\VisionRuntime.dll" `
    --require "sdk\native\vision_runtime.dll"
if ($LASTEXITCODE -ne 0) { throw "Payload contract validation failed." }
if ($RequireReleaseReadyModels) {
    python $validator $root --manifest $manifest `
        --require "models\default-model-catalog.json" `
        --require-release-ready-models `
        --model-pack-trust-store $modelTrustStore
    if ($LASTEXITCODE -ne 0) { throw "Release-ready model payload contract failed." }
}
if ($RequireOfflineWsl) {
    # The distro tar contains the pinned Docker Engine/Moby userspace.  Keep
    # the host WSL installer and distro as separate payload files so their
    # licenses, hashes, and replacement cadence remain auditable.
    python $validator $root --manifest $manifest --require-offline-wsl `
        --require "runtime\wsl\wsl-offline.msi" `
        --require "runtime\wsl\owned-distro.tar" `
        --require "runtime\wsl\licenses\manifest.json" `
        --require "runtime\wsl\bootstrap_wsl.ps1"
    if ($LASTEXITCODE -ne 0) { throw "Offline WSL payload contract failed." }
}

$requireWslValue = if ($RequireOfflineWsl) { "1" } else { "0" }
$common = @("-arch", "x64", "-ext", "WixToolset.Util.wixext",
    "-d", "Version=$Version", "-d", "PayloadRoot=$root",
    "-d", "RequireOfflineWsl=$requireWslValue")
$msiArgs = @("build") + $common + @("-o", $msi, (Join-Path $scriptRoot "bootstrapper\DeepVisionStudio.msi.wxs"))
& $wix.Source @msiArgs
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $msi)) { throw "WiX MSI build failed." }

function Sign-AndVerify([string] $Path) {
    if ([string]::IsNullOrWhiteSpace($CertificatePath)) { return }
    $arguments = @("sign", "/fd", "SHA256", "/f", $certificate)
    if (-not [string]::IsNullOrWhiteSpace($CertificatePassword)) {
        $arguments += @("/p", $CertificatePassword)
    }
    if (-not [string]::IsNullOrWhiteSpace($TimestampUrl)) {
        $arguments += @("/tr", $TimestampUrl, "/td", "SHA256")
    }
    $arguments += $Path
    & $signTool.Source @arguments
    if ($LASTEXITCODE -ne 0) { throw "Authenticode signing failed: $Path" }
    & $signTool.Source verify /pa /all $Path
    if ($LASTEXITCODE -ne 0) { throw "Authenticode verification failed: $Path" }
}

Sign-AndVerify $msi

$bundleArgs = @("build", "-arch", "x64", "-ext", "WixToolset.Bal.wixext",
    "-d", "Version=$Version", "-d", "RequireOfflineWsl=$requireWslValue",
    "-d", "MsiPath=$msi")
if ($RequireOfflineWsl) {
    $bundleArgs += @("-d", "WslMsiPath=$(Join-Path $root 'runtime\wsl\wsl-offline.msi')")
}
$bundleArgs += @("-o", $setup, (Join-Path $scriptRoot "bootstrapper\DeepVisionStudio.bundle.wxs"))
& $wix.Source @bundleArgs
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $setup)) { throw "WiX Burn bundle build failed." }
Sign-AndVerify $setup
$releaseSucceeded = $true
Write-Output $setup
