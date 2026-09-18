[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $PayloadRoot,
    [Parameter(Mandatory = $true)] [string] $Version,
    [Parameter(Mandatory = $true)] [string] $OutputDirectory,
    [string] $WixVersion = "4.0.5",
    [string] $CertificatePath = "",
    [string] $CertificatePassword = "",
    [string] $SignToolPath = "signtool.exe",
    [string] $TimestampUrl = "http://timestamp.digicert.com",
    [switch] $RequireSignature
)

$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") { throw "Windows is required for the WiX build." }
if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$') { throw "Version must be major.minor.patch." }
if ($RequireSignature -and [string]::IsNullOrWhiteSpace($CertificatePath)) {
    throw "-RequireSignature requires -CertificatePath."
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
$wix = Get-Command wix -ErrorAction SilentlyContinue
if (-not $wix) { throw "WiX v4 'wix' command is required on the locked Windows build image." }
New-Item -ItemType Directory -Force -Path $output | Out-Null

$collector = Join-Path $scriptRoot "collect_payloads.py"
$validator = Join-Path $scriptRoot "validate_payloads.py"
python $collector $root --manifest $manifest --version $Version --commit $env:GITHUB_SHA
python $validator $root --manifest $manifest `
    --require "app\DeepVisionStudio.exe" `
    --require "models\default-model-catalog.json" `
    --require "sdk\VisionRuntime.dll" `
    --require "sdk\native\vision_runtime.dll"

$common = @("-arch", "x64", "-dVersion=$Version", "-dPayloadRoot=$root")
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

$bundleArgs = @("build", "-arch", "x64", "-ext", "WixToolset.Bal.wixext", "-dVersion=$Version",
    "-dMsiPath=$msi", "-o", $setup, (Join-Path $scriptRoot "bootstrapper\DeepVisionStudio.bundle.wxs"))
& $wix.Source @bundleArgs
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $setup)) { throw "WiX Burn bundle build failed." }
Sign-AndVerify $setup
Write-Output $setup
