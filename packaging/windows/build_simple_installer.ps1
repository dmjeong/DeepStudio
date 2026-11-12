[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $AppRoot,
    [Parameter(Mandatory = $true)] [string] $OutputDirectory,
    [string] $Version = "",
    [string] $WixVersion = "7.0.0",
    [string] $CertificatePath = "",
    [string] $CertificatePassword = "",
    [string] $SignToolPath = "signtool.exe",
    [string] $TimestampUrl = "http://timestamp.digicert.com",
    [switch] $RequireSignature
)

$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") { throw "Windows is required for the WiX build." }
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$app = (Resolve-Path $AppRoot -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath (Join-Path $app "DeepVisionStudio.exe") -PathType Leaf)) {
    throw "AppRoot must contain DeepVisionStudio.exe: $app"
}

if ([string]::IsNullOrWhiteSpace($Version)) {
    $versionFile = Join-Path $repoRoot "gui\core\version.py"
    $rawVersion = (& python -c "import runpy,sys; print(runpy.run_path(sys.argv[1])['APP_VERSION'])" $versionFile).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Could not read gui/core/version.py." }
    $Version = $rawVersion
}
if ($Version -match '^[0-9]+\.[0-9]+$') { $Version = "$Version.0" }
if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$') { throw "Version must be major.minor or major.minor.patch." }

$output = [IO.Path]::GetFullPath($OutputDirectory)
$payload = Join-Path $output ".simple-payload-$Version"
if (Test-Path -LiteralPath $payload) { Remove-Item -LiteralPath $payload -Force -Recurse }
New-Item -ItemType Directory -Force -Path $output | Out-Null

$stager = Join-Path $scriptRoot "stage_simple_payload.py"
& python $stager --output $payload --app $app `
    --example-root (Join-Path $repoRoot "example") `
    --cpp-runtime-root (Join-Path $repoRoot "cpp") `
    --csharp-runtime-root (Join-Path $repoRoot "sdk\csharp") `
    --notice (Join-Path $repoRoot "THIRD_PARTY_NOTICES.md")
if ($LASTEXITCODE -ne 0) { throw "Minimal installer payload staging failed." }

$release = Join-Path $scriptRoot "build_release.ps1"
# Array splatting into a PowerShell script passes these strings positionally.
# Use named parameters so paths are not mistaken for parameter names/versions.
$releaseArgs = @{
    PayloadRoot = $payload
    Version = $Version
    OutputDirectory = $output
    WixVersion = $WixVersion
    Simple = $true
}
if (-not [string]::IsNullOrWhiteSpace($CertificatePath)) {
    $releaseArgs.CertificatePath = $CertificatePath
    $releaseArgs.CertificatePassword = $CertificatePassword
    $releaseArgs.SignToolPath = $SignToolPath
    $releaseArgs.TimestampUrl = $TimestampUrl
}
if ($RequireSignature) { $releaseArgs.RequireSignature = $true }
& $release @releaseArgs
if ($LASTEXITCODE -ne 0) { throw "Minimal installer build failed." }
