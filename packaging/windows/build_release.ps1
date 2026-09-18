[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $PayloadRoot,
    [Parameter(Mandatory = $true)] [string] $Version,
    [Parameter(Mandatory = $true)] [string] $OutputDirectory,
    [string] $WixVersion = "4.0.5"
)

$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") { throw "Windows is required for the WiX build." }
if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$') { throw "Version must be major.minor.patch." }
$root = (Resolve-Path $PayloadRoot).Path
$output = [IO.Path]::GetFullPath($OutputDirectory)
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifest = Join-Path $output "release-manifest.json"
$msi = Join-Path $output "DeepVisionStudio-$Version.msi"
$setup = Join-Path $output "DeepVisionStudio-Setup-$Version-win-x64.exe"
$wix = Get-Command wix -ErrorAction SilentlyContinue
if (-not $wix) { throw "WiX v4 'wix' command is required on the locked Windows build image." }
New-Item -ItemType Directory -Force -Path $output | Out-Null

$payloadTool = Join-Path $scriptRoot "payload_manifest.py"
python $payloadTool $root --manifest $manifest --version $Version --commit $env:GITHUB_SHA
python $payloadTool $root --manifest $manifest --verify

$common = @("-arch", "x64", "-dVersion=$Version", "-dPayloadRoot=$root")
$msiArgs = @("build") + $common + @("-o", $msi, (Join-Path $scriptRoot "bootstrapper\DeepVisionStudio.msi.wxs"))
& $wix.Source @msiArgs
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $msi)) { throw "WiX MSI build failed." }

$bundleArgs = @("build", "-arch", "x64", "-ext", "WixToolset.Bal.wixext", "-dVersion=$Version",
    "-dMsiPath=$msi", "-o", $setup, (Join-Path $scriptRoot "bootstrapper\DeepVisionStudio.bundle.wxs"))
& $wix.Source @bundleArgs
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $setup)) { throw "WiX Burn bundle build failed." }
Write-Output $setup

