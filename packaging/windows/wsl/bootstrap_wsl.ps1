[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $PayloadRoot,
    [Parameter(Mandatory = $true)] [string] $DistroName,
    [string] $StateRoot = "",
    [string] $Version = "dev",
    [switch] $InstallWslPackage
)

$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") { throw "Windows is required for WSL bootstrap." }
if ($DistroName -notmatch '^[A-Za-z][A-Za-z0-9_.-]{1,63}$') {
    throw "DistroName must be an app-owned WSL identifier."
}

$payload = (Resolve-Path -LiteralPath $PayloadRoot -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $payload -PathType Container)) {
    throw "PayloadRoot must be a directory."
}
$wsl = Join-Path $payload "wsl-offline.msi"
$distro = Join-Path $payload "owned-distro.tar"
foreach ($file in @($wsl, $distro)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required WSL payload file is missing: $file"
    }
}

if ([string]::IsNullOrWhiteSpace($StateRoot)) {
    if (-not [string]::IsNullOrWhiteSpace($env:DEEP_STUDIO_STATE_DIR)) {
        $StateRoot = Join-Path $env:DEEP_STUDIO_STATE_DIR "wsl"
    } elseif (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $StateRoot = Join-Path $env:LOCALAPPDATA "DeepVisionStudio\wsl"
    } else {
        throw "A per-user WSL state directory is required."
    }
}
$state = [IO.Path]::GetFullPath($StateRoot)
New-Item -ItemType Directory -Force -Path $state | Out-Null
$marker = Join-Path $state "owned-distro.json"
$installDir = Join-Path $state "distro"

$wslCommand = Get-Command "wsl.exe" -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) { throw "wsl.exe is not available on this Windows host." }

if ($InstallWslPackage) {
    $result = Start-Process -FilePath "msiexec.exe" -ArgumentList @(
        "/i", $wsl, "/qn", "/norestart"
    ) -Wait -PassThru
    if ($result.ExitCode -notin @(0, 3010)) {
        throw "Offline WSL package installation failed: $($result.ExitCode)"
    }
}

& $wslCommand.Source "--status" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "WSL status probe failed; enable WSL through the bundled installer first." }
$distros = @(& $wslCommand.Source "--list" "--quiet") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$existing = $distros -contains $DistroName
if ($existing) {
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
        throw "An existing WSL distro has no Deep Vision Studio ownership marker: $DistroName"
    }
    $owned = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
    if ($owned.schema_version -ne 1 -or $owned.owned -ne $true -or $owned.distro -ne $DistroName) {
        throw "The existing WSL distro is not owned by this application: $DistroName"
    }
} else {
    if (Test-Path -LiteralPath $installDir) {
        $entries = @(Get-ChildItem -LiteralPath $installDir -Force)
        if ($entries.Count -gt 0) { throw "WSL install directory is not empty: $installDir" }
    } else {
        New-Item -ItemType Directory -Path $installDir | Out-Null
    }
    & $wslCommand.Source "--import" $DistroName $installDir $distro "--version" "2"
    if ($LASTEXITCODE -ne 0) { throw "Owned WSL distro import failed: $DistroName" }
}

# Verify the Engine server through the owned distro.  No host Docker socket,
# TCP API, shell command, pull, or web download is used here.
& $wslCommand.Source "-d" $DistroName "--" "docker" "info" "--format" "{{.ServerVersion}}" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker Engine is not ready inside owned WSL distro: $DistroName" }

$record = [ordered]@{
    schema_version = 1
    owned = $true
    distro = $DistroName
    version = $Version
    engine = "docker"
}
$temporary = "$marker.pending"
$record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporary -Encoding UTF8
Move-Item -LiteralPath $temporary -Destination $marker -Force
Write-Output $marker
