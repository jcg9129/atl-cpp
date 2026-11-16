# CI only. Sets up the MSVC developer environment for the following steps of a GitHub Actions job: finds Visual
# Studio with vswhere, runs vcvarsall.bat for the requested target and toolset, and exports every variable it
# changes to GITHUB_ENV. Replaces a third-party action, so the job runs no JavaScript action at all.
# Usage: ci-msvc-env.ps1 -Arch x64|x86|arm64 -Toolset 14.44
param(
    [Parameter(Mandatory)][ValidateSet('x64', 'x86', 'arm64')][string]$Arch,
    [Parameter(Mandatory)][string]$Toolset
)
$ErrorActionPreference = 'Stop'

# vcvarsall target names: x86 is cross-compiled by the x64-hosted tools; x64 and arm64 run on native hosts.
$target = @{ x64 = 'x64'; x86 = 'x64_x86'; arm64 = 'arm64' }[$Arch]

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products * -property installationPath
if (-not $installation) { throw 'vswhere found no Visual Studio installation' }
$vcvarsall = Join-Path $installation 'VC\Auxiliary\Build\vcvarsall.bat'
if (-not (Test-Path $vcvarsall)) { throw "vcvarsall.bat not found under $installation" }

$before = @{}
Get-ChildItem env: | ForEach-Object { $before[$_.Name] = $_.Value }
$after = & cmd.exe /c "`"$vcvarsall`" $target -vcvars_ver=$Toolset >nul 2>&1 && set"
if ($LASTEXITCODE -ne 0) { throw "vcvarsall.bat $target -vcvars_ver=$Toolset failed ($LASTEXITCODE)" }

$changed = @{}
foreach ($line in $after) {
    if ($line -match '^([^=]+)=(.*)$' -and $before[$Matches[1]] -cne $Matches[2]) { $changed[$Matches[1]] = $Matches[2] }
}
# vcvarsall reports some failures only on its console, so check that the requested toolset is what it set up.
if (-not $changed['VCToolsVersion'] -or -not $changed['VCToolsVersion'].StartsWith($Toolset)) {
    throw "vcvarsall did not select toolset $Toolset (VCToolsVersion='$($changed['VCToolsVersion'])')"
}
foreach ($name in $changed.Keys) { "$name=$($changed[$name])" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8 }
Write-Host "MSVC $($changed['VCToolsVersion']) for $target ($($changed.Count) variables exported)"
