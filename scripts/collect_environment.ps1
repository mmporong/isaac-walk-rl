[CmdletBinding()]
param(
    [string]$OutputPath,
    [string]$IsaacSimPath = 'E:\IsaacSim\isaac-sim-4.5.0',
    [string]$IsaacLabPath
)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $repoRoot 'reports\environment_manifest.json'
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $repoRoot $OutputPath
}

if ([string]::IsNullOrWhiteSpace($IsaacLabPath)) {
    if (-not [string]::IsNullOrWhiteSpace($env:ISAACLAB_PATH)) {
        $IsaacLabPath = $env:ISAACLAB_PATH
    }
    else {
        $IsaacLabPath = Join-Path $HOME 'IsaacLab'
    }
}

function Get-CommandText {
    param(
        [Parameter(Mandatory)]
        [string]$Command,
        [string[]]$Arguments = @()
    )

    $resolved = Get-Command $Command -ErrorAction SilentlyContinue
    if ($null -eq $resolved) {
        return $null
    }

    $output = & $resolved.Source @Arguments 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    return (($output | Out-String).Trim())
}

function Convert-ToPortablePath {
    param([Parameter(Mandatory)][string]$Path)

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $homePath = [System.IO.Path]::GetFullPath($HOME).TrimEnd('\')
    if ($fullPath.Equals($homePath, [System.StringComparison]::OrdinalIgnoreCase)) {
        return '%USERPROFILE%'
    }
    if ($fullPath.StartsWith($homePath + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        return '%USERPROFILE%' + $fullPath.Substring($homePath.Length)
    }
    return $fullPath
}

$gitVersionRaw = Get-CommandText -Command 'git' -Arguments @('--version')
$gitAvailable = $null -ne $gitVersionRaw
$gitLfsVersionRaw = if ($gitAvailable) {
    Get-CommandText -Command 'git' -Arguments @('lfs', 'version')
}
else {
    $null
}

$osInfo = Get-CimInstance Win32_OperatingSystem
$os = [ordered]@{
    caption = $osInfo.Caption
    version = $osInfo.Version
    build = $osInfo.BuildNumber
    architecture = $osInfo.OSArchitecture
}

$gpuList = @()
$gpuCsv = Get-CommandText -Command 'nvidia-smi' -Arguments @(
    '--query-gpu=name,driver_version,memory.total',
    '--format=csv,noheader,nounits'
)
if (-not [string]::IsNullOrWhiteSpace($gpuCsv)) {
    foreach ($line in ($gpuCsv -split "`r?`n")) {
        $parts = $line.Split(',') | ForEach-Object { $_.Trim() }
        if ($parts.Count -eq 3) {
            $gpuList += [ordered]@{
                name = $parts[0]
                driver_version = $parts[1]
                vram_mib = [int]$parts[2]
            }
        }
    }
}

$isaacSimExists = Test-Path -LiteralPath $IsaacSimPath -PathType Container
$isaacPythonPath = Join-Path $IsaacSimPath 'kit\python\kit.exe'
$isaacPythonExists = Test-Path -LiteralPath $isaacPythonPath -PathType Leaf
$isaacPythonVersion = $null
if ($isaacPythonExists) {
    $pythonOutput = & $isaacPythonPath -c 'import platform; print(platform.python_version())' 2>$null
    if ($LASTEXITCODE -eq 0) {
        $isaacPythonVersion = (($pythonOutput | Out-String).Trim())
    }
}

$gitState = 'not_a_repository'
$gitCommit = $null
if ($gitAvailable) {
    $insideWorkTree = & git -C $repoRoot rev-parse --is-inside-work-tree 2>$null
    if ($LASTEXITCODE -eq 0 -and (($insideWorkTree | Out-String).Trim() -eq 'true')) {
        $head = & git -C $repoRoot rev-parse HEAD 2>$null
        if ($LASTEXITCODE -eq 0) {
            $gitState = 'commit'
            $gitCommit = (($head | Out-String).Trim())
        }
        else {
            $gitState = 'unborn'
        }
    }
}

$manifest = [ordered]@{
    schema_version = 1
    platform = [ordered]@{
        os = $os
        powershell = [ordered]@{
            edition = $PSVersionTable.PSEdition
            version = $PSVersionTable.PSVersion.ToString()
        }
    }
    tools = [ordered]@{
        git = [ordered]@{
            available = $gitAvailable
            version = $gitVersionRaw
        }
        git_lfs = [ordered]@{
            available = ($null -ne $gitLfsVersionRaw)
            version = $gitLfsVersionRaw
        }
    }
    gpu = [ordered]@{
        nvidia_smi_available = ($null -ne (Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue))
        devices = $gpuList
    }
    isaac_sim = [ordered]@{
        path = Convert-ToPortablePath -Path $IsaacSimPath
        exists = $isaacSimExists
        bundled_python = [ordered]@{
            path = Convert-ToPortablePath -Path $isaacPythonPath
            exists = $isaacPythonExists
            version = $isaacPythonVersion
        }
    }
    isaac_lab = [ordered]@{
        path = Convert-ToPortablePath -Path $IsaacLabPath
        exists = (Test-Path -LiteralPath $IsaacLabPath -PathType Container)
    }
    repository = [ordered]@{
        state = $gitState
        commit = $gitCommit
    }
}

$outputDirectory = Split-Path -Parent $OutputPath
[System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
$json = $manifest | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText(
    [System.IO.Path]::GetFullPath($OutputPath),
    $json + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host "환경 매니페스트 생성: $([System.IO.Path]::GetFullPath($OutputPath))"
