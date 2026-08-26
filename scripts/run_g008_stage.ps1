[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('command', 'friction', 'irregular_road', 'leg_mass')]
    [string]$Part,

    [ValidateSet('baseline', 'turn_air_time')]
    [string]$RewardVariant = 'baseline',

    [ValidateRange(0, 3)]
    [int]$Stage = 0,

    [ValidateRange(1, 65536)]
    [int]$NumEnvs = 64,

    [ValidateRange(1, 1000000)]
    [int]$MaxIterations = 1,

    [int]$Seed = 42,

    [string]$ResumeRun,

    [ValidatePattern('^$|^model_[0-9]+\.pt$')]
    [string]$ResumeCheckpoint,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9][a-z0-9_-]+$')]
    [string]$RunName
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($Part -eq 'command') {
    if ($Stage -ne 0) {
        throw 'command 파트는 Stage 0만 사용합니다.'
    }
    $task = 'Isaac-G008-Velocity-Rough-Go2-CommandSuite-v0'
}
elseif ($Part -eq 'irregular_road') {
    if ($Stage -eq 0) {
        if ($RewardVariant -eq 'turn_air_time') {
            $task = 'Isaac-G008-Velocity-IrregularRoad-Go2-G0-TurnAir-v0'
        }
        else {
            $task = 'Isaac-G008-Velocity-IrregularRoad-Go2-G0-v0'
        }
    }
    elseif ($Stage -eq 1) {
        $task = 'Isaac-G008-Velocity-IrregularRoad-Go2-S1-v0'
    }
    else {
        throw 'irregular_road 파트는 Stage 0~1을 사용합니다.'
    }
}
else {
    if ($Stage -lt 1) {
        throw "$Part 파트는 Stage 1~3을 사용합니다."
    }
    $suffix = if ($Part -eq 'friction') { 'Friction' } else { 'LegMass' }
    $task = "Isaac-G008-Velocity-Rough-Go2-$suffix-S$Stage-v0"
}

if ($RewardVariant -ne 'baseline' -and -not ($Part -eq 'irregular_road' -and $Stage -eq 0)) {
    throw 'RewardVariant는 irregular_road Stage 0에서만 사용합니다.'
}

$trainingEntrypoint = Join-Path $PSScriptRoot 'bootstrap_train_g008.py'
$reportPath = Join-Path (Split-Path -Parent $PSScriptRoot) "reports\runs\$RunName.json"

$trainingArguments = @{
    Task = $task
    NumEnvs = $NumEnvs
    MaxIterations = $MaxIterations
    Seed = $Seed
    RunName = $RunName
    ReportPath = $reportPath
    TrainingEntrypointPath = $trainingEntrypoint
}
if (-not [string]::IsNullOrWhiteSpace($ResumeRun)) {
    if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
        throw 'ResumeRun을 사용할 때 ResumeCheckpoint가 필요합니다.'
    }
    $trainingArguments.Resume = $true
    $trainingArguments.LoadRun = $ResumeRun
    $trainingArguments.ResumeCheckpoint = $ResumeCheckpoint
}
elseif (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    throw 'ResumeCheckpoint는 ResumeRun과 함께 사용해야 합니다.'
}

& (Join-Path $PSScriptRoot 'run_training.ps1') @trainingArguments

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
