$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskDrive = @('N:', 'O:', 'P:', 'Q:') | Where-Object { -not (Test-Path ($_ + '\')) } | Select-Object -First 1
if (-not $taskDrive) { throw 'No available temporary drive alias' }
$taskPreviousBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONDONTWRITEBYTECODE = '1'
subst $taskDrive $taskRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not create temporary drive alias' }
try {
    & ($taskDrive + '/.venv/Scripts/python.exe') ($taskDrive + '/tools/verify.py')
    $taskExit = $LASTEXITCODE
} finally {
    subst $taskDrive /d
    $env:PYTHONDONTWRITEBYTECODE = $taskPreviousBytecode
}
exit $taskExit
