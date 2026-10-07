param([string]$Model = 'qwen3.5:9b', [string]$Out = 'runs/pilot')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python -m expel doctor --model $Model --out $Out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m expel pipeline --model $Model --out $Out
exit $LASTEXITCODE
