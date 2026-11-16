param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PythonArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "amd_gpu_env.ps1")
$state = Set-AmdGpuProjectEnv
& $state.PythonExe @PythonArgs
