param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $HydraArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "amd_gpu_env.ps1")
$state = Set-AmdGpuProjectEnv

$defaults = @(
    (Join-Path $state.RepoRoot "tools\train_species_aware_hybrid.py"),
    "data=animal_breed_v0",
    "data.species_aware.hybrid.train.device=0"
)

Push-Location $state.RepoRoot
try {
    & $state.PythonExe @defaults @HydraArgs
}
finally {
    Pop-Location
}
