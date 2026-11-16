Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "amd_gpu_env.ps1")
$state = Set-AmdGpuProjectEnv

Write-Host "AMD GPU shell ready"
Write-Host "  repo:   $($state.RepoRoot)"
Write-Host "  python: $($state.PythonExe)"
Write-Host "  rocm:   $($state.RocmRoot)"
Write-Host ""
Write-Host "Examples:"
Write-Host "  animal-gpu-probe"
Write-Host "  animal-train-species-aware"
Write-Host "  python tools\\train_species_aware_hybrid.py data=animal_breed_v0 data.species_aware.hybrid.train.enabled=true"
Write-Host ""

Push-Location $state.RepoRoot
try {
    powershell -NoExit -NoLogo
}
finally {
    Pop-Location
}
