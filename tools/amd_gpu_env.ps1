Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-AmdGpuPythonRoot {
    $candidates = @()
    if ($env:ANIMAL_BREED_GPU_PYTHON_ROOT) {
        $candidates += $env:ANIMAL_BREED_GPU_PYTHON_ROOT
    }
    $candidates += "C:\amd_rocm_py312_real"
    $candidates += (Join-Path (Get-RepoRoot) ".gpu_amd_setup\python312_embed")
    foreach ($candidate in $candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path (Join-Path $candidate "python.exe"))) {
            return (Resolve-Path $candidate).Path
        }
    }
    throw "AMD GPU Python runtime not found. Checked: $($candidates -join ', ')"
}

function Set-AmdGpuProjectEnv {
    $repoRoot = Get-RepoRoot
    $pythonRoot = Get-AmdGpuPythonRoot
    $rocmRoot = Join-Path $pythonRoot "Lib\site-packages\_rocm_sdk_core"
    $repoFallbackRoot = Join-Path $repoRoot ".gpu_amd_setup\python312_embed"
    $pathParts = @(
        # Prefer the real ROCm LLVM binaries over Python entrypoint wrappers.
        # The generated wrapper in Scripts\offload-arch.exe may embed an old
        # install path with spaces, which breaks subprocess launches.
        (Join-Path $rocmRoot "lib\llvm\bin"),
        (Join-Path $rocmRoot "bin"),
        (Join-Path $pythonRoot "Scripts"),
        $pythonRoot
    ) | Where-Object { Test-Path $_ }

    $existing = @()
    if ($env:PATH) {
        $existing = $env:PATH -split ';' | Where-Object {
            -not [string]::IsNullOrWhiteSpace($_) -and
            $_ -notlike "$repoFallbackRoot*" -and
            $_ -notlike "*\.gpu_amd_setup\python312_embed*"
        }
    }
    $env:PATH = (($pathParts + $existing) | Select-Object -Unique) -join ';'

    $env:ANIMAL_BREED_GPU_REPO = $repoRoot
    $env:ANIMAL_BREED_GPU_PYTHON_ROOT = $pythonRoot
    $env:ANIMAL_BREED_GPU_PYTHON = Join-Path $pythonRoot "python.exe"
    $env:ROCM_PATH = $rocmRoot
    $env:HIP_PATH = $rocmRoot
    # The Windows ROCm Python packages attempt to rediscover the target family
    # by prepending sysconfig.get_path("scripts") and invoking offload-arch.
    # In this runtime, Scripts\offload-arch.exe is a stale wrapper that embeds
    # an old path with spaces. Pin the family explicitly so imports do not
    # re-enter that broken discovery path.
    $env:ROCM_SDK_TARGET_FAMILY = "custom"
    $env:PYTHONNOUSERSITE = "1"

    return @{
        RepoRoot = $repoRoot
        PythonRoot = $pythonRoot
        PythonExe = $env:ANIMAL_BREED_GPU_PYTHON
        RocmRoot = $rocmRoot
    }
}
