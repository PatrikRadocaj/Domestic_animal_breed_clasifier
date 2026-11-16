param(
    [int] $Stage1Epochs = 10,
    [int] $Stage2Epochs = 5,
    [int] $DirectYoloEpochs = 20,
    [int] $BreedBuildBatchSize = 32,
    [int] $DirectYoloBatch = 4,
    [int] $DirectYoloWorkers = 4,
    [string] $RunStamp = "",
    [string] $ResumeSpeciesBest = "",
    [switch] $SkipMerge,
    [switch] $SkipStage1,
    [switch] $SkipBreedBuild,
    [switch] $SkipFinalize,
    [switch] $SkipPrepareDirect
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "amd_gpu_env.ps1")
$state = Set-AmdGpuProjectEnv
$repoRoot = $state.RepoRoot
$runStamp = if ([string]::IsNullOrWhiteSpace($RunStamp)) { Get-Date -Format "yyyyMMdd_HHmmss" } else { $RunStamp }
$reportRoot = Join-Path $repoRoot "results\reports\strict_pipeline\$runStamp"
$hybridProject = Join-Path $repoRoot "runs\detect\artifacts\hybrid_shared_detector"
$speciesRunName = "hybrid_shared_strict_$runStamp"
$breedRunName = "temporary_breed_run_strict_$runStamp"
$breedOutDir = Join-Path $repoRoot "data\processed\breed_runs\$breedRunName"
$stage2ManifestCsv = Join-Path $breedOutDir "manifest.trainable.csv"
$stage2ManifestParquet = Join-Path $breedOutDir "manifest.trainable.parquet"
$stage2TaxonomyYaml = Join-Path $breedOutDir "taxonomy.trainable.yaml"
$directManifestCsv = Join-Path $breedOutDir "manifest.direct_yolo.csv"
$directManifestParquet = Join-Path $breedOutDir "manifest.direct_yolo.parquet"
$directTaxonomyYaml = Join-Path $breedOutDir "taxonomy.direct_yolo.yaml"
$directPreparedDir = Join-Path $repoRoot "data\processed\yolo_breed_${breedRunName}_direct_rgb"
$directYaml = Join-Path $repoRoot "data\processed\yolo_breed_${breedRunName}_direct_rgb.yaml"
$reviewCsv = Join-Path $breedOutDir "label_review_report.csv"
$reviewJson = Join-Path $breedOutDir "label_review_summary.json"

$pipelineLog = Join-Path $reportRoot "pipeline.log"
$mergeLog = Join-Path $reportRoot "01_merge_phr.log"
$stage1Log = Join-Path $reportRoot "02_stage1_species.log"
$breedBuildLog = Join-Path $reportRoot "03_breed_build.log"
$reviewLog = Join-Path $reportRoot "04_label_review.log"
$stage2Log = Join-Path $reportRoot "05_stage2_efficientnet.log"
$directLog = Join-Path $reportRoot "06_direct_breed_yolo.log"

function Write-Status {
    param([string] $Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $pipelineLog -Append
}

function Invoke-Step {
    param(
        [string] $Name,
        [string] $LogPath,
        [string[]] $CommandArgs
    )

    Write-Status "START $Name"
    New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent) | Out-Null
    Push-Location $repoRoot
    try {
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $state.PythonExe -u @CommandArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
        $ErrorActionPreference = $previousErrorActionPreference
        if ($LASTEXITCODE -ne 0) {
            throw "$Name failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        $ErrorActionPreference = "Stop"
        Pop-Location
    }
    Write-Status "DONE $Name"
}

function Resolve-LatestSpeciesBest {
    $roots = @(
        (Join-Path $repoRoot "runs\detect\artifacts\hybrid_shared_detector"),
        (Join-Path $repoRoot "runs\detect\runs\detect\artifacts\hybrid_shared_detector")
    )
    $candidates = @()
    foreach ($root in $roots) {
        if (Test-Path $root) {
            $candidates += Get-ChildItem -Path $root -Recurse -Filter best.pt |
                Where-Object { $_.FullName -match "hybrid_shared" -and $_.FullName -match "\\weights\\best\.pt$" }
        }
    }
    $latest = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) {
        throw "Could not resolve latest hybrid species checkpoint under known roots."
    }
    return $latest.FullName
}

function Resolve-LatestDirectBreedBest {
    $roots = @(
        (Join-Path $repoRoot "runs\detect\artifacts"),
        (Join-Path $repoRoot "runs\detect\runs\detect\artifacts")
    )
    $candidates = @()
    foreach ($root in $roots) {
        if (Test-Path $root) {
            $candidates += Get-ChildItem -Path $root -Recurse -Filter best.pt |
                Where-Object {
                    $_.FullName -match 'breed' -and
                    $_.FullName -match 'weights\\best\.pt$' -and
                    $_.FullName -notmatch 'hybrid_shared'
                }
        }
    }
    $latest = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) {
        return ""
    }
    return $latest.FullName
}

function Stop-StaleTrainingProcesses {
    $targets = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and
        ($_.Name -match 'python(\.exe)?$' -or $_.Name -match 'powershell(\.exe)?$') -and
        $_.CommandLine -and
        (
            $_.CommandLine -match 'train_species_aware_hybrid\.py' -or
            $_.CommandLine -match 'train_two_stage_breed_models\.py' -or
            $_.CommandLine -match 'train_detector\.py' -or
            $_.CommandLine -match 'build_temporary_breed_run_dataset\.py' -or
            $_.CommandLine -match 'rebuild_and_retrain_strict_labels_visible_amd\.ps1'
        )
    }

    foreach ($proc in $targets) {
        Write-Status ("Stopping stale process pid={0} name={1}" -f $proc.ProcessId, $proc.Name)
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
Write-Status "Strict rebuild/retrain pipeline starting."
Write-Status "repo=$repoRoot"
Write-Status "python=$($state.PythonExe)"
Write-Status "logs_root=$reportRoot"
Write-Status "epochs stage1=$Stage1Epochs stage2=$Stage2Epochs direct_yolo=$DirectYoloEpochs"

Stop-StaleTrainingProcesses
$previousBest = Resolve-LatestSpeciesBest
$previousDirectBreedBest = Resolve-LatestDirectBreedBest
$newSpeciesBest = Join-Path $hybridProject "$speciesRunName\weights\best.pt"
if (-not [string]::IsNullOrWhiteSpace($ResumeSpeciesBest)) {
    $newSpeciesBest = $ResumeSpeciesBest
}

Write-Status "previous_species_best=$previousBest"
if (-not [string]::IsNullOrWhiteSpace($previousDirectBreedBest)) {
    Write-Status "previous_direct_breed_best=$previousDirectBreedBest"
}
Write-Status "new_species_run=$speciesRunName"
Write-Status "breed_run=$breedRunName"

if (-not $SkipMerge) {
    Invoke-Step -Name "merge_phr_labeled_images" -LogPath $mergeLog -CommandArgs @(
        "tools\merge_phr_labeled_images.py"
    )
}
else {
    Write-Status "SKIP merge_phr_labeled_images"
}

if (-not $SkipStage1) {
    Invoke-Step -Name "stage1_species_continue" -LogPath $stage1Log -CommandArgs @(
        "tools\train_species_aware_hybrid.py",
        "data=animal_breed_v0",
        "data.species_aware.hybrid.train.enabled=true",
        "data.species_aware.hybrid.train.device=0",
        "data.species_aware.hybrid.train.base_model=$previousBest",
        "data.species_aware.hybrid.train.epochs=$Stage1Epochs",
        "data.species_aware.hybrid.train.project=$hybridProject",
        "data.species_aware.hybrid.train.name=$speciesRunName"
    )
}
else {
    Write-Status "SKIP stage1_species_continue"
}

if (-not (Test-Path $newSpeciesBest)) {
    throw "Stage-1 best checkpoint missing after training: $newSpeciesBest"
}
Write-Status "stage1_best=$newSpeciesBest"

if (-not $SkipBreedBuild) {
    Invoke-Step -Name "build_temporary_breed_run_dataset" -LogPath $breedBuildLog -CommandArgs @(
        "tools\build_temporary_breed_run_dataset.py",
        "--species-model", $newSpeciesBest,
        "--run-name", $breedRunName,
        "--device", "0",
        "--batch-size", "$BreedBuildBatchSize",
        "--resume-from-progress"
    )
}
else {
    Write-Status "SKIP build_temporary_breed_run_dataset"
}

if (-not $SkipFinalize) {
    Invoke-Step -Name "finalize_breed_run_strict" -LogPath $reviewLog -CommandArgs @(
        "tools\finalize_breed_run_strict.py",
        "--manifest-csv", (Join-Path $breedOutDir "manifest.csv"),
        "--taxonomy", (Join-Path $breedOutDir "taxonomy.yaml"),
        "--out-manifest-csv", $stage2ManifestCsv,
        "--out-manifest-parquet", $stage2ManifestParquet,
        "--out-taxonomy", $stage2TaxonomyYaml,
        "--out-direct-manifest-csv", $directManifestCsv,
        "--out-direct-manifest-parquet", $directManifestParquet,
        "--out-direct-taxonomy", $directTaxonomyYaml,
        "--review-report-csv", $reviewCsv,
        "--review-report-json", $reviewJson
    )
}
else {
    Write-Status "SKIP finalize_breed_run_strict"
}

if (-not $SkipPrepareDirect) {
    Invoke-Step -Name "prepare_direct_breed_yolo_dataset" -LogPath $reviewLog -CommandArgs @(
        "tools\prepare_yolo_dataset.py",
        "data=animal_breed_v0",
        "model/detector=yolo11_breed",
        "data.split_manifest_path=$directManifestCsv",
        "data.taxonomy_path=$directTaxonomyYaml",
        "data.prepared_yolo_breed_dir=$directPreparedDir",
        "model.detector.data_yaml_path=$directYaml",
        "model.detector.classes=species_or_breed"
    )
}
else {
    Write-Status "SKIP prepare_direct_breed_yolo_dataset"
}

Invoke-Step -Name "stage2_efficientnet_b2" -LogPath $stage2Log -CommandArgs @(
    "tools\train_two_stage_breed_models.py",
    "data=animal_breed_v0",
    "model/classifier=efficientnetb2",
    "data.split_manifest_path=$stage2ManifestCsv",
    "data.taxonomy_path=$stage2TaxonomyYaml",
    "model.classifier.train.epochs=$Stage2Epochs",
    "+stage2_warm_start_from_old=true",
    "train.device=0",
    "train.resume=false"
)

if ([string]::IsNullOrWhiteSpace($previousDirectBreedBest)) {
    $previousDirectBreedBest = $newSpeciesBest
}

Invoke-Step -Name "direct_breed_yolo" -LogPath $directLog -CommandArgs @(
    "tools\train_detector.py",
    "data=animal_breed_v0",
    "model/detector=yolo11_breed",
    "data.split_manifest_path=$directManifestCsv",
    "data.taxonomy_path=$directTaxonomyYaml",
    "data.prepared_yolo_breed_dir=$directPreparedDir",
    "model.detector.data_yaml_path=$directYaml",
    "model.detector.base_model=$previousDirectBreedBest",
    "model.detector.name=yolo11_breed_strict_$runStamp",
    "model.detector.train.epochs=$DirectYoloEpochs",
    "model.detector.train.batch=$DirectYoloBatch",
    "model.detector.train.workers=$DirectYoloWorkers",
    "train.device=0",
    "train.resume=false"
)

Write-Status "Strict rebuild/retrain pipeline completed."
Write-Status "stage2_manifest=$stage2ManifestCsv"
Write-Status "direct_yaml=$directYaml"
