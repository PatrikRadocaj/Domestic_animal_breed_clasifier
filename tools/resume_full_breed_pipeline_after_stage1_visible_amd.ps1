param(
    [string] $RunStamp = "qa_full_20260706_b12w12_yolo40",
    [int] $Stage2Epochs = 5,
    [int] $DirectYoloEpochs = 5,
    [int] $BreedBuildBatchSize = 32,
    [int] $DirectYoloBatch = 4,
    [int] $DirectYoloWorkers = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "amd_gpu_env.ps1")
$state = Set-AmdGpuProjectEnv
$repoRoot = $state.RepoRoot
$runStamp = $RunStamp
$reportRoot = Join-Path $repoRoot "results\reports\full_breed_pipeline\$runStamp"
$hybridProject = Join-Path $repoRoot "runs\detect\artifacts\hybrid_shared_detector"
$hybridOutputDir = Join-Path $repoRoot "data\processed\species_aware\hybrid_shared_$runStamp"
$speciesRunName = "hybrid_shared_fresh_$runStamp"
$breedRunName = "temporary_breed_run_retrain_$runStamp"
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
$breedBuildLog = Join-Path $reportRoot "03_breed_build.log"
$reviewLog = Join-Path $reportRoot "04_label_review.log"
$postMiningLabelAuditLog = Join-Path $reportRoot "04b_label_taxonomy_audit_post_mining.log"
$postMiningLabelAuditOut = Join-Path $reportRoot "04b_label_taxonomy_audit_post_mining"
$postStage2InputQaLog = Join-Path $reportRoot "04a_post_breed_finalize_qa.log"
$stage2Log = Join-Path $reportRoot "05_stage2_efficientnet.log"
$stage2EvalLog = Join-Path $reportRoot "06_stage2_confusions.log"
$postDirectPrepQaLog = Join-Path $reportRoot "06a_post_direct_prep_qa.log"
$directLog = Join-Path $reportRoot "07_direct_breed_yolo.log"

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

function Stop-StaleTrainingProcesses {
    $targets = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and
        ($_.Name -match 'python(\.exe)?$') -and
        $_.CommandLine -and
        (
            $_.CommandLine -match 'train_species_aware_hybrid\.py' -or
            $_.CommandLine -match 'evaluate_hybrid_species_confusion\.py' -or
            $_.CommandLine -match 'train_two_stage_breed_models\.py' -or
            $_.CommandLine -match 'train_detector\.py' -or
            $_.CommandLine -match 'build_temporary_breed_run_dataset\.py'
        )
    }

    foreach ($proc in $targets) {
        Write-Status ("Stopping stale process pid={0} name={1}" -f $proc.ProcessId, $proc.Name)
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Resolve-LatestStage2Summary {
    $root = Join-Path $repoRoot "artifacts"
    $latest = Get-ChildItem -Path $root -Directory -Filter "two-stage-breed-models-*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) {
        throw "Could not resolve latest stage2 artifact directory."
    }
    $summaryCsv = Join-Path $latest.FullName "metrics\two_stage_breed_models_summary.csv"
    if (-not (Test-Path $summaryCsv)) {
        throw "Missing stage2 summary csv: $summaryCsv"
    }
    return $summaryCsv
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

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
try {
    Write-Status "Resume full breed pipeline after completed Stage-1."
    Write-Status "repo=$repoRoot"
    Write-Status "python=$($state.PythonExe)"
    Write-Status "run_stamp=$runStamp"
    Write-Status "stage2_epochs=$Stage2Epochs direct_yolo_epochs=$DirectYoloEpochs"

    Stop-StaleTrainingProcesses

    $newSpeciesBest = Join-Path $hybridProject "$speciesRunName\weights\best.pt"
    $hybridSpecPath = Join-Path $hybridOutputDir "hybrid_dataset_spec.json"
    $stage2EvalOut = Join-Path $reportRoot "stage2_confusions"
    $previousDirectBreedBest = Resolve-LatestDirectBreedBest

    if (-not (Test-Path $newSpeciesBest)) {
        throw "Missing completed Stage-1 best checkpoint: $newSpeciesBest"
    }
    if (-not (Test-Path $hybridSpecPath)) {
        throw "Missing completed Stage-1 hybrid spec: $hybridSpecPath"
    }
    Write-Status "resume_stage1_best=$newSpeciesBest"
    Write-Status "resume_hybrid_spec=$hybridSpecPath"
    if (-not [string]::IsNullOrWhiteSpace($previousDirectBreedBest)) {
        Write-Status "previous_direct_breed_best=$previousDirectBreedBest"
    }

    Invoke-Step -Name "build_temporary_breed_run_dataset_resume" -LogPath $breedBuildLog -CommandArgs @(
        "tools\build_temporary_breed_run_dataset.py",
        "--species-model", $newSpeciesBest,
        "--run-name", $breedRunName,
        "--device", "0",
        "--batch-size", "$BreedBuildBatchSize",
        "--resume-from-progress"
    )

    Invoke-Step -Name "finalize_breed_run_strict_resume" -LogPath $reviewLog -CommandArgs @(
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

    Invoke-Step -Name "label_taxonomy_audit_post_mining_resume" -LogPath $postMiningLabelAuditLog -CommandArgs @(
        "tools\audit_label_taxonomy_qa.py",
        "--out-dir", $postMiningLabelAuditOut,
        "--processed-run-name", $breedRunName
    )

    Invoke-Step -Name "qa_post_breed_finalize_pre_stage2_resume" -LogPath $postStage2InputQaLog -CommandArgs @(
        "tools\qa_gate_full_breed_pipeline.py",
        "post_breed_finalize_pre_stage2",
        "--manifest-csv", $stage2ManifestCsv,
        "--taxonomy-yaml", $stage2TaxonomyYaml,
        "--build-summary-json", (Join-Path $breedOutDir "summary.json"),
        "--review-json", $reviewJson,
        "--review-csv", $reviewCsv,
        "--report-json", (Join-Path $reportRoot "qa_post_breed_finalize_pre_stage2_report.json")
    )

    Invoke-Step -Name "prepare_direct_breed_yolo_dataset_resume" -LogPath $reviewLog -CommandArgs @(
        "tools\prepare_yolo_dataset.py",
        "data=animal_breed_v0",
        "model/detector=yolo11_breed",
        "data.split_manifest_path=$directManifestCsv",
        "data.taxonomy_path=$directTaxonomyYaml",
        "data.prepared_yolo_breed_dir=$directPreparedDir",
        "model.detector.data_yaml_path=$directYaml",
        "model.detector.classes=species_or_breed"
    )

    Invoke-Step -Name "qa_post_direct_prep_pre_direct_yolo_resume" -LogPath $postDirectPrepQaLog -CommandArgs @(
        "tools\qa_gate_full_breed_pipeline.py",
        "post_direct_prep_pre_direct_yolo",
        "--data-yaml", $directYaml,
        "--manifest-csv", $directManifestCsv,
        "--report-json", (Join-Path $reportRoot "qa_post_direct_prep_pre_direct_yolo_report.json")
    )

    Invoke-Step -Name "stage2_efficientnet_b2_resume" -LogPath $stage2Log -CommandArgs @(
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

    $stage2SummaryCsv = Resolve-LatestStage2Summary
    Write-Status "stage2_summary=$stage2SummaryCsv"

    Invoke-Step -Name "stage2_confusion_export_resume" -LogPath $stage2EvalLog -CommandArgs @(
        "tools\evaluate_stage2_confusions.py",
        "--manifest-csv", $stage2ManifestCsv,
        "--taxonomy-yaml", $stage2TaxonomyYaml,
        "--summary-csv", $stage2SummaryCsv,
        "--output-dir", $stage2EvalOut,
        "--device", "0",
        "--batch-size", "64",
        "--num-workers", "4"
    )

    $directBaseModel = $previousDirectBreedBest
    if ([string]::IsNullOrWhiteSpace($directBaseModel)) {
        $directBaseModel = $newSpeciesBest
    }
    Write-Status "direct_breed_base_model=$directBaseModel"

    Invoke-Step -Name "direct_breed_yolo_resume" -LogPath $directLog -CommandArgs @(
        "tools\train_detector.py",
        "data=animal_breed_v0",
        "model/detector=yolo11_breed",
        "data.split_manifest_path=$directManifestCsv",
        "data.taxonomy_path=$directTaxonomyYaml",
        "data.prepared_yolo_breed_dir=$directPreparedDir",
        "model.detector.data_yaml_path=$directYaml",
        "model.detector.base_model=$directBaseModel",
        "model.detector.name=yolo11_breed_retrain_${runStamp}_resume",
        "model.detector.train.epochs=$DirectYoloEpochs",
        "model.detector.train.batch=$DirectYoloBatch",
        "model.detector.train.workers=$DirectYoloWorkers",
        "train.device=0",
        "train.resume=false"
    )

    Write-Status "Resume full breed pipeline completed."
    Write-Status "stage1_best=$newSpeciesBest"
    Write-Status "stage2_manifest=$stage2ManifestCsv"
    Write-Status "stage2_confusions=$stage2EvalOut"
    Write-Status "direct_yaml=$directYaml"
}
catch {
    Write-Status ("FAILED: {0}" -f $_.Exception.Message)
    throw
}
