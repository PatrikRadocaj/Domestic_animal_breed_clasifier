param(
    [int] $Stage1Epochs = 50,
    [int] $Stage1Batch = 16,
    [int] $Stage1Workers = 16,
    [int] $Stage2Epochs = 10,
    [int] $DirectYoloEpochs = 50,
    [int] $BreedBuildBatchSize = 32,
    [int] $DirectYoloBatch = 4,
    [int] $DirectYoloWorkers = 4,
    [switch] $SkipNormalization,
    [switch] $SkipDuplicateCleanup,
    [string] $RunStamp = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "amd_gpu_env.ps1")
$state = Set-AmdGpuProjectEnv
$repoRoot = $state.RepoRoot
$runStamp = if ([string]::IsNullOrWhiteSpace($RunStamp)) { Get-Date -Format "yyyyMMdd_HHmmss" } else { $RunStamp }
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
$stage1Log = Join-Path $reportRoot "01_stage1_species.log"
$normalizeLog = Join-Path $reportRoot "00_yolo_image_normalization.log"
$duplicateCleanupLog = Join-Path $reportRoot "00_duplicate_source_cleanup.log"
$stage1SpecLog = Join-Path $reportRoot "00_stage1_build_spec.log"
$stage1StressLog = Join-Path $reportRoot "00_stage1_dataloader_stress.log"
$stage1EvalLog = Join-Path $reportRoot "02_stage1_species_eval.log"
$preStage1QaLog = Join-Path $reportRoot "02a_pre_stage1_qa.log"
$postStage1QaLog = Join-Path $reportRoot "02b_post_stage1_qa.log"
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

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
try {
    Write-Status "Full breed retrain pipeline starting."
    Write-Status "repo=$repoRoot"
    Write-Status "python=$($state.PythonExe)"
    Write-Status "logs_root=$reportRoot"
    Write-Status "epochs stage1=$Stage1Epochs stage2=$Stage2Epochs direct_yolo=$DirectYoloEpochs"
    Write-Status "stage1_batch=$Stage1Batch"
    Write-Status "stage1_workers=$Stage1Workers"

    Stop-StaleTrainingProcesses

    $newSpeciesBest = Join-Path $hybridProject "$speciesRunName\weights\best.pt"
    $hybridSpecPath = Join-Path $hybridOutputDir "hybrid_dataset_spec.json"
    $stage1EvalOut = Join-Path $reportRoot "stage1_species_eval"
    $stage2EvalOut = Join-Path $reportRoot "stage2_confusions"

    Write-Status "new_species_run=$speciesRunName"
    Write-Status "hybrid_output_dir=$hybridOutputDir"
    Write-Status "breed_run=$breedRunName"

    if ($SkipNormalization) {
        Write-Status "SKIP normalize_yolo_source_images_in_place"
    }
    else {
        Invoke-Step -Name "normalize_yolo_source_images_in_place" -LogPath $normalizeLog -CommandArgs @(
            "tools\normalize_yolo_source_images.py",
            "--report-csv", (Join-Path $reportRoot "yolo_image_normalization_report.csv"),
            "--summary-json", (Join-Path $reportRoot "yolo_image_normalization_summary.json"),
            "--max-side", "2048"
        )
    }

    if ($SkipDuplicateCleanup) {
        Write-Status "SKIP delete_verified_phr_duplicate_yolo_sources"
    }
    else {
        Invoke-Step -Name "delete_verified_phr_duplicate_yolo_sources" -LogPath $duplicateCleanupLog -CommandArgs @(
            "tools\delete_phr_duplicate_yolo_sources.py",
            "--delete",
            "--report-csv", (Join-Path $reportRoot "duplicate_source_deletion_report.csv"),
            "--summary-json", (Join-Path $reportRoot "duplicate_source_deletion_summary.json")
        )
    }

    Invoke-Step -Name "qa_pre_stage1" -LogPath $preStage1QaLog -CommandArgs @(
        "tools\qa_gate_full_breed_pipeline.py",
        "pre_stage1",
        "--expect-gpu",
        "--report-json", (Join-Path $reportRoot "qa_pre_stage1_report.json")
    )

    Invoke-Step -Name "stage1_build_hybrid_spec_preflight" -LogPath $stage1SpecLog -CommandArgs @(
        "tools\train_species_aware_hybrid.py",
        "data=animal_breed_v0",
        "data.species_aware.hybrid.output_dir=$hybridOutputDir",
        "data.species_aware.hybrid.train.enabled=false",
        "data.species_aware.hybrid.train.device=0",
        "data.species_aware.hybrid.train.base_model=yolo11s.pt",
        "data.species_aware.hybrid.train.epochs=$Stage1Epochs",
        "data.species_aware.hybrid.train.batch=$Stage1Batch",
        "data.species_aware.hybrid.train.workers=$Stage1Workers"
    )

    if (-not (Test-Path $hybridSpecPath)) {
        throw "Hybrid dataset spec missing after preflight build: $hybridSpecPath"
    }

    Invoke-Step -Name "stage1_hybrid_dataloader_stress_b16_w16" -LogPath $stage1StressLog -CommandArgs @(
        "tools\stress_hybrid_dataloader.py",
        "--hybrid-spec", $hybridSpecPath,
        "--batch", "$Stage1Batch",
        "--workers", "$Stage1Workers",
        "--imgsz", "1280",
        "--min-batches", "240",
        "--report-json", (Join-Path $reportRoot "stage1_dataloader_stress_report.json")
    )

    Invoke-Step -Name "stage1_species_fresh" -LogPath $stage1Log -CommandArgs @(
        "tools\train_species_aware_hybrid.py",
        "data=animal_breed_v0",
        "data.species_aware.hybrid.output_dir=$hybridOutputDir",
        "data.species_aware.hybrid.train.enabled=true",
        "data.species_aware.hybrid.train.device=0",
        "data.species_aware.hybrid.train.base_model=yolo11s.pt",
        "data.species_aware.hybrid.train.epochs=$Stage1Epochs",
        "data.species_aware.hybrid.train.batch=$Stage1Batch",
        "data.species_aware.hybrid.train.workers=$Stage1Workers",
        "data.species_aware.hybrid.train.project=$hybridProject",
        "data.species_aware.hybrid.train.name=$speciesRunName"
    )

    if (-not (Test-Path $newSpeciesBest)) {
        throw "Stage-1 best checkpoint missing after training: $newSpeciesBest"
    }
    if (-not (Test-Path $hybridSpecPath)) {
        throw "Hybrid dataset spec missing after stage1: $hybridSpecPath"
    }
    Write-Status "stage1_best=$newSpeciesBest"

    Invoke-Step -Name "qa_post_stage1_for_mining" -LogPath $postStage1QaLog -CommandArgs @(
        "tools\qa_gate_full_breed_pipeline.py",
        "post_stage1_for_mining",
        "--hybrid-spec", $hybridSpecPath,
        "--stage1-best", $newSpeciesBest,
        "--stage1-results", (Join-Path $hybridProject "$speciesRunName\results.csv"),
        "--report-json", (Join-Path $reportRoot "qa_post_stage1_for_mining_report.json")
    )

    Invoke-Step -Name "stage1_species_confusion" -LogPath $stage1EvalLog -CommandArgs @(
        "tools\evaluate_hybrid_species_confusion.py",
        "--weights", $newSpeciesBest,
        "--hybrid-spec", $hybridSpecPath,
        "--output-dir", $stage1EvalOut,
        "--conf", "0.001",
        "--iou-match", "0.5"
    )

    Invoke-Step -Name "build_temporary_breed_run_dataset" -LogPath $breedBuildLog -CommandArgs @(
        "tools\build_temporary_breed_run_dataset.py",
        "--species-model", $newSpeciesBest,
        "--run-name", $breedRunName,
        "--device", "0",
        "--batch-size", "$BreedBuildBatchSize",
        "--resume-from-progress"
    )

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

    Invoke-Step -Name "label_taxonomy_audit_post_mining" -LogPath $postMiningLabelAuditLog -CommandArgs @(
        "tools\audit_label_taxonomy_qa.py",
        "--out-dir", $postMiningLabelAuditOut,
        "--processed-run-name", $breedRunName
    )

    Invoke-Step -Name "qa_post_breed_finalize_pre_stage2" -LogPath $postStage2InputQaLog -CommandArgs @(
        "tools\qa_gate_full_breed_pipeline.py",
        "post_breed_finalize_pre_stage2",
        "--manifest-csv", $stage2ManifestCsv,
        "--taxonomy-yaml", $stage2TaxonomyYaml,
        "--build-summary-json", (Join-Path $breedOutDir "summary.json"),
        "--review-json", $reviewJson,
        "--review-csv", $reviewCsv,
        "--report-json", (Join-Path $reportRoot "qa_post_breed_finalize_pre_stage2_report.json")
    )

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

    Invoke-Step -Name "qa_post_direct_prep_pre_direct_yolo" -LogPath $postDirectPrepQaLog -CommandArgs @(
        "tools\qa_gate_full_breed_pipeline.py",
        "post_direct_prep_pre_direct_yolo",
        "--data-yaml", $directYaml,
        "--manifest-csv", $directManifestCsv,
        "--report-json", (Join-Path $reportRoot "qa_post_direct_prep_pre_direct_yolo_report.json")
    )

    Invoke-Step -Name "stage2_efficientnet_b2" -LogPath $stage2Log -CommandArgs @(
        "tools\train_two_stage_breed_models.py",
        "data=animal_breed_v0",
        "model/classifier=efficientnetb2",
        "data.split_manifest_path=$stage2ManifestCsv",
        "data.taxonomy_path=$stage2TaxonomyYaml",
        "model.classifier.train.epochs=$Stage2Epochs",
        "+stage2_warm_start_from_old=false",
        "train.device=0",
        "train.resume=false"
    )

    $stage2SummaryCsv = Resolve-LatestStage2Summary
    Write-Status "stage2_summary=$stage2SummaryCsv"

    Invoke-Step -Name "stage2_confusion_export" -LogPath $stage2EvalLog -CommandArgs @(
        "tools\evaluate_stage2_confusions.py",
        "--manifest-csv", $stage2ManifestCsv,
        "--taxonomy-yaml", $stage2TaxonomyYaml,
        "--summary-csv", $stage2SummaryCsv,
        "--output-dir", $stage2EvalOut,
        "--device", "0",
        "--batch-size", "64",
        "--num-workers", "4"
    )

    Invoke-Step -Name "direct_breed_yolo" -LogPath $directLog -CommandArgs @(
        "tools\train_detector.py",
        "data=animal_breed_v0",
        "model/detector=yolo11_breed",
        "data.split_manifest_path=$directManifestCsv",
        "data.taxonomy_path=$directTaxonomyYaml",
        "data.prepared_yolo_breed_dir=$directPreparedDir",
        "model.detector.data_yaml_path=$directYaml",
        "model.detector.base_model=$newSpeciesBest",
        "model.detector.name=yolo11_breed_retrain_$runStamp",
        "model.detector.train.epochs=$DirectYoloEpochs",
        "model.detector.train.batch=$DirectYoloBatch",
        "model.detector.train.workers=$DirectYoloWorkers",
        "train.device=0",
        "train.resume=false"
    )

    Write-Status "Full breed retrain pipeline completed."
    Write-Status "stage1_best=$newSpeciesBest"
    Write-Status "stage1_confusions=$stage1EvalOut"
    Write-Status "stage2_manifest=$stage2ManifestCsv"
    Write-Status "stage2_confusions=$stage2EvalOut"
    Write-Status "direct_yaml=$directYaml"
}
catch {
    Write-Status ("FAILED: {0}" -f $_.Exception.Message)
    throw
}
