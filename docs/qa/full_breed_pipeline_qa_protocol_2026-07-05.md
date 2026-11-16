# Full Breed Pipeline QA Protocol

Date: 2026-07-05
Scope: Stage 1 species detection, pseudo-label mining on unlabeled images, Stage 2 breed classification, and direct breed YOLO.
Hardware target: AMD Radeon RX 7800 XT
Execution mode: visible PowerShell terminals with live logs and persistent artifacts

## 1. QA Objectives

This protocol is designed to prevent four classes of failure:

1. Data integrity failures
2. Pseudo-label quality failures
3. Training stability and reproducibility failures
4. Silent model quality regressions between iterations

The protocol is intentionally conservative. A stage does not proceed downstream until its inputs and outputs pass the defined gates.

## 2. Pipeline Under Test

Current implementation entrypoints in this repository:

- Stage 1 hybrid species YOLO training:
  - `tools/train_species_aware_hybrid.py`
  - `src/animal_breed/detection/hybrid_species_training.py`
  - `src/animal_breed/detection/hybrid_species_dataset.py`
- Breed rebuild and pseudo-label generation:
  - `tools/build_temporary_breed_run_dataset.py`
  - `tools/finalize_breed_run_strict.py`
- Stage 2 breed classification:
  - `tools/train_two_stage_breed_models.py`
  - `src/animal_breed/trainers/classifier_trainer.py`
- Direct breed YOLO:
  - `tools/prepare_yolo_dataset.py`
  - `tools/train_detector.py`
- Post-train confusion export:
  - `tools/evaluate_hybrid_species_confusion.py`
  - `tools/evaluate_stage2_confusions.py`

## 3. Known Current Risks

These are not hypothetical. They already appeared in this codebase and must be part of QA:

1. Stage 1 custom dataset incompatibility with Ultralytics helper plotting
2. Stage 1 worker crashes from memory-heavy Albumentations noise transforms
3. Corrupt-but-readable JPEGs producing warning noise during training
4. Potential mismatch between Stage 1 snapshot data and later rebuilt Stage 2/direct-breed data if raw sources are modified mid-run
5. Windows visible-terminal quoting and launcher fragility
6. Breed taxonomy drift caused by synonym, spelling, and species-vs-breed leakage

## 4. Release Model

Each full run is treated as a release candidate with these immutable artifacts:

- source snapshot description
- Stage 1 dataset spec snapshot
- Stage 1 best and last checkpoints
- mining report and acceptance statistics
- strict label review report
- Stage 2 trainable manifest and taxonomy
- direct-breed YOLO manifest and taxonomy
- confusion matrices and metrics summaries for all trained stages

No downstream stage may silently reuse stale artifacts from a previous run without explicit QA signoff.

## 5. Pre-Flight QA Before Any Training

### 5.1 Dataset immutability gate

Freeze the source tree during a full run:

- `data/raw/yolo_sources`
- `data/raw/non_yolo_sources`

Rule:

- No manual integration, renaming, or file movement after Stage 1 starts.
- If raw sources must change, the full pipeline is invalidated and must restart from a new Stage 1 snapshot.

Alarm:

- Any file-count delta or modified timestamp burst in source directories during an active run.

### 5.2 Manifest and path integrity gate

For every manifest used downstream:

- all image paths must exist
- all label paths must exist when expected
- all metadata files must parse
- no duplicate row keys after canonicalization
- no empty train or validation split

Hard fail conditions:

- missing image path rate > 0.1%
- missing label path rate > 0.1%
- unreadable image rate > 0.5%
- duplicate primary sample IDs > 0

### 5.3 Taxonomy hygiene gate

Enforce:

- one canonical species ID per species
- one canonical breed ID per breed
- no unknown, unlabeled, placeholder, location, color-only, phenotype-only, or pure species tags inside Stage 2 breed classes
- species with fewer than 2 real breed classes must not spawn Stage 2 submodels
- those species remain valid direct-breed YOLO outputs at species level

Required artifacts:

- canonical mapping report
- excluded label report with reasons
- ambiguous label report marked for review

Hard fail:

- any Stage 2 class without provenance to a reviewed canonical raw label

## 6. Stage 1 QA Protocol: Species Detection

### 6.1 Data QA

Measure and record:

- number of source train records
- number of Stage 1 train plans
- number of validation samples
- number of test samples
- per-species counts
- excluded species and exclusion reasons
- source contribution by dataset

Required distribution checks:

- max/min per-species train count ratio
- percentage of synthetic vs real vs augmented_single train plans
- percentage of mixed-species images
- percentage of pure-species images

Alarm thresholds:

- any species count drops > 20% from previous accepted run without an explained data change
- train-plan mix changes by > 10 percentage points without explicit config change

### 6.2 Model performance QA

Primary metrics:

- mAP50
- mAP50-95
- precision
- recall
- per-class AP
- confusion matrix counts and normalized matrix

Secondary diagnostics:

- PR curves
- confidence distributions for TP, FP, FN
- object-size breakdown AP: small, medium, large
- crowded-image error rate

Minimum acceptance gates for Stage 1 candidate:

- no catastrophic species collapse: no common species with recall < 0.40 unless count is extremely small
- global mAP50-95 must not regress > 3 absolute points versus last accepted run without documented justification
- any species with support >= 300 images and AP drop > 7 absolute points triggers manual review
- FP-heavy species with precision < 0.35 trigger manual error audit

### 6.3 Manual review protocol for Stage 1

Sample manually from validation predictions:

- 50 highest-confidence false positives
- 50 highest-confidence false negatives
- 20 borderline detections near confidence threshold
- 20 crowded scenes
- 20 extreme domain-shift images

Review questions:

1. wrong species
2. box on background artifact
3. missed object despite clear visibility
4. box too large or too small
5. source-domain-specific failure

Escalation:

- if > 15% of audited high-confidence detections are clearly wrong, freeze downstream mining

### 6.4 Statistical QA for Stage 1

Use bootstrap confidence intervals on:

- mAP50
- mAP50-95
- recall
- per-species AP for key species

Recommended report:

- 1000 bootstrap resamples over validation images

Alarm:

- if the new model appears better only within overlapping confidence intervals and not by robust margin, do not automatically promote it as best for mining

## 7. Mining / Pseudo-Labeling QA Protocol

### 7.1 Mining policy

Mining uses only the best accepted Stage 1 checkpoint.

Pseudo-labels may be accepted only when all of the following are true:

- predicted species matches expected labeled species when species is known
- confidence passes strict threshold
- box size passes geometric filters
- agreement checks pass
- duplicate suppression passes

### 7.2 Required filters

Per accepted pseudo-label candidate:

- confidence threshold:
  - default strict gate: >= 0.80
  - high-risk species or domain-shift species: >= 0.88
- IoU consistency for test-time agreement or flip agreement:
  - minimum agreement IoU: >= 0.50
- minimum box side:
  - >= 24 px
- minimum area:
  - >= 32 x 32 equivalent
- maximum area fraction:
  - reject implausibly huge boxes covering nearly the whole image unless species/domain explicitly warrants it
- maximum detections per image:
  - species-specific cap or conservative global cap

### 7.3 Pseudo-label QA metrics

Track:

- number of candidate unlabeled images scanned
- accepted pseudo-label count
- accepted rate
- rejected rate by reason
- confidence distribution
- box area distribution
- per-species pseudo-label counts
- per-source-domain pseudo-label counts
- agreement IoU distribution

Hard fail:

- pseudo-label set dominated by a few species with no class cap
- pseudo-label confidence median < configured threshold margin
- one domain contributes > 50% of mined labels without explicit approval

### 7.4 Pseudo-label sample audit

Manual audit sample sizes:

- 200 accepted pseudo-labels overall minimum
- at least 20 per major species
- all species with support < 50 must be fully reviewed or excluded

Audit targets:

- 50 random accepted labels
- 50 lowest-confidence accepted labels
- 50 highest-confidence accepted labels
- 50 domain-shift samples

Acceptance thresholds:

- estimated precision of accepted pseudo-labels >= 95% overall
- no major species < 90% estimated precision in audited sample

If not met:

- raise threshold
- tighten agreement filter
- reduce source domains
- exclude unstable species from pseudo-label use

### 7.5 Domain shift QA

Partition mined samples by source domain:

- YOLO-native labeled domains
- Wikimedia-like web imagery
- mobile/phone imagery
- studio/pet breed dataset imagery
- noisy scanned or low-resolution domains

Required checks:

- compare confidence distributions across domains
- compare acceptance rate across domains
- compare manual precision estimates across domains

Alarm:

- domain acceptance rate much higher than expected but manual precision low
- indicates spurious confidence calibration

### 7.6 Degeneration checks during iterative mining

If mining is repeated across iterations, monitor:

- growth in pseudo-label share
- teacher-student agreement
- validation AP on fully human-labeled data
- class entropy and confidence collapse

Alarm:

- model improves on pseudo-labeled domains but degrades on human-labeled validation
- pseudo-label count grows while confidence quality falls
- confusion concentrates on a few sibling species

## 8. Stage 2 QA Protocol: Breed Classification

### 8.1 Input data QA

For each species submodel:

- number of human-labeled breed samples
- number of pseudo-labeled breed samples
- pseudo/human ratio
- number of classes
- per-class counts
- train/val split sizes

Hard fail:

- any class with < 5 validation images unless explicitly exempted as exploratory
- pseudo-labeled rows exceed 2x human-labeled rows for a species unless manually approved
- any species submodel with only one real breed class

### 8.2 Label cleanliness QA

Every Stage 2 row must have:

- canonical species_id
- canonical breed_id
- provenance to source row or pseudo-label record
- review status for ambiguous merges

Hard fail:

- species-level tags appearing in breed_id space
- unknown placeholders surviving into trainable Stage 2 taxonomy

### 8.3 Model performance QA

Per species:

- top-1 accuracy
- top-5 accuracy
- macro F1
- weighted F1
- per-class precision/recall/F1
- confusion matrix counts and normalized

Global aggregated:

- global breed confusion matrix in canonical breed space
- macro F1 across species
- weighted F1 across all validation rows

Acceptance gates:

- no species submodel with top-1 below naive majority baseline + 10 points
- no major breed class with recall collapse below 0.25 unless class support is tiny
- macro F1 regression > 5 absolute points vs previous accepted Stage 2 run triggers hold

### 8.4 Manual review protocol for Stage 2

For each trained species model:

- inspect top 20 highest-confidence wrong predictions
- inspect top 20 most frequent confusion pairs
- inspect all classes with support < 20

Questions:

1. real visual ambiguity or label issue
2. taxonomy synonym issue
3. species detector crop problem
4. pseudo-label contamination
5. class imbalance artifact

### 8.5 Overfitting / noisy-label detection

Track by epoch:

- train loss
- val loss
- train top-1
- val top-1
- val top-5

Alarm:

- train improves while val stalls for >= 3 epochs
- val loss worsens while pseudo-label-heavy classes dominate train accuracy
- high-confidence errors cluster in pseudo-labeled rows

Mitigations:

- lower pseudo-label weight
- cap pseudo-label ratio
- stricter filtering
- class-balanced sampling
- label smoothing

## 9. Direct Breed YOLO QA Protocol

### 9.1 Taxonomy QA

Direct breed YOLO classes must be:

- breed classes where breed taxonomy is valid
- species fallback classes where no real breed taxonomy exists

Hard fail:

- same semantic category represented both as species class and breed class for the same case
- species with no real breeds leaking fake breed placeholders

### 9.2 Detection + breed QA metrics

Track:

- mAP50
- mAP50-95
- precision
- recall
- per-class AP
- confusion matrix
- species-level collapsed metrics
- breed-level metrics within species

Because this model mixes species and breed outputs, report both:

- flat class metrics
- collapsed species metrics

### 9.3 Comparison with two-stage pipeline

Compare on the same held-out validation set:

- detection recall at species level
- breed top-1 or AP within detected regions
- end-to-end breed success rate
- latency / throughput
- failure decomposition:
  - Stage 1 miss
  - Stage 2 breed confusion
  - direct YOLO miss
  - direct YOLO sibling-breed confusion

Primary decision metric:

- end-to-end breed correctness on human-labeled validation

## 10. Class Balance QA

For every stage, produce:

- per-class sample counts
- effective sample counts after filtering
- human vs pseudo counts
- long-tail ratio

Alarm:

- max/min effective class count ratio > 100 without mitigation
- pseudo-labels disproportionately inflate already dominant classes

Mitigations:

- class-balanced sampling
- focal loss where appropriate
- per-species caps during mining
- minority oversampling only after validating it does not amplify label noise

## 11. Bias and Fairness QA Within This Dataset Context

This is not demographic fairness; this is data-source and class bias.

Check for:

- source-domain bias
- geography bias from web sources
- breed popularity bias
- pet-photo vs livestock-photo bias
- background leakage

Tests:

- leave-one-domain-out validation where feasible
- background sensitivity audit
- class-wise calibration plots

Alarm:

- performance is strong only on familiar photo style
- confidence is inflated on one source family

## 12. Reproducibility Protocol

For every accepted run, store:

- run config
- code snapshot / git status
- manifest paths
- taxonomy paths
- training device info
- seed
- output artifact paths
- checkpoint hashes

Run-to-run reproducibility check:

- repeat at least one reduced pilot run with same seed
- metric drift beyond expected random variance must be explained

## 13. Operational Rules for Visible AMD GPU Runs

All heavy jobs must run in visible terminals:

- Stage 1 training terminal
- mining/build terminal
- Stage 2 terminal
- direct breed YOLO terminal

Required startup logging:

- requested device
- resolved device
- `torch.cuda.is_available()`
- `torch.version.hip`
- GPU name
- first model parameter device
- first batch tensor device

Hard fail:

- requested GPU but model or tensors end up on CPU

Preferred operational pattern:

- one visible PowerShell for the orchestrated pipeline
- one visible tail window per active stage log

## 14. Alarm Signals and Stop Conditions

Immediate stop:

- training on CPU when GPU requested
- any manifest/taxonomy mismatch
- unreadable image explosion
- NaN loss
- checkpoint save failure
- pseudo-label audit precision below acceptance gate
- large unexplained metric regression

Yellow flags:

- rising confidence with worsening validation
- concentration of mined labels in one domain or one class
- recurrent dataloader worker crashes
- inconsistent progress logs

## 15. Recommended Threshold Table

These are starting defaults, not universal truths.

- Stage 1 mining confidence: `>= 0.80`
- High-risk mining confidence: `>= 0.88`
- flip/agreement IoU: `>= 0.50`
- Stage 1 regression tolerance on mAP50-95: `<= 3 absolute points`
- Stage 2 regression tolerance on macro F1: `<= 5 absolute points`
- pseudo-label audited precision: `>= 95% overall`
- per-major-species pseudo-label audited precision: `>= 90%`
- missing/unreadable input rate: `< 0.5%`

## 16. Training Procedure After QA Approval

After QA gates pass:

1. Stage 1
   - train best species detector path for 50 epochs on AMD GPU
   - export best and last checkpoints
   - export species confusion matrix from best checkpoint
2. Mining / breed rebuild
   - run best accepted Stage 1 checkpoint on unlabeled/non-YOLO breed images
   - apply strict confidence and consistency filters
   - produce mining report and audit sample
   - run strict label finalization
3. Stage 2
   - train EfficientNetB2 breed submodels for 10 epochs on GPU
   - export per-species and global confusion matrices
4. Direct breed YOLO
   - initialize from best accepted Stage 1 checkpoint
   - train for 50 epochs on GPU
   - export detector metrics and confusion outputs
5. Comparative evaluation
   - compare two-stage and direct end-to-end performance on same validation split

## 17. Future Iteration Improvements

Recommended next upgrades:

1. Active learning
   - prioritize manual labeling of high-confidence errors and confusion hotspots
2. Confidence calibration
   - calibrate detector confidence per species/domain before mining
3. Teacher-student co-training
   - validate pseudo-labels using more than one model family
4. Domain-aware filtering
   - source-specific thresholds instead of one global threshold
5. Noise-robust Stage 2 training
   - lower weight for pseudo-labeled rows
6. Background leakage tests
   - saliency or masking audits to detect spurious context reliance
7. Small-class protection
   - minimum-support policy and uncertainty quarantine buckets

## 18. QA Approval Checklist

Approve a run only if all are true:

- source snapshot frozen
- manifests/taxonomies validated
- Stage 1 passes metric and audit gates
- mining passes pseudo-label precision gates
- Stage 2 passes per-species and global gates
- direct breed YOLO comparison completed
- all artifacts saved with reproducible metadata
- visible terminal logs confirm AMD GPU execution
