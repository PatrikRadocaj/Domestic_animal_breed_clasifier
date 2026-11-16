# Strojno Učenje — Priprema za Usmeni Ispit
## Projekt: Animal Breed Detection (Two-Stage YOLO11 + EfficientNetV2)

---

## PROJEKT — 5-minutna prezentacija

**Cilj projekta:**
Izgraditi sustav koji na fotografiji automatski detektira životinju i klasificira njenu **pasminu** (breed).
Primjer: kamera snima farmu → sustav identificira "ovo je *Holstein Friesian* krava" ili "ovo je *Labrador Retriever* pas".

**Dva pristupa koja smo usporedili:**

```
[A] Dvostupanjski pipeline:
   Slika → YOLO11 detektor vrsta → BBox + {cat/dog/cattle/...}
               ↓ crop životinje (+ 10% kontekstni padding)
          EfficientNetV2-S klasifikator pasmina → "dog:labrador_retriever"

[B] Jednostupanjski pipeline:
   Slika → YOLO11 detektor s klasama na razini pasmina → direktno breed
   + curriculum training (A→B→C faze)
```

**Podaci:** Više javnih dataseta (Roboflow, OID, Stanford Dogs, ...) spojenih kroz
`CrossDatasetLabelMap` koji normalizira labele (`COW → cattle`, `Holstein-Friesian breed → cattle:holstein_friesian`).
Sve meta-informacije pohranjene u **manifest CSV** i **taxonomy YAML**.

**Treniranje:**
- YOLO11 treniran Ultralytics frameworkom (PyTorch)
- EfficientNetV2-S iz `timm` biblioteke, **pretreniran na ImageNet-21k + IN1k fine-tune** (transfer learning)
- Optimizator: **AdamW**, LR scheduler: **CosineAnnealingLR**, regularizacija: **L2 weight decay + label smoothing**
- GPU training s **AMP (Automatic Mixed Precision)** na AMD ROCm

**Evaluacija (dva modela se uspoređuju):**

| Metrika | Gdje |
|---------|------|
| mAP@50, mAP@50-95 | YOLO detektor |
| Top-1 accuracy, Top-5 accuracy | EfficientNetV2 klasifikator |
| Macro F1 | klasifikator po pasminama |
| ECE (Expected Calibration Error) | kalibriranost probabiliteta |
| Brier Score | probabilistička točnost |
| Latency p50/p95/p99 (ms) | cijeli pipeline |

**Bonus — mining neoznačenih slika (SSL element):**
Treniranim YOLO detektorom skeniramo sve neoznačene slike → generiramo pseudo-kandidate
→ ručna verifikacija u web aplikaciji → verificirani primjeri ulaze u sljedeći trening krug.

---

## PITANJE 1 — Što je strojno učenje? Tipovi i redoslijed radnji.

**Strojno učenje** je grana AI gdje algoritmi automatski uče iz podataka umjesto da budu eksplicitno programirani pravilima. Model optimizira parametre na temelju primjera kako bi naučio preslikavanje ulaz → izlaz.

### Tri tipa:

| Tip | Ulaz | Izlaz | Primjer |
|-----|------|-------|---------|
| **Nadzirano (Supervised)** | x + oznaka y | Naučena f(x)→y | Klasifikacija e-pošte: ulaz=tekst, izlaz=spam/nije spam |
| **Nenadzirano (Unsupervised)** | samo x | Skrivena struktura | K-means grupiranje korisnika po ponašanju |
| **Podržano (Reinforcement)** | stanje + nagrada | Politika akcija | Robot koji uči hodati; nagrada=napredak prema cilju |

### Tipičan redoslijed radnji:
1. Definicija problema i prikupljanje podataka
2. Označavanje (anotacija) podataka
3. Predobrada: čišćenje, normalizacija, podjela train/val/test
4. Odabir modela i arhitekture
5. Treniranje (forward pass → loss → backward → update)
6. Evaluacija na validacijskom skupu, podešavanje hiperparametara
7. Finalna evaluacija na testnom skupu
8. Deployment

---

### 🔵 U projektu (opširno):

Projekt koristi **nadzirano učenje** na dva mjesta:

**1) Detektor vrste (YOLO11) — detekcija objekata:**
- Svaka slika ima oznake u YOLO formatu: `class_id cx cy w h` (normalizirano)
- Model uči prepoznati bounding box i vrstu životinje (cat, dog, cattle, chicken, horse, ...)
- Ovo je nadzirano učenje jer svaka anotacija ima točnu klasu i koordinate boxa
- Labele dolaze iz više dataseta i normaliziraju se kroz `CrossDatasetLabelMap`

**2) Klasifikator pasmina (EfficientNetV2-S) — klasifikacija slika:**
- Ulaz: izrezana slika životinje (crop iz detekcije)
- Oznaka: `breed_id` u formatu `"species:breed_slug"` (npr. `"dog:labrador_retriever"`)
- Model uči razlikovati pasmina unutar iste vrste (npr. 15+ pasmina psa)
- Hijerarhijska arhitektura: **Stage 1** = klasifikator vrste, **Stage 2** = poseban klasifikator po vrsti

**3) Mining neoznačenih slika — element polu-nadziranog učenja:**
- Treniranim detektorom skeniramo slike koje **nemaju breed anotacije** (samo detekcijski dataseti)
- YOLO generira pseudo-predikcije s confidence ocjenama
- Implementirano u `src/animal_breed/mining/unlabeled_animals.py`
- Visoko-pouzdane predikcije → `unlabeled_candidates_high_priority.csv`
- Verifikacija kroz `animal-review-boxes` web app → potvrđene oznake ulaze u naredni trening
- Ovo je točno **samo-učenje (self-training)** — najjednostavniji SSL pristup

**Redoslijed radnji u projektu:**
```
1. animal-build-manifest     → gradi manifest_v0.csv iz raw dataseta
2. animal-validate-manifest  → provjerava shemu, duplicates, null vrijednosti
3. animal-audit-dataset      → statistike klasne ravnoteže, hijerarhije
4. animal-data-qa            → quality gate (bbox bounds, near-duplicates, split leakage)
5. animal-split-data         → stratificirana podjela train/val/test
6. animal-prepare-yolo       → generira YOLO data.yaml i slike za detekciju
7. animal-train-detector     → trenira YOLO11 detektor vrsta
8. animal-train-hierarchical-classifier → trenira EfficientNetV2 hijerarhiju
9. animal-eval               → evaluacija: mAP + Top-1/5 + F1 + ECE + latency
10. animal-mine-unlabeled-animals → SSL mining loop
11. animal-export-models     → pakirane modele za deployment
```

---

## PITANJE 2 — Tablična organizacija, predobrada, matrična notacija, slike/tekst/zvuk

**Tablica podataka:** Svaki **redak** = jedan primjer, svaki **stupac** = jedna značajka (feature).

**Matrična notacija:**
- **X** ∈ ℝ^(N×D) — N primjera, D dimenzija
- **y** ∈ ℝ^N — vektor oznaka
- x^(i) = i-ti primjer (redak), x_j = j-ta značajka (stupac)

### Predobrada — glavni koraci:
1. **Čišćenje** — uklanjanje duplikata, pogrešnih anotacija, outliera
2. **Normalizacija/standardizacija** — pikselne vrijednosti [0,255] → [0,1], ImageNet μ/σ
3. **Popunjavanje nedostajućih vrijednosti** — median, mean, mode
4. **Enkodiranje kategoričkih varijabli** — one-hot, label encoding
5. **Augmentacija** — samo na train skupu (flip, crop, color jitter, ...)
6. **Podjela train/val/test** — stratificirana po klasi

### Slike:
- Tensor H×W×C; normalizacija po ImageNet mean/std = [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
- Augmentacija: RandomResizedCrop, HorizontalFlip, ColorJitter, blur, CoarseDropout
- Na inferenci: samo resize → center crop → normalize

### Tekst: tokenizacija → embedding (TF-IDF, BERT)
### Zvuk: Mel-spektrogram / MFCC → 2D matrica → CNN

---

### 🔵 U projektu (opširno):

**Manifest kao tablica podataka:**
Manifest CSV je centralna "tablica" projekta. Svaki redak = jedna slika (ili jedan bounding box).

Stupci manifesta:
```
path, source_dataset, species_id, breed_id, split,
bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, width, height,
split_group, duplicate_cluster
```

Čita se u `pd.DataFrame` i prenosi svuda kroz kod (`read_manifest` funkcija).

**Label normalizacija (CrossDatasetLabelMap):**
Problem: 5+ dataseta, svaki ima drugačije nazive za iste životinje.
Rješenje: `configs/mappings/cross_dataset_label_map.yaml` + `CrossDatasetLabelMap` klasa u `data/label_mapping.py`.

Primjeri normalizacije:
```
"COW"                              → "cattle"
"Holstein-Friesian breed"          → "cattle:holstein_friesian"
"peacock" (farm-animals dataset)   → "peafowl"
"cpw" (farm-animals dataset)       → "manual_review"
```

4 koraka normalizacije (primijenjena u `animal-build-manifest`):
1. Label cleanup (lowercase, trim, separator normalization)
2. Canonical alias resolution
3. Dataset-specific overrides
4. Taxonomy resolution

**Taxonomy YAML** (`configs/taxonomy/taxonomy_v0.yaml`):
Definira hijerarhiju: Vrsta → Pasmina. Format ID: `"species_id:breed_slug"`.
```yaml
species:
  - id: dog
    name: Dog
    aliases: [canis lupus, domestic dog]
breeds:
  dog:
    - id: dog:labrador_retriever
      name: Labrador Retriever
      aliases: [labrador, lab]
```

Taxonomy klasa (`data/taxonomy.py`) validira:
- Svaki breed_id mora biti oblika `"species_id:breed_slug"`
- Breed ne smije referencirati nepostojeću vrstu

**Podjela podataka — stratificirana (`data/splitting.py`):**
- `train_test_split` sa `stratify` po `breed_id` (ili fallback na `species_id`)
- Grupiranje po `split_group` osigurava da sve slike iste životinje idu u isti split (nema leakage)
- Tipična podjela: 70% train, 15% val, 15% test

**Quality gate (`data/quality/qa.py`):**
Automatski provjeri 6 vrsta problema:
1. `bbox_bounds` — koordinate izvan dimenzija slike (error)
2. `bbox_aspect` — abnormalni aspect ratio > 12:1 (warning)
3. `taxonomy_hierarchy` — breed_id i species_id ne odgovaraju taksonomiji (error)
4. `split_leakage` — isti duplicate_cluster u različitim splitovima (error)
5. `exact_duplicates` — SHA256 hash duplikati (warning)
6. `near_duplicates` — perceptualni hash (ahash 8×8) duplikati (warning)
Treniranje se ne može pokrenuti ako QA gate ne prođe (`if not qa_report["valid"]: raise`).

**Augmentacija slika (`data/transforms.py`):**

*Train transforms (albumentations):*
```python
A.RandomResizedCrop(size=(384,384), scale=(0.65, 1.0), ratio=(0.75, 1.33))
A.HorizontalFlip(p=0.5)
A.ColorJitter(p=0.4)
A.OneOf([MotionBlur, GaussianBlur, ImageCompression], p=0.2)
A.CoarseDropout(holes=1-2, size=5-18% slike, p=0.25)  # "Cutout" regularizacija
A.Normalize()  # ImageNet mean/std
ToTensorV2()
```

*Eval transforms (bez augmentacije):*
```python
A.LongestMaxSize(max_size=384)
A.PadIfNeeded(min_height=384, min_width=384)  # zero-padding za maintain aspect ratio
A.CenterCrop(384, 384)
A.Normalize()
ToTensorV2()
```

**Crop kontekst:** U inferenci, iz YOLO bboxa uzimamo crop s **10-12% paddinga** s svake strane:
```python
px = bw * pad_ratio  # pad_ratio = 0.10
py = bh * pad_ratio
crop = image[y0-py : y1+py, x0-px : x1+px]
```
Razlog: finim razlikama pasmina (dlaka, oblik glave) potrebbe je malo više konteksta od samog tijesnog boxa.

---

## PITANJE 3 — Jednostavna linearna regresija

**Model**: ŷ = w·x + b  
**Pretpostavke**: linearna veza, normalno distribuirani reziduali, homoskedastičnost, neovisnost primjera

**Kriterijska funkcija (MSE)**:  
J(w,b) = (1/N) Σ (y_i - ŷ_i)²

**Gradijentni spust:**
- ∂J/∂w = -(2/N) Σ x_i(y_i - ŷ_i)
- ∂J/∂b = -(2/N) Σ (y_i - ŷ_i)
- Update: w ← w - α·∂J/∂w; b ← b - α·∂J/∂b

**Scenariji prema learning rate α:**
- Premalen → konvergira ali presporo
- Optimalan → brza konvergencija
- Prevelik → divergencija ili oscilacije

**Višedimenzionalni podaci**: ŷ = w^T·x + b, normalne jednadžbe: w = (X^TX)^{-1}X^Ty

---

### 🔵 U projektu:
Linearna regresija nije direktno korištena, ali njezin temelj — **gradijentni spust i learning rate** — je osnova svakog optimizatora koji koristimo.

U našem projektu specifično:
- **AdamW** je adaptivna verzija gradijentnog spusta: umjesto fiksnog α, svaki parametar ima vlastitu efektivnu stopu učenja računatu iz prvih i drugih momenata gradijenata.
- Problemi s prevelikim learning rateom riješeni su **CosineAnnealingLR** schedulerom koji postupno smanjuje lr od početne vrijednosti prema skoro-nuli kroz `T_max=epochs`.
- Kod YOLO treniranja: `lr0` (initial lr) + `cos_lr=True` (cosine schedule) + `patience` za early stopping.

---

## PITANJE 4 — Logistička regresija

**Model**: P(y=1|x) = σ(w^Tx + b) = 1/(1+e^{-z})  
**Kriterijska funkcija (Binary Cross-Entropy)**:  
J = -(1/N) Σ [y_i·log(ŷ_i) + (1-y_i)·log(1-ŷ_i)]

**Višeklasna klasifikacija:**
- **One-vs-Rest (OvR)**: K binarnih klasifikatora
- **Softmax**: P(k|x) = e^{z_k} / Σ e^{z_j} → Categorical Cross-Entropy

---

### 🔵 U projektu:
EfficientNetV2-S na izlazu ima **softmax višeklasnu klasifikaciju**:

```python
# U classifier_trainer.py:
criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.model.classifier.train.label_smoothing))
```

`nn.CrossEntropyLoss` u PyTorchu interno primjenjuje **log-softmax** + **negative log likelihood**, što je identično Categorical Cross-Entropy.

**Label smoothing** kao modifikacija standardne CE:
Umjesto "hard" targeta [0, 0, 1, 0, ...], koristimo "soft" targete:
- Ispravna klasa: `1 - ε` (npr. 0.9)
- Ostale klase: `ε / (K-1)` (npr. 0.1 / 14)

Korist: model nije pretjerano siguran u krive klase, bolje generalizira.

**Klasifikacijski head EfficientNetV2:**
```
GlobalAveragePooling → Dropout → Linear(in_features, num_classes) → softmax
```

Za Stage 1 (species classifier): `num_classes = broj vrsta` (npr. 6)
Za Stage 2 (breed classifier): poseban model po vrsti, `num_classes = broj pasmina te vrste`

---

## PITANJE 5 — Evaluacija modela

### Regresijske metrike:
- **MAE** = (1/N) Σ |y_i - ŷ_i|
- **MSE** = (1/N) Σ (y_i - ŷ_i)²
- **RMSE** = √MSE
- **R²** = 1 - SS_res/SS_tot

### Klasifikacijske metrike (binarne):
- **Accuracy** = (TP+TN)/(TP+TN+FP+FN)
- **Precision** = TP/(TP+FP)
- **Recall** = TP/(TP+FN)
- **F1** = 2·P·R/(P+R)
- **AUC-ROC**: površina ispod ROC krivulje

### Višeklasne klasifikacijske metrike:
- **Macro F1** = (1/K) Σ F1_k — jednaka težina svakoj klasi
- **Weighted F1** = Σ (n_k/N)·F1_k
- **Top-k accuracy**: točan odgovor unutar top-k predikcija
- **Confusion matrix**: N×N matrica

---

### 🔵 U projektu (opširno):

Evaluacija je implementirana u `src/animal_breed/evaluation/metrics.py` i `evaluation/runner.py`.

**1) Top-k Accuracy (`topk_accuracy`):**
```python
def topk_accuracy(y_true, y_topk, k=1):
    hits = sum(1 for target, preds in zip(y_true, y_topk) if target in preds[:k])
    return hits / len(y_true)
```
- **Top-1**: mora biti točno na prvom mjestu (standardna accuracy)
- **Top-5**: točna klasa mora biti u prvih 5 predikcija
- Za pasminu psa s 30+ klasa, Top-5 je realističnija mjera korisnosti

**2) Macro F1 (`macro_f1`):**
```python
f1_score(y_true, y_pred, average="macro", zero_division=0)
```
Zašto **macro** a ne weighted? Kod pasmina, neke pasmine imaju malo slika (long-tail problem).
Macro F1 daje jednaku važnost svakoj pasmini → kažnjava ako model zanemaruje rijetke pasmine.

**3) Brier Score (`brier_score_multiclass`):**
```python
y_onehot = np.zeros((N, num_classes))
y_onehot[np.arange(N), y_true] = 1.0
brier = np.mean(np.sum((probs - y_onehot)**2, axis=1))
```
Mjeri kvalitetu **probabiliteta** (ne samo klase). Brier=0 → savršene vjerojatnosti.
Važno kod pasmina jer veterinar želi znati koliko je model siguran, ne samo predikciju.

**4) Expected Calibration Error (`expected_calibration_error`):**
```python
# Dijeli confidence u 15 binova [0, 1/15, 2/15, ...]
# Za svaki bin: |mean_accuracy - mean_confidence|, ponderirano veličinom bina
```
Kalibriranost = je li model "90% siguran" zaista točan u 90% slučajeva?
Nekalibriran model (overconfident) daje visoku confidence na krive predikcije → loše za production.

**5) Latency mjerenje (`summarize_latency`):**
```python
{"p50_ms": ..., "p95_ms": ..., "p99_ms": ..., "mean_ms": ...}
```
Mjerimo **percentile** jer mean može biti zavaravajući (sporo za outlierse).
p95_ms = 95% zahtjeva završi unutar tog vremena → SLA metrika.

**6) Promotion gate (`relative_promotion_gate`):**
```python
quality_improved = current_top1 > baseline_top1
latency_ok = current_p95 <= baseline_p95 * (1 + max_latency_regression_pct / 100)
gate_passed = quality_improved AND latency_ok
```
Automatski odlučuje smije li novi model zamijeniti baseline.
Sprječava scenarij "model je točniji ali 3× sporiji".

**7) mAP za YOLO detektor:**
```python
metrics = model.val(data=data_yaml, verbose=False)
map50 = metrics.box.map50   # mAP@IoU=0.5
map   = metrics.box.map     # mAP@IoU=0.5:0.95
```
YOLO evaluacija je standardna COCO metrika.

**Ablation matrix (usporedba pristupa):**
```
pipeline        primary_metric  p95_ms  notes
single_stage    mAP             X ms    YOLO only
two_stage       breed_top1      Y ms    YOLO + hierarchical classifier
```

---

## PITANJE 6 — Struktura neurona, aktivacijske funkcije, numerički primjer

### Struktura neurona:
```
x₁ →─w₁─┐
x₂ →─w₂─┤   z = Σ wᵢxᵢ + b  →  a = f(z)  →  izlaz
xD →─wD─┘
```

### 4 aktivacijske funkcije:

| Funkcija | Formula | Prednosti | Nedostaci |
|----------|---------|-----------|-----------|
| **Sigmoid** | 1/(1+e^{-z}) | izlaz [0,1], interpretabilna | vanishing gradient, sporija |
| **ReLU** | max(0,z) | brza, rješava vanishing gradient | "dying ReLU" — mrtvi neuroni za z<0 |
| **Tanh** | (e^z-e^{-z})/(e^z+e^{-z}) | izlaz [-1,1], zero-centered | i dalje vanishing gradient |
| **Softmax** | e^{zk}/Σe^{zj} | vjerojatnosna distribucija, Σ=1 | numerički nestabilna bez log-sum-exp tricka |

### Numerički primjer:
x=2, y=1, w=0.5, b=0, α=0.1, sigmoid, BCE
- **Forward**: z = 0.5·2 + 0 = 1.0, ŷ = σ(1.0) = 0.731
- **Loss**: L = -log(0.731) ≈ 0.313
- **Backward**: ∂L/∂z = ŷ-y = -0.269; ∂L/∂w = -0.269·2 = -0.538; ∂L/∂b = -0.269
- **Update**: w_new = 0.5 - 0.1·(-0.538) = **0.554**; b_new = 0 - 0.1·(-0.269) = **0.027**
- **Smanjenje gubitka**: novi L ≈ 0.278 → smanjenje ≈ **0.035**

---

### 🔵 U projektu:
EfficientNetV2 koristi **SiLU** (Swish) aktivacijsku funkciju u svim MBConv blokovima:

SiLU(z) = z · σ(z) = z / (1 + e^{-z})

Zašto SiLU umjesto ReLU?
- Glatka je (nema oštre granice na 0)
- Negativne vrijednosti nisu uvijek 0 (za razliku od ReLU) → bolji gradijenti
- EfficientNet paper pokazuje poboljšanje performansi u odnosu na ReLU

Izlazni sloj klasifikatora: **Softmax** (implicitan u `nn.CrossEntropyLoss`) za višeklasnu klasifikaciju pasmina.

---

## PITANJE 7 — Artificijelna neuronska mreža (feedforward, potpuno povezana)

**FC mreža (fully connected):** Svaki neuron spojen sa svim neuronima susjednog sloja. Nema recikliranja signala (feedforward).

### Skica (3 ulaza, 4 u skrivenom, 1 izlaz):
```
x₁ ─┐                    ┌─ h₁ ─┐
x₂ ─┼──[W¹ 4×3, b¹ 4×1]─┼─ h₂ ─┼──[W² 1×4, b² 1×1]──→ ŷ
x₃ ─┘                    ├─ h₃ ─┘
                          └─ h₄
```

**Broj parametara:**
- Sloj 1: 3×4 + 4 = **16**
- Sloj 2: 4×1 + 1 = **5**
- **Ukupno: 21**

**Forward propagacija:**
- z¹ = W¹·x + b¹  (4×1)
- a¹ = ReLU(z¹)
- z² = W²·a¹ + b²  (1×1)
- ŷ = σ(z²) ili softmax(z²)

**Prednosti FC:** Opće primjenjiva, laka za implementaciju  
**Nedostaci FC:** Ne iskorištava prostornu strukturu slika, puno parametara (eksplodira za slike), osjetljiva na overfitting

---

### 🔵 U projektu:
EfficientNetV2 **nije** čista FC mreža — ona je konvolucijska. Ali na samom kraju mreže postoji **FC glava (head)**:

```
[Convolutional Backbone EfficientNetV2]
         ↓
GlobalAveragePooling2D
  (1280-dimenzionalni vektor)
         ↓
Dropout(p=0.2)
         ↓
Linear(1280 → num_classes)   ← ovo je FC sloj
         ↓
[softmax]
```

Klasifikacijska glava je potpuno povezan sloj koji mapu značajki iz CNN-a (1280 dimenzija) pretvara u vjerojatnosti klasa. Broj parametara ovog FC sloja:
- Za species classifier (6 vrsta): 1280×6 + 6 = **7.686**
- Za dog breed classifier (15 pasmina): 1280×15 + 15 = **19.215**

Ovo je primjer **feature extraction** iz pretreniranog CNN-a → FC head koji se trenira na novoj zadaći.

---

## PITANJE 8 — Izgradnja neuronske mreže, hiperparametri, praćenje napretka

### Koraci izgradnje:
1. Definicija arhitekture (broj slojeva, neurona, tip)
2. Inicijalizacija parametara (Xavier, He, pretrained weights)
3. Odabir loss funkcije i optimizatora
4. Treniranje po epohama: forward → loss → backward → update
5. Praćenje na val skupu, podešavanje hiperparametara
6. Early stopping ili scheduler

### Hiperparametri:
| Hiperparametar | Opis |
|---------------|------|
| Learning rate | veličina koraka; AdamW ga adaptivno skalira |
| Batch size | broj primjera po ažuriranju |
| Broj epoha | prolazi kroz cijeli dataset |
| Arhitektura | slojevi, neuroni, aktivacije |
| Weight decay | L2 regularizacija u AdamW |
| Label smoothing | blaža regularizacija targeta |
| Dropout | nasumično isključivanje neurona |
| LR scheduler | kako lr pada kroz treniranje |
| Early stopping patience | epohe bez poboljšanja do zaustavljanja |

### Grafički prikazi napretka:
```
Loss krivulje:            Accuracy krivulje:
│                         │         val_top1
│ train_loss\             │      ___/
│             \___val     │  ___/ train_top1
│                         │_/
└──── epohe               └──────── epohe

Overfitting signal: val_loss počne rasti dok train_loss pada
```

---

### 🔵 U projektu (opširno):

Svaki aspekt izgradnje mreže je eksplicitno konfiguriran u YAML filovima i implementiran u `classifier_trainer.py`.

**Arhitektura:**
```python
model = timm.create_model(
    "tf_efficientnetv2_s.in21k_ft_in1k",  # pretreniran na ImageNet-21k
    pretrained=True,                        # učitava pretrenirne težine
    num_classes=len(label_to_index),        # adaptira glavu za naš broj klasa
).to(device)
```

**Optimizator — AdamW:**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(cfg.model.classifier.train.lr),
    weight_decay=float(cfg.model.classifier.train.weight_decay),
)
```
AdamW = Adam + decoupled weight decay. Razlika od standardnog Adam + L2: u AdamW, weight decay se primjenjuje direktno na parametre (ne kroz gradijent), što je matematički ispravnije.

**LR Scheduler — CosineAnnealingLR:**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=int(cfg.model.classifier.train.epochs)
)
```
lr pada po kosinusnom rasporedu: visok na početku (brzo učenje) → skoro-nula na kraju (fino ugađanje).
Zašto kosinusni a ne step decay? Glatkiji pad, model ne "šokira" naglim padom lr-a.

**AMP — Automatic Mixed Precision:**
```python
scaler = torch.cuda.amp.GradScaler(enabled=True)
with torch.autocast(device_type="cuda", enabled=True):
    logits = model(images)
    loss = criterion(logits, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
Forward/backward u float16 (pola memorije, ~2× brže na GPU), update u float32.
`GradScaler` skalira loss da spriječi underflow u float16 gradijentima.

**Early stopping:**
```python
if val_metrics["top1"] > best_top1 + early_stopping_min_delta:
    best_top1 = val_metrics["top1"]
    epochs_without_improvement = 0
    _save_checkpoint(bundle, "best.pt", checkpoint)
else:
    epochs_without_improvement += 1
    if epochs_without_improvement >= early_stopping_patience:
        break
```
Prati `val_top1`. Ako nema poboljšanja `patience` epoha, treniranje staje.
Sprema se `best.pt` (checkpoint s najboljim val_top1) i `last.pt` (zadnja epoha).

**Praćenje napretka:**
- Svaka epoha logira: `train_loss, train_top1, val_loss, val_top1, val_top5, seconds`
- Sprema se u `artifacts/<run_id>/metrics/classifier_epoch_history.csv` i `.json`
- Real-time progress JSON: `artifacts/<run_id>/progress/status.json` (svaki batch)
```json
{
  "species_id": "dog",
  "phase": "train",
  "epoch_current": 5,
  "epoch_total": 30,
  "batch_current": 120,
  "batch_total": 450,
  "train_loss_latest": 0.4231,
  "device": "cuda:0",
  "last_update_time": "2026-06-10T14:23:11"
}
```

**Reproducibilnost:**
```python
seed_everything(int(cfg.seed), deterministic=True)
# + snapshot hashes manifest/taxonomy/source_registry
```
Svaki run sprema SHA256 hash dataseta i konfiga → audit trail.

---

## PITANJE 9 — Konvolucijske neuronske mreže (CNN)

### Struktura CNN-a:
1. **Konvolucijski slojevi** — lokalna ekstrakcija značajki s filterima
2. **Aktivacijske funkcije** — ReLU/SiLU iza svake konvolucije
3. **Pooling slojevi** — smanjenje prostornih dimenzija
4. **FC slojevi** — klasifikacija na kraju

### Konvolucijski sloj:
- Filter K×K×C_in → C_out filtera
- **Stride s**: korak klizanja (default 1, pri downsamplingu 2)
- **Padding p**: SAME=zadržava dimenzije, VALID=nema paddinga
- Izlaz: H_out = (H_in - K + 2P)/S + 1
- **Broj parametara**: C_out × (K×K×C_in + 1)

**Primjer:** Ulaz 32×32×3, 16 filtera 3×3, S=1, P=1  
→ Izlaz 32×32×16  
→ Parametri: 16 × (3×3×3 + 1) = **448**

### Pooling:
- **Max pooling**: max vrijednost u receptivnom polju → invarijantnost na male pomake
- **Average pooling**: prosjek
- **Global Average Pooling (GAP)**: svaki kanal → jedan broj (H×W→1×1)
- Nema parametara koji se uče

---

### 🔵 U projektu (opširno):

Koristimo **EfficientNetV2-S** — jedna od najefikasnijih CNN arhitektura za klasifikaciju slika.

**Arhitektura EfficientNetV2-S:**
```
Input 384×384×3
    ↓
[Stem: Conv2d 3×3, stride=2] → 192×192×24
    ↓
[Stage 1: Fused-MBConv × 2, E=1, k=3] → 192×192×24
    ↓
[Stage 2: Fused-MBConv × 4, E=4, k=3] → 96×96×48
    ↓
[Stage 3: Fused-MBConv × 4, E=4, k=3] → 48×48×64
    ↓
[Stage 4: MBConv × 6, E=4, k=3, SE] → 24×24×128
    ↓
[Stage 5: MBConv × 9, E=6, k=3, SE] → 12×12×160
    ↓
[Stage 6: MBConv × 15, E=6, k=3, SE] → 6×6×256
    ↓
[Head Conv 1×1] → 6×6×1280
    ↓
GlobalAveragePooling → 1280
    ↓
Dropout(0.2)
    ↓
Linear(1280 → num_classes)
```

**Fused-MBConv** (u ranim slojevima): Conv2d(stride) → BN → Activation → Conv1×1 → BN  
**MBConv** (u kasnijim slojevima): Depthwise Conv → BN → Activation → SE → Pointwise Conv  
**SE (Squeeze-and-Excitation)**: uči koje kanale naglasiti → kanal-pažnja (channel attention)

**Zašto EfficientNetV2 a ne ResNet50, VGG16?**
- EfficientNet skalira width/depth/resolution uniformno (compound scaling)
- V2 dodaje Fused-MBConv koji je brži za treniranje (manje memorije)
- ~20M parametara vs ~25M ResNet50, ali bolja točnost
- Pretreniran na IN-21k (21.000 klasa, 14M slika) → iznimno bogata reprezentacija

**Kod Convolucijiskog sloja u kontekstu detekcije (YOLO11):**
YOLO backbone (C3k2 blokovi, SPPF) → feature mape različitih rezolucija → FPN neck → Detection head

Primjer dimenzija YOLO feature mapa za 640×640 input:
- P3: 80×80×256 (detekcija malih objekata)
- P4: 40×40×512 (srednji objekti)
- P5: 20×20×512 (veliki objekti)

Svaka razina P predviđa: `(num_anchors × (5 + num_classes))` vrijednosti po ćeliji.

---

## PITANJE 10 — Treniranje: optimizatori, overfitting, batch norm, transfer learning

### Optimizatori:

| Optimizator | Karakteristike |
|-------------|---------------|
| **SGD** | Klasičan, jednostavan; treba pažljiv lr i momentum |
| **SGD+Momentum** | Akumulira smjer gradijenta → brža konvergencija |
| **Adam** | Adaptivni lr po parametru, m1+m2 momenti |
| **AdamW** | Adam + decoupled weight decay (bolje od Adam+L2) |
| **RMSprop** | Adaptivna lr, dobar za RNN |

### Sprječavanje overfittinga:
- **Rano zaustavljanje**: zaustavi treniranje kad val_loss prestane padati
- **L2 regularizacija**: kažnjava velike težine, J += λ·Σw²
- **Dropout**: nasumično isključivanje neurona, p=0.2–0.5
- **Augmentacija**: proširuje efektivnu veličinu train skupa
- **Label smoothing**: blažiji targeti, model nije pretjerano siguran

### Batch normalizacija:
- Normalizira aktivacije po batch-u: z_norm = (z - μ_batch) / σ_batch
- Trenira γ (scale) i β (shift)
- Stabilizira gradijente, dopušta veće lr, smanjuje osjetljivost na inicijalizaciju
- Na inferenci koristi running mean/var (ne batch)

### Transfer learning:
1. Uzmi pretreniran model (ImageNet backbone)
2. Zamijeni izlazni sloj za novu zadaću
3. **Fine-tuning**: trenira se cijela mreža (duže, ali bolje)
4. **Feature extraction**: zamrzni backbone, trenira samo head (brzo, manje podataka)

---

### 🔵 U projektu (opširno):

**Transfer learning — ključna odluka projekta:**

`timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=N)`

`in21k_ft_in1k` znači: najprije treniran na **ImageNet-21k** (21.000 klasa, 14M slika), 
zatim fine-tuned na **ImageNet-1k** (1.000 klasa, 1.2M slika).

**Zašto je ovo kritično za projekt:**
- Naš dataset pasmina ima relativno malo slika po klasi (long-tail problem)
- Pretreniran model već "razumije" dlaku, oblike životinja, teksture — sve što je relevantno
- Bez transfer learninga bi trebali 10-50× više podataka za sličnu točnost
- Uvrtanjem novog `num_classes` sloja inicijaliziramo samo posljednji sloj nasumično; backbone ostaje s ImageNet weights

**Što se točno dogodi pri fine-tuningu:**
- Gradient flow prolazi kroz cijelu mrežu (svi slojevi se ažuriraju)
- Niski lr (npr. 1e-4) → spore, stabilne promjene pretreniranih težina
- CosineAnnealingLR → lr postupno pada: backbone se minimalno mijenja pri kraju

**Batch normalizacija u EfficientNetV2:**
Svaki MBConv i Fused-MBConv blok sadrži BatchNorm2d. Bitna razlika:
- Tijekom `model.train()`: statistike računaju se iz trenutnog batcha
- Tijekom `model.eval()`: koriste se running mean/var akumulirani kroz treniranje
```python
model.train(is_train)  # is_train=True za train fazu, False za val
```

**Augmentacija kao regularizacija (transforms.py):**
- `RandomResizedCrop(scale=(0.65, 1.0))` → model ne overfita na točnu veličinu životinje
- `HorizontalFlip(p=0.5)` → invarijantnost na lijevu/desnu orijentaciju
- `ColorJitter(p=0.4)` → invarijantnost na osvjetljenje i boju
- `CoarseDropout` → simulira okluziju (životinja djelomično zakrivena)
- `MotionBlur/GaussianBlur` → robustnost na zamagljenje

**Label smoothing u CrossEntropyLoss:**
```python
nn.CrossEntropyLoss(label_smoothing=0.1)
```
Umjesto one-hot [0,0,1,0,...] koristi [ε/K, ..., 1-ε, ..., ε/K].
Sprječava model da "uči napamet" i bude previše siguran → bolja generalizacija.

**AdamW vs Adam — zašto nam je važna razlika:**
U standardnom Adam + L2 reg, weight decay interagira s momentima adaptivnog lr-a i efektivno ga smanjuje za parameter sa samo veliku gradijentnu magnitudu. AdamW odvaja weight decay od gradijentnog ažuriranja → konzistentniji regularizacijski efekt. EfficientNet radovi potvrđuju da AdamW daje bolju točnost.

**Early stopping u praksi:**
```python
early_stopping_patience = 10   # epohe bez poboljšanja
early_stopping_min_delta = 0.001  # minimalno poboljšanje koje se broji
```
Ako val_top1 ne poraste za >0.1% unutar 10 epoha → stop.
Čuva `best.pt` (model u trenutku najveće val_top1) za finalnu evaluaciju.

**YOLO specifični hiperparametri:**
```python
model.train(
    imgsz=640,       # veličina ulazne slike
    epochs=100,      # maksimalan broj epoha
    batch=16,        # batch size
    patience=50,     # early stopping za YOLO
    optimizer="AdamW",
    lr0=0.001,       # initial learning rate
    cos_lr=True,     # cosine LR schedule
    close_mosaic=10, # zadnjih N epoha bez mosaic augmentacije
    amp=True,        # Automatic Mixed Precision
    cache=True,      # cache slika u RAM za brže treniranje
)
```
`close_mosaic`: mosaic augmentacija (spajanje 4 slike) se isključuje zadnjih 10 epoha → model se stabilizira na "normalnim" slikama.

**Curriculum training (YOLO, `yolo_trainer.py`):**
Troetapno treniranje za jedinstveni YOLO:
- **Stage A**: YOLO11 trenira na detektiranju vrsta (lake klase)
- **Stage B**: Uzima Stage A best.pt i fine-tune na razini pasmina (teže)
- **Stage C**: Još jedan round s pojačanom augmentacijom za rijetke klase

---

## PITANJE 11 — Računalni vid: detekcija objekata, R-CNN, YOLO

### Temeljni zadaci računalnog vida:
- **Klasifikacija**: koja klasa je na slici?
- **Lokalizacija**: gdje je objekt + klasa?
- **Detekcija**: svi objekti + klase + bboxovi
- **Segmentacija** (instance/semantic): maska po pikselu

### Dva tipa detektora:

| Tip | Pristup | Prednosti | Nedostaci |
|-----|---------|-----------|-----------|
| **Two-stage** | 1) Region proposals; 2) Klasifikacija | Visoka točnost | Sporiji (2 forward passa) |
| **One-stage** | Direktno predviđanje u jednom prolasku | Brz (real-time) | Malo lošiji za male objekte |

### R-CNN → Fast R-CNN → Faster R-CNN:
- **R-CNN**: Selective Search (~2000 regija) → CNN za svaku → spor
- **Fast R-CNN**: Cijela slika kroz CNN → RoI Pooling na feature mapi → brže, end-to-end
- **Faster R-CNN**: Dodaje RPN (Region Proposal Network) → GPU proposals, end-to-end

### YOLO:
- Slika podijeljena na S×S grid
- Svaka ćelija predviđa B bounding boxova + klase
- **Anchor boxovi**: preddefiniranih oblika; predviđamo offsete od anchora (tx, ty, tw, th)
- **NMS**: od preklapajućih predikcija (IoU > threshold) zadržava onu s max confidence

---

### 🔵 U projektu (opširno):

**YOLO11 za detekciju vrsta — ključni detalji:**

Koristimo **Ultralytics YOLO11** (najnovija verzija YOLO ekosistema).
YOLO11 uvodi C3k2 blokove (poboljšanje C2k) i poboljšani FPN neck.

**Što YOLO detektor radi u dvostupanjskom pipelineu:**
1. Dobiva cijelu sliku (640×640 ili originalna rezolucija)
2. Detektira **vrstu** životinje (cat, dog, cattle, chicken, horse, ...)
3. Daje bounding box koordinate (x1,y1,x2,y2) + confidence + klasa
4. Konfig pragovi: `detector_conf_threshold`, `detector_iou_threshold`

```python
detector = YOLO("artifacts/shared_species_detector/weights/best.pt")
results = detector.predict(
    source=[str(p) for p in image_paths],
    conf=0.25,    # minimalni confidence za predikciju
    iou=0.45,     # NMS IoU threshold
    verbose=False
)
```

**Batched inference:** Svi image pathovi se šalju odjednom → YOLO interno batcha → efikasnije.

**Iz YOLO rezultata izvlačimo:**
```python
for i in range(len(boxes)):
    xyxy = boxes.xyxy[i].tolist()        # [x1, y1, x2, y2] u pikselima
    conf = float(boxes.conf[i].item())    # confidence score 0-1
    cls_idx = int(boxes.cls[i].item())    # indeks klase
    detector_species = result.names[cls_idx]  # "dog", "cat", ...
```

**Crop i padding (`_crop_from_xyxy`):**
```python
px = bw * pad_ratio  # pad_ratio = 0.10 (10% širine boxa)
py = bh * pad_ratio
crop = image[y0-py : y1+py, x0-px : x1+px]
```
Padding je neophodan jer EfficientNetV2 klasifikator treba vidjeti malo konteksta
(npr. oblik tijela koji pomaže pri razlikovanju dugodlake od kratkodlake pasmine).

**Arbitration mehanizam (species arbitration):**
```python
species_top2 = predict_topk(species_model, crop, topk=2)
species_pred, species_conf = species_top2[0]
# Ako se YOLO i classifier ne slažu, a razlika u conf je mala:
if species_pred != detector_species and (species_conf - species_top2[1][1]) < arbitration_gap:
    species_pred = detector_species  # koristi YOLO predikciju
```
Ovo je mini-ansambl između YOLO detektora i species classifiera!
`arbitration_gap = 0.15`: ako je klasifikator "neodlučan" (margin < 15%), vjerujemo YOLO-u.

**Breed caching:**
```python
breed_cache: dict[str, tuple[model, labels, img_size] | None] = {}
if species_pred not in breed_cache:
    breed_ckpt = breed_models_dir / species_pred / "best.pt"
    breed_cache[species_pred] = _load_checkpoint(breed_ckpt, device) if breed_ckpt.exists() else None
```
Breed modeli se učitavaju lazy (samo kad zatrebaju) i cachiraju → jedan model po vrsti, ne po slici.

**Hijerarhijska arhitektura klasifikatora (`train_hierarchical.py`):**
```
Stage 1: species_model (EfficientNetV2, num_classes=N_vrsta)
Stage 2: breed_models/{species_id}/best.pt (poseban po vrsti)
```
Treniraju se sekvencijalno: najprije species, onda breed-per-species.
Zašto hijerarhija? Razlikovanje pasmina istovrsnih životinja zahtijeva specijalizirani model —
psi i mačke imaju potpuno različite breed feature-e, spajanje bi zbunilo model.

**Hybrid/species-aware training (`hybrid_species_training.py`):**
Alternativni pristup: za svaku vrstu poseban YOLO detektor (ne species-agnostik).
Koristi `HybridSpeciesYoloDataset` koji dinamički sastavlja batch iz više izvora s definiranim sampling ratom.

---

## PITANJE 12 — Evaluacija detektora i segmentacijskih modela

### Evaluacija detektora:

**IoU (Intersection over Union):**
IoU = Area(Predicted ∩ GT) / Area(Predicted ∪ GT)
- IoU ≥ 0.5 → TP (ispravna detekcija)
- IoU < 0.5 → FP (kriva detekcija)

**Precision-Recall po confidence pragu:**
- Precision = TP/(TP+FP)
- Recall = TP/(TP+FN)

**AP (Average Precision)** = površina ispod PR krivulje  
**mAP@50** = mean AP svih klasa pri IoU=0.5  
**mAP@50-95** = mean AP za IoU ∈ {0.50, 0.55, ..., 0.95} → strožija COCO metrika

**Vizualni primjer:**
```
GT box:        [100,100,200,200]  → površina 10000
Predicted box: [120,120,220,220]  → površina 10000
Intersection:  [120,120,200,200]  → 80×80=6400
Union: 10000+10000-6400=13600
IoU = 6400/13600 ≈ 0.47 → FP (< 0.5)
```

### Evaluacija segmentacijskih modela:
- **Pixel Accuracy**: ispravno klasificirani pikseli / ukupni pikseli
- **IoU po klasi** (Jaccard): |A∩B| / |A∪B| za maske
- **mIoU** = mean IoU svih klasa

---

### 🔵 U projektu (opširno):

**Evaluacija detektora (`evaluation/runner.py`):**
```python
def _evaluate_detector_map(model_path, data_yaml):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, verbose=False)
    return {"map50": metrics.box.map50, "map": metrics.box.map}
```
Ultralytics automatski izračunava mAP@50 i mAP@50-95 prema COCO standardu.

Uspoređujemo:
- `single_stage_detector_weights` → direktni breed detektor (mAP)
- `two_stage_detector_weights` → species detektor (mAP) → + breed_top1 iz classifiera

**IoU u mining pipelineu (`mining/unlabeled_animals.py`):**
```python
def _iou_xyxy(a, b):
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)
```
Koristi se za **suzbijanje same-species rediscovery**: ako YOLO predvidi novu detekciju na istom mjestu kao postojeća oznaka iste vrste (IoU > threshold), ignoriraj je — nije "novo" otkriće.

**Promotion gate kao evaluacijski mehanizam:**
```python
gate = relative_promotion_gate(
    current={"primary_metric": breed_top1, "p95_ms": p95_latency},
    baseline=baseline_json,        # prethodni best model
    primary_metric_key="primary_metric",
    latency_key="p95_ms",
    max_latency_regression_pct=15.0  # max 15% latency regresija
)
```
Novi model mora biti **bolji u točnosti** I **nije značajno sporiji** od baseline-a.
Ovo je kvalitativna evalucijska metrika koja sprječava deployment lošijeg modela.

**Ablation tablica:**
```csv
pipeline,     primary_metric, p95_ms, notes
single_stage, mAP,            X ms,   YOLO breed only
two_stage,    breed_top1,     Y ms,   YOLO species + EfficientNetV2
```
Direktna usporedba dvaju pristupa pod istim uvjetima.

---

## PITANJE 13 — Nenadzirano učenje, grupiranje, K-means

**Nenadzirano učenje**: nema oznaka; model traži skrivenu strukturu u podacima.

**3 primjera:**
1. Grupiranje kupaca po kupovnim navikama (segmentacija korisnika)
2. Detekcija anomalija u mrežnom prometu (abnormalni pattern)
3. Kompresija slika (K-means kvantizacija boja: 256 boja → 16)

**Pristupi klasteriranju:**
- **K-means**: centroid-based, sferični klasteri, treba K unaprijed
- **Hierarchical**: dendrogram, ne treba K, ali skup (O(N²))
- **DBSCAN**: density-based, automatski K, robusno na outliere

**K-means algoritam:**
1. Inicijaliziraj K centroida (nasumično ili K-means++)
2. E-korak: dodjeli svaki primjer najbližem centroidu (Euclidska udaljenost)
3. M-korak: ažuriraj centroide = mean dodijeljenih točaka
4. Ponovi dok centroidi ne konvergiraju

**Inicijalizacija:**
- Nasumična: nestabilna, lokalni optimum
- **K-means++**: proporcionalna slučajna inicijalizacija (dalji centroidi vjerojatniji)

**Optimalni K — Elbow metoda:**
Crtaj inertia (WCSS = Σ min udaljenost²) vs K → odaberi "koljeno" (knee point)

---

### 🔵 U projektu:
Nenadzirano učenje **nije direktno korišteno** u treniranju, ali se pojavljuje u analizi:
- `analysis/audit.py` vizualizira distribuciju klasa i neravnomjernosti (class imbalance)
- Feature embeddingi iz EfficientNetV2 mogu se vizualizirati t-SNE-om (2D projekcija) da se vidi grupiraju li se slične pasmine zajedno — to je nenadzirano korištenje naučenih reprezentacija
- Near-duplicate detekcija u QA-u koristi perceptualni hash koji je sličan konceptu klasteriranja vizualno identičnih slika

---

## PITANJE 14 — Smanjenje dimenzionalnosti, PCA, Autoenkoder

**PCA:**
1. Centraliziraj: X' = X - μ
2. C = (1/N)X'^T X' (kovarijancijska matrica)
3. Eigen-dekompozicija: C = V Λ V^T
4. Projekcija: Z = X' · V[:, :k]  (prvih k eigenvektora)

**Autoenkoder:**
```
x → [Encoder f] → z (usko grlo) → [Decoder g] → x̂
Loss: ||x - x̂||²
```

**PCA vs Autoenkoder:**
| | PCA | Autoenkoder |
|--|-----|-------------|
| Transformacija | **linearna** | **nelinearna** |
| Parametre | nema (analitičko) | uči se gradijentnim spustom |
| Složenost | O(D³) | O(N·epohe) |

**Konvolucijski autoenkoder:**
- Encoder: Conv + Pool (kompresija)
- Decoder: TransposedConv + Upsample (rekonstrukcija)
- Primjena: **denoising** (ulaz=zašumljeno, cilj=čisto), **anomaly detection** (visoka greška → anomalija)

---

### 🔵 U projektu:
Autoenkoderi i PCA nisu direktno implementirani, ali je konceptualna veza:

EfficientNetV2 backbone funkcionira kao **encoder**: 384×384×3 sliku komprimira u 1280-dimenzionalni vektor (embedding).

Ova kompresija je nelinearna (analognija autoenkodera nego PCA):
```
Input: 384×384×3 = 442.368 dimenzija
Embedding: 1.280 dimenzija
Kompresija: 345:1
```

Za vizualizaciju naučenih representacija, embedding vektore možemo projicirati t-SNE-om → pasmine iste vrste bi trebale formirati klastere. To bi pokazalo je li model naučio smislenu strukturu u latentnom prostoru.

---

## PITANJE 15 — Učenje sličnosti, One-shot learning, Sijamske mreže

**One-shot learning**: klasificirati primjer nakon samo jednog primjera po klasi.

**Sijamska mreža:**
```
Slika A → [CNN (dijeljeni param.)] → f(A)─┐
                                            ├→ d(f(A),f(B)) → slično/različito
Slika B → [CNN (dijeljeni param.)] → f(B)─┘
```

**Contrastive loss:** L = (1-y)·½·d² + y·½·max(0, m-d)²  
**Triplet loss:** L = max(0, d(a,p) - d(a,n) + margin)  
(anchor, positive=iste klase, negative=različite klase)

---

### 🔵 U projektu:
Sijamske mreže nisu implementirane, ali je Triplet/Contrastive loss relevantan za razmatranje:

Naš klasifikator pasmina treba razlikovati vizualno **vrlo slične** pasmine (npr. Labrador vs Golden Retriever, ili Holstein vs Simmental goveda). Ovo je izazovan zadatak za standardnu Cross-Entropy.

Alternativa koja bi bila prirodna proširenja projekta:
- **Metric learning** s Triplet loss: nauči embeddinge u kojima su iste pasmine blizu, a različite daleko
- **Contrastive loss**: parovi slika iste pasmine moraju biti bliže u embedding prostoru od parova različitih pasmina

Ovo bi posebno pomoglo kod **long-tail** pasmina s malo primjera — s metrikama bih mogli koristiti **K-NN klasifikator** u embedding prostoru (ne treba puno primjera po klasi).

---

## PITANJE 16 — Sekvencijalni podaci, RNN, LSTM

**Kategorije modeliranja sekvenci:**
- one-to-one: klasifikacija, one-to-many: generiranje
- many-to-one: sentiment analiza, many-to-many: prijevod

**RNN:** h_t = f(W_h·h_{t-1} + W_x·x_t + b) — problem: vanishing gradient

**LSTM gating:**
- Forget gate: f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
- Input gate: i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
- Cell candidate: g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)
- Cell state: C_t = f_t⊙C_{t-1} + i_t⊙g_t
- Output gate: o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
- Hidden state: h_t = o_t⊙tanh(C_t)

---

### 🔵 U projektu:
RNN/LSTM **nisu korišteni** — radimo s pojedinačnim slikama (ne sekvencama).

Potencijalna proširenja gdje bi bili relevantni:
- **Video analiza**: praćenje ponašanja životinja kroz video frame-ove (LSTM nad detektiranim bboxovima)
- **Višekratna detekcija**: ako kamera snima istu životinju kroz više sekundi, LSTM bi mogao akumulirati evidenciju i smanjiti false positives

---

## PITANJE 17 — Podržano učenje: terminologija, funkcije vrijednosti

**Terminologija:** Agent, Okolina, Stanje s, Akcija a, Nagrada r, Politika π(a|s)

**Povrat:** G_t = Σ_{k=0}^∞ γ^k · R_{t+k+1}

**V^π(s)** = E_π[G_t | S_t = s] — funkcija vrijednosti stanja  
**Q^π(s,a)** = E_π[G_t | S_t = s, A_t = a] — Q-funkcija (akcija-stanje)

---

### 🔵 U projektu:
Podržano učenje nije korišteno. Međutim, **mining pipeline** ima površinsku sličnost s RL petljom:

```
Agent (model) → predviđa na neoznačenim podacima (akcija)
             → dobiva povratnu informaciju od eksperta (nagrada/verifikacija)
             → ažurira se s novim podacima (učenje iz nagrade)
```

Ovo je više **active learning** nego RL, ali ideja "agent koji pretražuje prostor podataka i prima feedback" je konceptualno bliska.

---

## PITANJE 18 — Monte Carlo, Q-učenje, SARSA

**Monte Carlo:** procjena V(s) iz stvarnih epizoda (first-visit ili every-visit)

**Q-učenje (off-policy):**  
Q(s,a) ← Q(s,a) + α·[R + γ·max_{a'}Q(s',a') - Q(s,a)]  
→ Uči optimalnu Q neovisno o politici

**SARSA (on-policy):**  
Q(s,a) ← Q(s,a) + α·[R + γ·Q(s',a') - Q(s,a)]  
→ Ažurira prema stvarnoj akciji a'

**Razlika:** Q-učenje uzima max akciju (optimistično), SARSA uzima sljedeću akciju (konzervativno, sigurnije u rizičnim okolinama).

---

### 🔵 U projektu:
Nije korišteno. (Vidjeti napomenu u Q17.)

---

## PITANJE 19 — Polu-nadzirano učenje (SSL)

**Definicija:** Metoda koja koristi **mali skup označenih** (L) i **veliki skup neoznačenih** (U) podataka.
Tipično: 1-10% označenih, 90-99% neoznačenih.

**Kada:** Označavanje je skupo ili sporo (medicinska dijagnostika, audio, pravni dokumenti).

### Usporedba triju tipova:
| | Nadzirano | Nenadzirano | Polu-nadzirano |
|--|-----------|-------------|----------------|
| Označeni podaci | sve | ništa | malo (1-10%) |
| Neoznačeni podaci | ignoriraju se | sve | iskorišteni |
| Trošak anotacije | visok | nema | nizak |
| Točnost (kad malo oznaka) | niska | niža | između ili bolje |

### 4 pretpostavke SSL-a:
1. **Klaster**: točke u istom klasteru → ista oznaka
2. **Glatkoća**: bliske točke → iste oznake (tranzitivno!)
3. **Niska gustoća**: granica odluke kroz područja male gustoće
4. **Mnogostrukost**: visokodim. podaci leže na niskodim. plohi (manifold)

---

### 🔵 U projektu (opširno):

**Mining pipeline = implementacija SSL samo-učenja (self-training):**

Naš projekt ima eksplicitan SSL element koji je projektno odluka, ne slučajnost.

**Problem koji rješavamo:**
- Imamo detekcijske datasete (YOLO format) koji imaju samo **vrstu** životinje (cat, dog, cattle)
- Ti dataseti **nemaju breed anotacije** (nitko nije označio "ovo je Labrador")
- Breed anotacije su skuplje → mali skup L
- Ali ti dataseti imaju puno više slika → veliki skup U

**Implementacija u `mining/unlabeled_animals.py`:**

```python
# Korak 1: Treniranim YOLO detektorom skeniramo neoznačene slike
detector = YOLO("artifacts/species_detectors/{species}/weights/best.pt")
results = detector.predict(source=all_raw_images, conf=0.3, ...)

# Korak 2: Za svaku detekciju izračunamo:
# - species confidence
# - IoU s postojećim oznakama (suzbijanje same-species rediscovery)
# - flip-TTA agreement (konzistentnost predikcije s original i flipped slikom)

# Korak 3: Filtriramo visoko-pouzdane kandidate
# → unlabeled_candidates.csv (svi)
# → unlabeled_candidates_high_priority.csv (visoka conf + TTA agreement)

# Korak 4: Ručna verifikacija
# animal-review-boxes → web app za accept/reject/reannotate

# Korak 5: Verificirani primjeri → novi trening krug
```

**Flip-TTA (Test-Time Augmentation) agreement:**
Ista slika s i bez horizontalnog flipa → ako YOLO daje istu predikciju za oba → veća pouzdanost.
Ovo je automatski filter umjesto ručnog pregleda svakog primjera.

**Vizualizacija kandidata:**
```
results/mining/visual_samples/*.jpg  → spot-check vizualni uzorci
results/mining/per_species_summary.json → koliko kandidata po vrsti
results/mining/spotcheck_queue.csv → prioritizirani za ručnu provjeru
```

**Source mode kontrola:**
- `source_mode=yolo_only` (default): samo YOLO-format dataseti → sigurno, fokusirano
- `source_mode=all_raw`: sve slike iz raw tree → šire pokrivanje ali više FP

**Veza s pretpostavkama SSL-a:**
- **Pretpostavka klasteringa**: životinje iste vrste u sličnom okruženju (farma/studio) tvore vizualne klastere
- **Pretpostavka glatkoće**: vizualno slične slike → iste labele (zato TTA agreement ima smisla)
- **Pretpostavka niske gustoće**: YOLO granicu odluke između vrsta postavlja kroz "prazne" vizualne zone

---

## PITANJE 20 — Metode polu-nadziranog učenja

### 1. Samo-učenje (Self-training):
- Model M trenira na L → predviđa U → visoko-pouzdane pseudo-oznake → dodaje u L → repeat
- Rizik: confirmation bias (greška se amplificira kroz iteracije)
- Prag pouzdanosti τ je ključan hiperparametar

### 2. Uzajamno učenje (Co-training):
- 2 modela, 2 neovisna "pogleda" (views) na iste podatke
- Svaki generira pseudo-oznake za drugoga
- Ključno: pogledi moraju biti uvjetno neovisni

### 3. Propagacija oznaka (Label Propagation):
- Graf: čvorovi=primjeri, bridovi=sličnost (wij)
- Oznake se šire od označenih prema neoznačenima
- Slučajna šetnja: vjerojatnost oznake ∝ putova do označenih čvorova

### 4. Probabilistički generativni modeli:
- Modeliraju p(x,y) = p(y)·p(x|y)
- Neoznačeni podaci pomažu naučiti p(x) (GMM + EM algoritam)

### 5. Polu-nadzirani SVM (S3VM/TSVM):
- Granica odluke kroz nisku gustoću
- Neoznačeni primjeri moraju biti dalje od granice od margine

---

### 🔵 U projektu (opširno):

**Metoda koja je implementirana: Samo-učenje (Self-training)**

Naš mining pipeline je točna implementacija self-traininga:

```
Iteracija 1:
  L = označeni breed dataset (mali)
  U = neoznačeni detekcijski dataseti (veliki)
  M = YOLO11 treniran na L
  M.predict(U) → pseudo-kandidati → filtriranje → verifikacija
  Verificirani → L_new = L ∪ verificirani

Iteracija 2:
  Trenira novi M na L_new (više podataka)
  → bolji model → bolji pseudo-kandidati → ...
```

**Ručna verifikacija kao pouzdanosti filter:**
Umjesto automatskog praga confidence-a (standardni self-training), mi koristimo **ručnu verifikaciju** u `box_review_app.py`. Expert pregleda i potvrđuje/odbacuje.

Ovo je konzervativniji pristup od punog automatskog self-traininga:
- Automatski: samo prag confidence-a (τ ≈ 0.9) → može propagirati greške
- Naš pristup: prag + flip-TTA + ručna verifikacija → manje grešaka, ali sporije

**Zašto nije Co-training:**
Co-training bi zahtijevao dva nezavisna "pogleda" na iste slike.
Potencijalan co-training u našem projektu: Pogled A = RGB image classifier, Pogled B = edge-map classifier.
Ovo nije implementirano, ali je razumno proširenje.

**Zašto nije Label Propagation:**
Label propagation zahtijeva graf sličnosti između svih primjera.
S tisućama slika to je O(N²) memorijski problem.
EfficientNetV2 embeddingi bi mogli biti osnova za graf (kosinusna sličnost), ali nije implementirano.

**Veza s pretpostavkama:**
Naš mining pristup implicitno koristi:
- **Klaster pretpostavku**: YOLO je naučio da iste vrste imaju slične vizualne karakteristike
- **Glatkoću**: flip-TTA agreement provjerava je li predikcija konzistentna za "bliske" slike (original i flip)
- **Nisku gustoću**: YOLO confidence > threshold znači da je primjer "jasno" u regiji jedne klase

---

## PITANJE 21 — Ansambl metode, bias-variance tradeoff

**Ansambl metoda:** Kombinacija više modela → bolja generalizacija.

**Bias-Variance Tradeoff:**
- Pristranost (Bias): greška od pregrubih pretpostavki (underfitting)
- Varijanca (Variance): osjetljivost na šum u train podacima (overfitting)
- MSE = Bias² + Variance + Noise (nereducibilna)

**Bagging**: bootstrap uzorci → smanji varijancu  
**Boosting**: sekvencijalni, fokus na pogreške → smanji pristranost  
**Stacking**: meta-model trenira na predikcijama baznih modela

---

### 🔵 U projektu:

**Implicitni ansambl: Arbitration mehanizam**

Naš dvostupanjski pipeline kombinira **dva neovisna modela** za predikciju vrste:
1. **YOLO11 detektor** → species predikcija (detekcijski pristup)
2. **EfficientNetV2 species classifier** → species predikcija (crop klasifikacijski pristup)

```python
species_top2 = predict_topk(species_model, crop, topk=2)
species_pred, species_conf = species_top2[0]
if species_pred != detector_species and (species_conf - top2[1][1]) < arbitration_gap:
    species_pred = detector_species  # YOLO "pobijedi"
```

Ovo je **weighted voting** između dva modela:
- Ako se classifier slaže s YOLO → uzmi classifier (viši confidence)
- Ako se ne slažu ali classifier je nesiguran (margin < 15%) → uzmi YOLO
- Ako se ne slažu ali classifier je siguran → uzmi classifier

Takva kombinacija smanjuje grešku kada jedan od modela pogriješi (različite vrste grešaka).

**Bias-Variance u kontekstu projekta:**
- EfficientNetV2 s pretreniranjem: nizak bias (moćan model), ali potencijalno visoka varijanca kod rijetkih pasmina
- Rješavamo varijancu: augmentacija, dropout, weight decay, early stopping
- Rješavamo bias: transfer learning od IN-21k daje bogatiju reprezentaciju

**Curriculum training kao ansambl ideja:**
Tri faze treniranja YOLO (A: species → B: breed → C: long-tail) su analogne boosting pristupu: svaka faza "popravlja" greške prethodne, s fokuson na teže slučajeve (rijetke pasmine u fazi C).

---

## PITANJE 22 — Stablo odlučivanja, Slučajna šuma, AdaBoost

### Stablo odlučivanja:
- Rekurzivna podjela prostora značajki
- Kriterij: Gini impurity = 1 - Σ p_k²; Info Gain (entropija)
- Prednosti: interpretabilno, brzo, ne treba normalizaciju
- Nedostaci: overfitting, nestabilno (high variance)

### Slučajna šuma:
- Bagging N stabala
- Na svakom čvoru: nasumičan podskup √D od D značajki
- Glasanje za finalnu predikciju
- Smanjuje varijancu bez povećavanja pristranosti

### AdaBoost:
1. w_i = 1/N (jednake težine)
2. Treniraj stump h_t
3. ε_t = Σ w_i·𝟙[h_t(x_i) ≠ y_i]
4. α_t = ½·ln((1-ε_t)/ε_t)
5. w_i ← w_i·exp(-α_t·y_i·h_t(x_i)), normalizacija
6. H(x) = sign(Σ α_t·h_t(x))

---

### 🔵 U projektu:

Ansambl metode (stablo, RF, AdaBoost) nisu direktno korištene — odabrali smo duboke mreže.

**Zašto duboke mreže umjesto Random Forest/AdaBoost:**
- **Slike su visokodimenzionalni podaci**: 384×384×3 = 442.368 dimenzija
- RF/AdaBoost rade dobro na tablični (low-dim) podacima, ali na sirovim pikselima loše skaliraju
- CNN automatski uči hijerarhijske feature (rub → tekstura → dio tijela → životinja) — to nema RF
- Transfer learning nije moguć s RF (ne postoje RF pretrenirani na ImageNetu)

**Gdje bi RF mogao biti koristan u projektu:**
- Na CNN embeddingima (1280-dim.) moglo bi biti RF-based klasifikatora pasmina — to je poznata tehnika
- RF nad EfficientNetV2 embeddinzima bi bio brži za deploy i interpretabilniji
- `relative_promotion_gate` koristi jednostavna if/else pravila — to je degenerisano stablo odlučivanja

**Analogija: Quality gate kao stablo odlučivanja:**
QA gate u `data/quality/qa.py` je eksplicitno stablo pravila:
```
IF bbox_bounds_error > 0 → INVALID
IF taxonomy_hierarchy_error > 0 → INVALID
IF split_leakage > 0 → INVALID
IF near_duplicates > threshold AND fail_on_warnings → INVALID
ELSE → VALID
```
Ovo je ručno kodiran decision tree koji čuva integritet dataseta.

---

## KRATKI CHEATSHEET — Ključni pojmovi za ispit

| Pojam | Što je u projektu |
|-------|-------------------|
| **Transfer learning** | EfficientNetV2 pretreniran na IN-21k; `timm.create_model(..., pretrained=True)` |
| **CNN** | EfficientNetV2-S backbone; YOLO11 detektor |
| **Fine-tuning** | Cijela mreža se ažurira (ne samo head) s malim lr |
| **AdamW** | Optimizator s decoupled weight decay; bolja regularizacija od Adam+L2 |
| **CosineAnnealingLR** | LR scheduler; glatko smanjenje lr kroz epohe |
| **AMP** | Automatic Mixed Precision (float16 forward, float32 update) → ~2× brže |
| **Label smoothing** | Regularizacija targeta; sprječava overconfident model |
| **Early stopping** | Zaustavljanje kad val_top1 prestane rasti; čuva best.pt |
| **Two-stage pipeline** | YOLO detektira vrstu → EfficientNetV2 klasificira pasminu |
| **Arbitration** | Mini-ansambl YOLO+classifier za species predikciju |
| **SSL/Mining** | Self-training; pseudo-label kandidati iz neoznačenih slika |
| **Macro F1** | Jednaka težina svakoj klasi; kažnjava zanemarivanje rijetkih pasmina |
| **ECE** | Kalibriranost; je li 90% confidence = 90% točnost? |
| **Brier score** | Probabilistička greška; niži = bolji probabiliteti |
| **mAP@50** | Standardna COCO metrika za detektor |
| **Promotion gate** | Automatski odlučuje smije li novi model zamijeniti baseline |
| **Taxonomy** | YAML hijerarhija species → breeds; validira ID format "species:breed" |
| **Manifest** | CSV tablica svih slika s metapodacima; centralna baza projekta |
| **QA gate** | 6 provjera kvalitete dataseta; blokira treniranje ako postoje greške |
| **Curriculum training** | 3-fazno YOLO treniranje (A→B→C); analogno boosting pristupu |

---

*Puno sreće na ispitu! 12.06.2026 u 8:30h, Kampus.*
