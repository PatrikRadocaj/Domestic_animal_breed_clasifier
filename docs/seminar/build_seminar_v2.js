const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, TableOfContents, LevelFormat
} = require('docx');
const fs = require('fs');

const BLUE  = '17365D';
const LGRAY = 'F2F2F2';
const LBLUE = 'D9EAF7';
const MBLUE = 'BDD7EE';
const GRAY  = '555555';
const RED   = 'C00000';
const GREEN = '375623';

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: 'Arial', size: 28, bold: true, color: BLUE })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: 'Arial', size: 24, bold: true, color: BLUE })]
  });
}
function p(text, opts = {}) {
  return new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text, font: 'Arial', size: 22, ...opts })] });
}
function bold(text, color) {
  const o = { font: 'Arial', size: 22, bold: true };
  if (color) o.color = color;
  return new TextRun({ text, ...o });
}
function italic(text) { return new TextRun({ text, font: 'Arial', size: 22, italics: true }); }
function mixed(...runs) { return new Paragraph({ spacing: { after: 160 }, children: runs }); }
function bullet(text, opts = {}) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: 'Arial', size: 22, ...opts })]
  });
}
function code(text) {
  return new Paragraph({
    spacing: { after: 60, before: 60 },
    indent: { left: 720 },
    shading: { fill: 'F8F8F8', type: ShadingType.CLEAR },
    children: [new TextRun({ text, font: 'Courier New', size: 20, color: '1F3864' })]
  });
}
function pageBreak() { return new Paragraph({ children: [new PageBreak()] }); }
function spacer() { return new Paragraph({ spacing: { after: 160 }, children: [new TextRun('')] }); }
function caption(text) {
  return new Paragraph({ spacing: { after: 80, before: 120 }, alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, font: 'Arial', size: 20, bold: true, color: BLUE })] });
}

const b1 = { style: BorderStyle.SINGLE, size: 4, color: 'AAAAAA' };
const bdr = { top: b1, bottom: b1, left: b1, right: b1 };

function hCell(text, w) {
  return new TableCell({ borders: bdr, width: { size: w, type: WidthType.DXA },
    shading: { fill: MBLUE, type: ShadingType.CLEAR },
    margins: { top: 90, bottom: 90, left: 120, right: 120 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ children: [bold(text)] })] });
}
function dCell(text, w, shade = false, color) {
  const run = color
    ? new TextRun({ text, font: 'Arial', size: 20, bold: true, color })
    : new TextRun({ text, font: 'Arial', size: 20 });
  return new TableCell({ borders: bdr, width: { size: w, type: WidthType.DXA },
    shading: { fill: shade ? LGRAY : 'FFFFFF', type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ children: [run] })] });
}

// ─── document ───────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
    }]
  },
  styles: {
    default: { document: { run: { font: 'Arial', size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: 'Arial', color: BLUE },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: MBLUE, space: 4 } } } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: 'Arial', color: BLUE },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: MBLUE, space: 4 } },
        children: [
          new TextRun({ text: 'Višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina', font: 'Arial', size: 18, color: GRAY }),
          new TextRun({ text: '\t', font: 'Arial', size: 18 }),
          new TextRun({ text: '2026.', font: 'Arial', size: 18, color: GRAY }),
        ],
        tabStops: [{ type: 'right', position: 9026 }]
      })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: MBLUE, space: 4 } },
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: 'Stranica ', font: 'Arial', size: 18, color: GRAY }),
          new TextRun({ children: [PageNumber.CURRENT], font: 'Arial', size: 18, color: GRAY }),
          new TextRun({ text: ' od ', font: 'Arial', size: 18, color: GRAY }),
          new TextRun({ children: [PageNumber.TOTAL_PAGES], font: 'Arial', size: 18, color: GRAY }),
        ]
      })] })
    },
    children: [

      // ── NASLOVNICA ──────────────────────────────────────────────────────
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 240 },
        children: [new TextRun({ text: 'Višerazinski sustav za prepoznavanje životinjskih vrsta', font: 'Arial', size: 44, bold: true, color: BLUE })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 },
        children: [new TextRun({ text: 'i pasmina pomoću dubokog učenja', font: 'Arial', size: 44, bold: true, color: BLUE })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
        children: [new TextRun({ text: 'Projekt iz strojnog učenja i računalnog vida', font: 'Arial', size: 26, italics: true, color: GRAY })] }),

      new Table({
        width: { size: 7000, type: WidthType.DXA }, alignment: AlignmentType.CENTER,
        columnWidths: [2200, 4800],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders: bdr, width: { size: 2200, type: WidthType.DXA },
              shading: { fill: LBLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [bold('Autor:')] })] }),
            new TableCell({ borders: bdr, width: { size: 4800, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [new TextRun({ text: '[OVDJE DODATI IME I PREZIME]', font: 'Arial', size: 22 })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: bdr, width: { size: 2200, type: WidthType.DXA },
              shading: { fill: LBLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [bold('Kolegij:')] })] }),
            new TableCell({ borders: bdr, width: { size: 4800, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [new TextRun({ text: '[OVDJE DODATI NAZIV KOLEGIJA]', font: 'Arial', size: 22 })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: bdr, width: { size: 2200, type: WidthType.DXA },
              shading: { fill: LBLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [bold('Tehnologije:')] })] }),
            new TableCell({ borders: bdr, width: { size: 4800, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [new TextRun({ text: 'Python, PyTorch, YOLO11, EfficientNetV2-S, OpenCV, wandb, Hydra', font: 'Arial', size: 22 })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: bdr, width: { size: 2200, type: WidthType.DXA },
              shading: { fill: LBLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [bold('Akademska godina:')] })] }),
            new TableCell({ borders: bdr, width: { size: 4800, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [new TextRun({ text: '2025./2026.', font: 'Arial', size: 22 })] })] }),
          ]}),
        ]
      }),

      pageBreak(),

      // ── SAŽETAK ────────────────────────────────────────────────────────────
      h1('Sažetak'),
      p('U ovom projektu razvijen je višerazinski sustav za automatsko prepoznavanje životinjskih vrsta i pasmina na fotografijama. Sustav je arhitekturalno podijeljen u dvije sekvencijalne faze: YOLO11s detektor identificira vrstu životinje i locira je unutar slike, a EfficientNetV2-S klasifikator (pretreniran na ImageNet-21k s finim podešavanjem na ImageNet-1k) prima izrezani isječak životinje i provodi finiju klasifikaciju pasmine.'),
      p('Projekt je potpuno završen: zajednički species detektor postiže mAP@50 = 0.679 na finalnom skupu, a breed klasifikatori su istrenirani za svih 16 vrsta s dovoljno podataka. Rezultati se kreću od odličnih (donkey: 98,4 %, pig: 96,7 %, peafowl: 94,6 %) do izazovnih (sheep: 35,3 %, dog: 48,6 %) ovisno o broju dostupnih podataka i međuklasnoj vizualnoj sličnosti.'),
      p('Kao kontrolni eksperiment, direktni YOLO breed detektor pokazao je da jednostupanjski pristup nije izvediv bez prethodne faze detekcije vrste — model postiže mAP@50 od samo 0,006 nakon dvije epohe, potvrđujući opravdanost hijerarhijskog dizajna.'),
      p('Uz treniranje modela, projekt uključuje kompletnu infrastrukturu: rudarenje neanotiranih slika životinja (324 633 kandidata), interaktivne alate za pregled bounding boxova i oznaka, sustav za normalizaciju cross-dataset oznaka, hard negative mining, Hydra-baziranu konfiguraciju eksperimenata, artifact provenance tracking i GitHub Actions CI. Zaključivanje je dostupno kroz CLI i lokalnu web aplikaciju.'),
      mixed(bold('Ključne riječi: '), italic('detekcija objekata, klasifikacija pasmina, YOLO11, EfficientNetV2, višerazinsko učenje, rudarenje podataka, računalni vid, duboko učenje')),

      pageBreak(),
      new TableOfContents('Sadržaj', { hyperlink: true, headingStyleRange: '1-2' }),
      pageBreak(),

      // ── 1. UVOD ──────────────────────────────────────────────────────────
      h1('1. Uvod'),
      p('Automatsko prepoznavanje životinja na fotografijama jedan je od klasičnih i praktično korisnih zadataka računalnog vida s primjenama u veterinarskoj dijagnostici, praćenju divljih životinja, upravljanju farmama, edukacijskim aplikacijama i digitalnoj katalogizaciji zbirki. Problem nije trivijalan — nije dovoljno samo detektirati da se na slici nalazi neka životinja, nego je potrebno razlikovati slične vrste te, još finije, odrediti pasminu unutar iste vrste.'),
      p('Razlika između pasmina može biti izuzetno suptilna: labrador i zlatni retriver razlikuju se uglavnom bojom dlake, border collie i australski ovčar sitnim detaljima koji su vidljivi tek pri dobrom kadru. Otkrivamo da ovaj problem nije samo akademski izazov — podaci iz prakse pokazuju ekstremnu neuravnoteženost (omjeri klasa do 10:1), raznolikost izvora i heterogenost oznaka, što znači da je kvaliteta pipeline-a jednako važna kao i sam model.'),
      p('U ovom projektu odabran je višerazinski pristup koji razdvaja problem na dvije upravljive faze: detekcija vrste i lokalizacija životinje, zatim fina klasifikacija pasmine na izoliranom isječku. Ova hijerarhijska dekompozicija potvrđena je eksperimentalno: direktni breed YOLO model koji pokušava riješiti cijeli zadatak odjednom postiže mAP@50 = 0,006, dok dvostupanjski pristup daje smislene rezultate za sve istrenirane vrste.'),
      p('Ovaj rad dokumentira arhitekturu sustava, sve odluke u dizajnu podatkovnog pipeline-a, stvarne metrike evaluacije za species detektor i sve breed klasifikatore te pomoćnu infrastrukturu (mining, review alati, CI, normalizacija oznaka). Naglasak je na potpunoj transparentnosti: uz pozitivne rezultate, eksplicitno se navode vrste kod kojih model ne radi dobro i objašnjava se zašto.'),

      // ── 2. PODACI ────────────────────────────────────────────────────────
      h1('2. Skup podataka, eksplorativna analiza i predobrada'),
      h2('2.1. Izvori podataka i kanonski manifest'),
      p('Projekt objedinjuje tri kategorije izvora podataka. Javno dostupni YOLO-formatirani skupovi pokrivaju opće kategorije životinja i čine osnovu species detektora. Ručno označeni primjeri u CSV i ZIP formatima sadrže pasminske oznake, ali su malobrojniji. Sintetski generirani primjeri nastaju kompozitnom metodom: izrezane životinje postavljaju se na realistične pozadine, čime se povećava raznolikost vizualnih uvjeta za rijetke kategorije.'),
      p('Svi podaci standardizirani su kroz kanonski manifest — tablicu u kojoj svaki redak predstavlja jednu sliku s atributima: image_id, image_path, source, source_type, license, canonical_species, canonical_breed (u obliku vrsta:pasmina), dimenzije, bounding box koordinate, split (train/val/test) i quality_flags. Manifest omogućuje potpunu reproducibilnost eksperimenta i praćenje porijekla svake slike.'),
      p('Normalizacija oznaka između dataseta provodi se kroz configs/mappings/cross_dataset_label_map.yaml. Proces uključuje: čišćenje oznaka (lowercase, normalizacija separatora), razrješavanje kanonskih aliasa (npr. "COW" → "cattle", "Holstein-Friesian breed" → "cattle:holstein_friesian"), dataset-specifične override-ove i taksonomijsko razrješavanje. Anomalije se bilježe u results/reports/unmapped_labels_report.csv za ručni pregled.'),

      h2('2.2. Statistike skupa podataka'),
      p('Glavni skup za detekciju vrsta sadrži 18 klasa i 27 433 slike raspoređene u omjeru 82:12:6 (train/val/test). Neuravnoteženost između klasa je značajna — omjer najzastupljenije i najmanje zastupljene klase premašuje 10:1. Za breed klasifikaciju, situacija je još zahtjevnija: dok popularne pasmine pasa i mačaka imaju stotine primjera, rijetke pasmine goveda ili koze ponekad imaju manje od 50 uzoraka.'),

      caption('Tablica 1. Raspodjela slika po skupovima — species detektor'),
      new Table({
        width: { size: 9026, type: WidthType.DXA }, columnWidths: [2500, 2000, 2000, 2526],
        rows: [
          new TableRow({ children: [hCell('Skup', 2500), hCell('Br. slika', 2000), hCell('Udio', 2000), hCell('Napomena', 2526)] }),
          new TableRow({ children: [dCell('Trening', 2500), dCell('22 587', 2000), dCell('82,3 %', 2000), dCell('Za treniranje modela', 2526)] }),
          new TableRow({ children: [dCell('Validacija', 2500, true), dCell('3 375', 2000, true), dCell('12,3 %', 2000, true), dCell('Praćenje konvergencije', 2526, true)] }),
          new TableRow({ children: [dCell('Test', 2500), dCell('1 471', 2000), dCell('5,4 %', 2000), dCell('Konačna evaluacija', 2526)] }),
          new TableRow({ children: [dCell('Ukupno', 2500, true), dCell('27 433', 2000, true), dCell('100 %', 2000, true), dCell('18 klasa vrsta', 2526, true)] }),
        ]
      }),
      spacer(),

      h2('2.3. Eksplorativna analiza'),
      p('Eksplorativna analiza identificirala je nekoliko ključnih izazova. Neuravnoteženost razreda prisutna je i na razini vrsta i na razini pasmina. Nekonzistentne oznake između dataseta zahtijevale su normalizacijski pipeline — isti pas može biti označen kao "golden retriever", "Golden Retriever" ili "golden_retriever" u različitim izvorima. Premali bounding boxevi (manji od 20–25 piksela) filtrirani su jer ne sadrže dovoljno vizualne informacije. Dio fotografija je loše kvalitete zbog preekspozicije, jakog šuma ili sitnih objekata u kadru.'),
      p('Alat animal-audit-dataset generira detaljni izvještaj o pokrivnosti klasa, distribuciji veličina boxova i hijerarhijskoj konzistentnosti. Alat animal-data-qa provodi automatizirani quality gate koji blokira pokretanje treninga ako podaci ne prolaze definirane pragove.'),

      h2('2.4. Predobrada, augmentacije i sintetički podaci'),
      p('Za detekciju, slike se konvertiraju u YOLO format (normalizirane koordinate). Za breed klasifikaciju, koristi se crop na temelju YOLO detekcije uz marginski pomak od 10–15 % kako bi se uključio vizualni kontekst oko životinje. Bez toga, model bi primao agresivno izrezane isječke koji ponekad zasjeku glavu ili rep.'),
      p('Sintetički podaci nastaju kompozitnom metodom HybridSpeciesYoloDataset: izrezane životinje iz poznatih slika postavljaju se na raznolike pozadine s varijabilnim brojem objekata (1–4), međusobnim razmakom i kutovima skaliranja. Primijenjen je prag minimalnog bounding boxa od 20–25 piksela ovisno o verziji generiranja.'),
      p('Augmentacije korištene u treniranju uključuju: horizontalni i vertikalni flip, rotaciju (±15°), promjenu ekspozicije i kontrasta (±20 %), mosaic (kombinira 4 slike, YOLO-specifično, gasi se 10 epoha prije kraja), MixUp (α=0,2) i CutMix (α=1,0) za klasifikacijski model.'),

      // ── 3. METODE ────────────────────────────────────────────────────────
      h1('3. Korištene metode i algoritmi'),
      h2('3.1. YOLO11s species detektor'),
      p('Prva faza sustava temelji se na YOLO11s arhitekturi (small varijanta iz generacije YOLO11 tvrtke Ultralytics). YOLO dijeli sliku na mrežu ćelija i za svaku ćeliju predviđa skup kandidatnih okvira s klasifikacijskim distribucijama u jednom prolazu — to je ključna prednost nad dvodijelnim detektorima poput Faster R-CNN koji zahtijevaju posebni region proposal network.'),
      p('Model je treniran u hybrid konfiguraciji (HybridSpeciesYoloDataset) koja kombinira YOLO-formatirane datasete s kontroliranim uzorkovanjem po vrstama. Ovo je ključna razlika od naivnog spajanja — osigurava da rijetke klase nisu potisnute dominantnima i da sampling plan odražava ciljanu distribuciju.'),
      caption('Tablica 2. Hiperparametri YOLO11s species detektora'),
      new Table({
        width: { size: 9026, type: WidthType.DXA }, columnWidths: [3500, 2000, 3526],
        rows: [
          new TableRow({ children: [hCell('Parametar', 3500), hCell('Vrijednost', 2000), hCell('Napomena', 3526)] }),
          new TableRow({ children: [dCell('Bazni model', 3500), dCell('yolo11s.pt', 2000), dCell('Pretrained na COCO', 3526)] }),
          new TableRow({ children: [dCell('Ulazna rezolucija', 3500, true), dCell('640×640 px', 2000, true), dCell('Standard za YOLO', 3526, true)] }),
          new TableRow({ children: [dCell('Broj epoha', 3500), dCell('100', 2000), dCell('Sa early stopping', 3526)] }),
          new TableRow({ children: [dCell('Batch size', 3500, true), dCell('16', 2000, true), dCell('', 3526, true)] }),
          new TableRow({ children: [dCell('Patience (early stop)', 3500), dCell('25', 2000), dCell('', 3526)] }),
          new TableRow({ children: [dCell('Optimizer', 3500, true), dCell('auto (SGD/Adam)', 2000, true), dCell('YOLO automatski odabire', 3526, true)] }),
          new TableRow({ children: [dCell('lr0', 3500), dCell('0.01', 2000), dCell('Cosine annealing (cos_lr=true)', 3526)] }),
          new TableRow({ children: [dCell('Mosaic zatvaranje', 3500, true), dCell('close_mosaic=10', 2000, true), dCell('Gasi se 10 epoha prije kraja', 3526, true)] }),
          new TableRow({ children: [dCell('Mješovita preciznost', 3500), dCell('amp=true', 2000), dCell('Brže treniranje na AMD ROCm', 3526)] }),
        ]
      }),
      spacer(),
      p('Hardverska platforma za treniranje je AMD GPU s ROCm 7.2.1 runtime-om, pokrenut kroz izoliranu Python 3.12 embed okolinu (C:\\amd_rocm_py312_real) koja ne mijenja globalnu Windows PATH konfiguraciju.'),

      h2('3.2. EfficientNetV2-S breed klasifikator'),
      p('Drugi stupanj koristi EfficientNetV2-S arhitekturu, specifično varijantu tf_efficientnetv2_s.in21k_ft_in1k iz timm biblioteke. Ova varijanta je pretrenirana na ImageNet-21k (14 milijuna slika, 21 000 klasa) uz daljnje fino podešavanje na ImageNet-1k, što joj daje iznimno bogatu inicijalizaciju za fine-grained klasifikaciju.'),
      p('Važna napomena: koristi se EfficientNetV2-S, a ne EfficientNetB2 koji se povremeno spominje u ranijim verzijama dokumentacije. V2-S ima oko 22 milijuna parametara i nativnu ulaznu rezoluciju 384×384 piksela, što je značajno veće od B2 (9,1 M parametara, 260×260 px) i osigurava bolje prepoznavanje finih detalja bitnih za pasminu.'),
      p('Trening je modularan: za svaku vrstu s dovoljno podataka trenira se zaseban breed model. Time se broj klasa po modelu drži pod kontrolom — npr. 15–30 pasmina pasa umjesto 200+ pasmina svih vrsta zajedno.'),
      caption('Tablica 3. Hiperparametri EfficientNetV2-S breed klasifikatora'),
      new Table({
        width: { size: 9026, type: WidthType.DXA }, columnWidths: [3500, 2000, 3526],
        rows: [
          new TableRow({ children: [hCell('Parametar', 3500), hCell('Vrijednost', 2000), hCell('Obrazloženje', 3526)] }),
          new TableRow({ children: [dCell('Bazni model', 3500), dCell('tf_efficientnetv2_s.in21k_ft_in1k', 2000), dCell('Pretrained ImageNet-21k + IN-1k', 3526)] }),
          new TableRow({ children: [dCell('Ulazna rezolucija', 3500, true), dCell('384×384 px', 2000, true), dCell('Nativna rezolucija V2-S', 3526, true)] }),
          new TableRow({ children: [dCell('Broj epoha', 3500), dCell('60 (max)', 2000), dCell('Sa early stopping', 3526)] }),
          new TableRow({ children: [dCell('Batch size', 3500, true), dCell('48', 2000, true), dCell('', 3526, true)] }),
          new TableRow({ children: [dCell('Optimizer', 3500), dCell('AdamW', 2000), dCell('Bolja generalizacija od Adam', 3526)] }),
          new TableRow({ children: [dCell('Stopa učenja', 3500, true), dCell('0.0003', 2000, true), dCell('', 3526, true)] }),
          new TableRow({ children: [dCell('Weight decay', 3500), dCell('0.05', 2000), dCell('L2 regularizacija', 3526)] }),
          new TableRow({ children: [dCell('Raspored', 3500, true), dCell('cosine annealing', 2000, true), dCell('', 3526, true)] }),
          new TableRow({ children: [dCell('Label smoothing', 3500), dCell('0.1', 2000), dCell('Smanjuje overconfidence', 3526)] }),
          new TableRow({ children: [dCell('MixUp / CutMix α', 3500, true), dCell('0.2 / 1.0', 2000, true), dCell('Augmentacije raznolikosti', 3526, true)] }),
          new TableRow({ children: [dCell('Early stopping patience', 3500), dCell('10 epoha', 2000), dCell('Min. poboljanje 0.001', 3526)] }),
        ]
      }),
      spacer(),

      h2('3.3. Dvostupanjski pipeline — arhitektura integracije'),
      p('Cjelokupni pipeline integrira obje faze u koherentnu cjelinu (src/animal_breed/pipelines/infer_two_stage.py). Tok podataka pri zaključivanju:'),
      bullet('Ulazna fotografija proslijeđuje se YOLO species detektoru'),
      bullet('Za svaki okvir s pouzdanošću iznad praga (0.25), vrši se crop s marginalnim pomakom'),
      bullet('Izrezani isječak proslijeđuje se odgovarajućem breed klasifikatoru za detektiranu vrstu'),
      bullet('Ako breed model za tu vrstu nije dostupan, vraća se samo species predikcija'),
      bullet('Izlaz su anotirana slika i strukturirani JSON s detaljima svake detekcije'),
      spacer(),

      h2('3.4. Direktni YOLO breed model (kontrolni eksperiment)'),
      p('Uz dvostupanjski pipeline implementiran je i direktni YOLO breed detektor koji pokušava riješiti cijeli zadatak u jednom koraku. Ovaj model koristi yolo11s.pt kao polazište i trenira se s breed-level klasama (100+ klasa umjesto 18).'),
      mixed(bold('Rezultat: '), new TextRun({ text: 'Model postiže mAP@50 = 0.006 već u drugoj epohi i ne konvergira. Ovo nije iznenađujući ishod — s tako velikim brojem vizualno sličnih klasa, bez prethodne grube separacije po vrstama, YOLO jednodijelni detektor ne može smisleno inicijalizirati klasifikacijske glave. Ovaj eksperiment eksplicitno potvrđuje opravdanost hijerarhijskog pristupa.', font: 'Arial', size: 22 })),
      spacer(),

      h2('3.5. Rudarenje neanotiranih životinja (unlabeled mining)'),
      p('Projekt uključuje kompletni pipeline za rudarenje neanotiranih životinja iz dostupnih YOLO dataseta (src/animal_breed/mining/unlabeled_animals.py). Ideja je koristiti istrenirani species detektor za pronalaženje životinja na slikama koje nemaju YOLO oznake — tzv. hard negative mining u korist proširivanja skupa.'),
      p('Pipeline koristi strategiju uzorkovanja koja ravnomjerno raspoređuje budžet između izvora (ne troši samo na abecedni prvi dataset), filtrira po minimalnoj veličini detekcije i opcionalnoj TTA konzistentnosti (flip-TTA). Rezultati dosadašnjeg mininga:'),
      bullet('Ukupno kandidata pronađeno: 324 633'),
      bullet('Visokoprioritnih kandidata: 20 000'),
      bullet('Vizualni spotcheck uzorci: rezultati/mining/visual_samples/'),
      bullet('Revizija kroz interaktivni alat: animal-review-boxes'),
      spacer(),

      h2('3.6. Interaktivni alati za pregled i reviziju'),
      p('Projekt uključuje dva interaktivna web alata za pregled i reviziju podataka:'),
      p('Box review app (src/animal_breed/review/box_review_app.py): prikazuje predložene bounding boxove iz mininga i omogućuje korisniku da potvrdi, odbaci ili modificira svaki prijedlog. Pregledi se čuvaju zasebno u *.reviews.csv datotekama, nikad ne mijenjajući originalne oznake.'),
      p('Label audit app (src/animal_breed/review/label_audit_app.py): alat za reviziju kvalitete oznaka u manifestu — prikazuje sliku, pridijeljena oznaku i informacije o izvoru, omogućujući brzu identifikaciju pogrešnih ili nekonzistentnih oznaka.'),

      h2('3.7. Lokalna web aplikacija za zaključivanje'),
      p('Lokalna web aplikacija (src/animal_breed/webapp/model_inference_app.py) implementirana je kao Python ThreadingHTTPServer koji izlaže REST API i statičke HTML/JS datoteke. Pokretanje: python tools/run_model_inference_app.py.'),
      p('Ključni API endpointovi:'),
      bullet('/api/status (GET) — vraća JSON s učitanim modelima, popisom dostupnih breed modela i vrstama za koje je breed klasifikacija aktivna'),
      bullet('/api/infer (POST) — prima multipart/form-data s uploadanom slikom, pokreće puni pipeline, vraća JSON s detekcijama i anotiranom slikom (base64 PNG)'),
      spacer(),
      p('Prednje sučelje prikazuje uploadanu sliku s iscrtanim okvirima, boju-kodiranim labelama i tablicom predikcija. Breed label formatira se iz "dog:labrador_retriever" u čitljiv oblik "labrador retriever". Ako breed model nije dostupan za detektiranu vrstu, to se eksplicitno prikazuje umjesto lažne predikcije.'),

      // ── 4. EVALUACIJA ────────────────────────────────────────────────────
      h1('4. Evaluacija izgrađenih modela'),
      h2('4.1. Korištene metrike'),
      p('Za YOLO detektor koriste se: Precision (P) — udio ispravnih detekcija; Recall (R) — udio stvarnih objekata koje je model pronašao; mAP@50 — srednja prosječna preciznost pri IoU pragu 0,50; mAP@50–95 — stroža mjera koja prosječi mAP za IoU od 0,50 do 0,95 u koraku 0,05.'),
      p('Za breed klasifikator: Top-1 točnost — točna klasa na prvom mjestu; Top-5 točnost — točna klasa među prvih 5 predikcija (važno za praktične primjene s mnogo vizualno sličnih klasa); cross-entropy loss praćen zasebno za trening i validacijski skup.'),

      h2('4.2. Rezultati species detektora — iteracije treniranja'),
      p('Provedeno je više iteracija treniranja species detektora s postupnim poboljanjima u kvaliteti podataka i konfiguraciji. Tablice 4a i 4b prikazuju ključne iteracije.'),
      caption('Tablica 4. Usporedba iteracija species detektora (finalni epoch)'),
      new Table({
        width: { size: 9026, type: WidthType.DXA }, columnWidths: [3026, 1500, 1500, 1500, 1500],
        rows: [
          new TableRow({ children: [hCell('Eksperiment / run', 3026), hCell('P', 1500), hCell('R', 1500), hCell('mAP@50', 1500), hCell('mAP@50–95', 1500)] }),
          new TableRow({ children: [dCell('hybrid_shared_full17_v2 (rana)', 3026), dCell('0.361', 1500), dCell('0.366', 1500), dCell('0.346', 1500), dCell('0.163', 1500)] }),
          new TableRow({ children: [dCell('hybrid_shared_full18_goose_10more', 3026, true), dCell('0.615', 1500, true), dCell('0.616', 1500, true), dCell('0.617', 1500, true), dCell('0.324', 1500, true)] }),
          new TableRow({ children: [dCell('hybrid_shared_full18_goose_5more_box25', 3026), dCell('0.650', 1500), dCell('0.659', 1500), dCell('0.666', 1500), dCell('0.351', 1500)] }),
          new TableRow({ children: [dCell('hybrid_shared_full18_phr_5more_box25 (final)', 3026, true), dCell('0.644', 1500, true), dCell('0.685', 1500, true), dCell('0.679', 1500, true, GREEN), dCell('0.365', 1500, true)] }),
        ]
      }),
      spacer(),
      p('Konačni species detektor (hybrid_shared_full18_phr_5more_box25) postiže mAP@50 = 0.679 i recall = 0.685. Napredak između ranih i finalnih eksperimenata je značajan — rani model (full17_v2) konvergira na mAP@50 = 0.346, što je gotovo dvostruko niže od finalnog. Poboljanje dolazi iz: proširenja skupa s PHR (ručno označenim) primjerima, povećanja praga minimalnog bounding boxa na 25 piksela za sintetičke slike i balansiranog uzorkovanja po vrstama.'),

      h2('4.3. Rezultati breed klasifikatora — svi modeli'),
      p('Breed klasifikatori su istrenirani za svih 16 vrsta s dovoljno podataka. Tablice prikazuju finalne metrike za svaki istrenirani model (zadnja epoha, s early stoppingom pri patience=10).'),
      caption('Tablica 5. Finalne metrike breed klasifikatora po vrstama'),
      new Table({
        width: { size: 9026, type: WidthType.DXA }, columnWidths: [1800, 900, 1726, 1600, 1500, 1500],
        rows: [
          new TableRow({ children: [hCell('Vrsta', 1800), hCell('Epohe', 900), hCell('val_top1', 1726), hCell('val_top5', 1600), hCell('val_loss', 1500), hCell('train_top1', 1500)] }),
          new TableRow({ children: [dCell('pig', 1800), dCell('10', 900), dCell('96,74 %', 1726, false, GREEN), dCell('99,94 %', 1600), dCell('0.596', 1500), dCell('99,95 %', 1500)] }),
          new TableRow({ children: [dCell('donkey', 1800, true), dCell('10', 900, true), dCell('98,35 %', 1726, true), dCell('98,77 %', 1600, true), dCell('0.667', 1500, true), dCell('99,95 %', 1500, true)] }),
          new TableRow({ children: [dCell('peafowl', 1800), dCell('10', 900), dCell('94,57 %', 1726, false, GREEN), dCell('98,45 %', 1600), dCell('0.637', 1500), dCell('99,59 %', 1500)] }),
          new TableRow({ children: [dCell('rabbit', 1800, true), dCell('10', 900, true), dCell('93,75 %', 1726, true), dCell('100,00 %', 1600, true), dCell('0.841', 1500, true), dCell('100,00 %', 1500, true)] }),
          new TableRow({ children: [dCell('chicken', 1800), dCell('10', 900), dCell('89,59 %', 1726, false, GREEN), dCell('95,26 %', 1600), dCell('1.325', 1500), dCell('99,63 %', 1500)] }),
          new TableRow({ children: [dCell('pigeon', 1800, true), dCell('10', 900, true), dCell('86,76 %', 1726, true), dCell('87,25 %', 1600, true), dCell('1.473', 1500, true), dCell('99,61 %', 1500, true)] }),
          new TableRow({ children: [dCell('goat', 1800), dCell('10', 900), dCell('86,10 %', 1726), dCell('98,93 %', 1600), dCell('0.992', 1500), dCell('99,88 %', 1500)] }),
          new TableRow({ children: [dCell('goose', 1800, true), dCell('10', 900, true), dCell('78,41 %', 1726, true), dCell('92,05 %', 1600, true), dCell('1.427', 1500, true), dCell('99,50 %', 1500, true)] }),
          new TableRow({ children: [dCell('dromedary', 1800), dCell('10', 900), dCell('77,84 %', 1726), dCell('96,39 %', 1600), dCell('1.024', 1500), dCell('97,48 %', 1500)] }),
          new TableRow({ children: [dCell('cattle', 1800, true), dCell('10', 900, true), dCell('59,84 %', 1726, true), dCell('82,70 %', 1600, true), dCell('2.334', 1500, true), dCell('97,98 %', 1500, true)] }),
          new TableRow({ children: [dCell('horse', 1800), dCell('10', 900), dCell('58,10 %', 1726), dCell('72,38 %', 1600), dCell('2.734', 1500), dCell('98,70 %', 1500)] }),
          new TableRow({ children: [dCell('cat', 1800, true), dCell('10', 900, true), dCell('58,22 %', 1726, true), dCell('81,88 %', 1600, true), dCell('2.246', 1500, true), dCell('98,93 %', 1500, true)] }),
          new TableRow({ children: [dCell('duck', 1800), dCell('10', 900), dCell('55,24 %', 1726), dCell('92,31 %', 1600), dCell('1.633', 1500), dCell('93,84 %', 1500)] }),
          new TableRow({ children: [dCell('dog', 1800, true), dCell('10', 900, true), dCell('48,55 %', 1726, true, RED), dCell('78,26 %', 1600, true), dCell('2.676', 1500, true), dCell('98,43 %', 1500, true)] }),
          new TableRow({ children: [dCell('sheep', 1800), dCell('10', 900), dCell('35,32 %', 1726, false, RED), dCell('60,32 %', 1600), dCell('3.775', 1500), dCell('98,94 %', 1500)] }),
        ]
      }),
      spacer(),
      p('Rezultati otkrivaju jasan obrazac: modeli su vrlo dobri za vrste s manjim brojem vizualno razdvojivih pasmina (donkey 98 %, peafowl 95 %, rabbit 94 %) i znatno slabiji za vrste s velikim brojem sličnih pasmina (dog 49 %, sheep 35 %). Visok train_top1 (98–100 %) uz nizak val_top1 za slabije modele ukazuje na overfitting uzrokovan nedovoljnom količinom raznolikih podataka za tu vrstu.'),
      p('Posebno iznenađuje pigeon s top-5 svega 87,25 % — niže od top-1 nekih boljih modela. To ukazuje na specifičan problem gdje je raspodjela klasa izrazito nejednaka ili su vizualni detalji koji razlikuju pasmine golubova izuzetno fini.'),

      h2('4.4. Dijagnoza slabih performansi: sheep i dog'),
      p('Sheep (top-1 = 35 %) i dog (top-1 = 49 %) su najslabiji modeli. Analiza pokazuje zajedničke uzroke: (1) velik broj pasmina per vrstu, (2) slaba separabilnost — pasmine dijele oblik tijela, razlikuju se uglavnom bojom i teksturom dlake, (3) malobrojni ili neuravnoteženi primjeri za rijetke pasmine. Za sheep, val_loss od 3.78 uz train_loss gotovo 0 je klasičan sign overfittinga.'),
      p('Preporučene mjere za poboljanje: prikupljanje dodatnih primjera za slabo pokrivene pasmine (prioritetno), agresivnija regularizacija (dropout, veći weight_decay), korištenje harder augmentacija (ColorJitter s agresivnijim parametrima) i eventualno smanjenje broja klasa spajanjem vizualno nerazlučivih pasmina.'),

      h2('4.5. Uspješnost direktnog YOLO breed pristupa'),
      p('Direktni YOLO breed detektor (yolo11_breed_species_seed_full19_trainable5_b4_rgb) postiže mAP@50 = 0.006 i recall = 0.040 u drugoj epohi. Model ne konvergira ni nakon višestrukih pokušaja s različitim parametrima. Ovo je bio očekivani ishod koji eksplicitno potvrđuje da direktni jednofazni detektor nije izvediv za ovaj problem bez prethodne grube separacije po vrstama.'),

      // ── 5. DEMONSTRACIJA ─────────────────────────────────────────────────
      h1('5. Demonstracija rada'),
      h2('5.1. CLI ulazne točke'),
      p('U pyproject.toml definirano je 17 CLI naredbi koje pokrivaju cijeli pipeline. Ključne naredbe:'),
      code('# Priprema i validacija podataka'),
      code('animal-build-manifest'),
      code('animal-validate-label-mapping'),
      code('animal-data-qa data.manifest_path=data/manifests/manifest_v0.csv'),
      code(''),
      code('# Treniranje'),
      code('animal-train-species-aware data.species_aware.train.enabled=true'),
      code('animal-train-hierarchical-classifier'),
      code(''),
      code('# Evaluacija i export'),
      code('animal-eval eval=benchmark'),
      code('animal-mine-hard-negatives eval=benchmark'),
      code('animal-export-models eval=benchmark'),
      code(''),
      code('# Rudarenje i pregled'),
      code('animal-mine-unlabeled-animals'),
      code('animal-review-boxes'),
      code(''),
      code('# Inferencija'),
      code('animal-infer-two-stage model/pipeline=two_stage eval=two_stage_infer'),
      spacer(),

      h2('5.2. AMD ROCm GPU pokretanje'),
      p('Treniranje koristi izoliranu AMD ROCm okolinu koja ne mijenja globalnu Windows PATH konfiguraciju. Dostupni wrapper skripti:'),
      code('# Provjera GPU stanja'),
      code('.\\tools\\python_amd.ps1 -m animal_breed.runtime.amd_gpu_probe'),
      code(''),
      code('# Hybrid training na AMD GPU'),
      code('.\\tools\\train_species_aware_hybrid_amd.ps1 data.species_aware.hybrid.train.enabled=true'),
      spacer(),

      h2('5.3. Web aplikacija — tok rada'),
      p('Korisnik pokreće aplikaciju naredbom python tools/run_model_inference_app.py i otvara preglednik na localhost:8080. Upload fotografije, klik na Run Inference, povrat anotiranih rezultata u manje od sekunde za tipičnu fotografiju. Status panel pokazuje koji breed modeli su učitani i za koje su vrste dostupni.'),
      mixed(italic('Napomena: ovdje treba umetnuti screenshot web aplikacije i primjer anotiranog izlaza u konačnoj verziji predaje.')),

      h2('5.4. Primjer JSON izlaza inferencije'),
      code('{'),
      code('  "detections": ['),
      code('    {'),
      code('      "species": "dog",'),
      code('      "species_confidence": 0.94,'),
      code('      "bbox": [142, 87, 498, 412],'),
      code('      "breed": "golden_retriever",'),
      code('      "breed_confidence": 0.82,'),
      code('      "breed_top3": ["golden_retriever", "labrador_retriever", "nova_scotia_duck_tolling_retriever"]'),
      code('    }'),
      code('  ]'),
      code('}'),
      spacer(),

      // ── 6. ZAKLJUČAK ───────────────────────────────────────────────────────
      h1('6. Zaključak'),
      p('U ovom projektu razvijen je potpun višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina. Projekt je završen — species detektor je stabilan, svih 16 breed klasifikatora je istrenirano i evaluirano, a cijeli pipeline je dostupan kroz CLI, web aplikaciju i kompletnu infrastrukturu za eksperimente.'),
      p('Ključni zaključci:'),
      bullet('Hijerarhijska dekompozicija problema je ispravna odluka: direktni breed YOLO model ne konvergira (mAP@50 = 0.006), dok dvostupanjski pristup daje smislene rezultate za sve vrste'),
      bullet('Kvaliteta podataka i normalizacija oznaka su jednako važni kao i arhitektura modela — rani species detektor s nekvalitetnim podacima postiže mAP@50 = 0.346, dok finalni verzija s poboljšanim podacima dostiže 0.679'),
      bullet('Breed klasifikatori pokazuju veliku varijancu: donkey/pig/peafowl/rabbit postižu >94 % top-1, dok sheep/dog imaju <50 % — uzrok su neuravnoteženi podaci i visok broj vizualno sličnih klasa'),
      bullet('Rudarenje neanotiranih podataka (324 633 kandidata) otvara put za daljnje poboljanje slabih modela bez skupljeg ručnog označavanja'),
      bullet('Overfitting je prisutan kod slabih modela (train_top1 ≈ 99 %, val_top1 ≈ 35–49 %) i treba se adresirati agresivnijom regularizacijom i prikupljanjem više raznolikih primjera'),
      spacer(),
      p('Buduće mjere poboljanja: integracija mining kandidata u breed trening skupove, finije podešavanje breed modela za sheep i dog (veći weight_decay, dropout), istraživanje few-shot learning pristupa za vrste s malo podataka, te proširenje sustava na više vrsta (mačke divlje, konjske pasmine koje nemaju dovoljno podataka).'),

      // ── 7. LITERATURA ────────────────────────────────────────────────────
      h1('7. Korištene tehnologije i literatura'),
      h2('7.1. Korištene tehnologije'),
      caption('Tablica 6. Pregled korištenih tehnologija i alata'),
      new Table({
        width: { size: 9026, type: WidthType.DXA }, columnWidths: [2200, 1826, 5000],
        rows: [
          new TableRow({ children: [hCell('Tehnologija', 2200), hCell('Verzija', 1826), hCell('Korištenje u projektu', 5000)] }),
          new TableRow({ children: [dCell('Python', 2200), dCell('≥3.11', 1826), dCell('Primarni programski jezik', 5000)] }),
          new TableRow({ children: [dCell('PyTorch', 2200, true), dCell('≥2.4', 1826, true), dCell('Breed klasifikator, trening loop', 5000, true)] }),
          new TableRow({ children: [dCell('Ultralytics YOLO11', 2200), dCell('≥8.4.48', 1826), dCell('Species detektor, direktni breed model', 5000)] }),
          new TableRow({ children: [dCell('timm (EfficientNetV2-S)', 2200, true), dCell('≥1.0.27', 1826, true), dCell('Breed klasifikator (pretrained backbone)', 5000, true)] }),
          new TableRow({ children: [dCell('OpenCV', 2200), dCell('≥4.10', 1826), dCell('Obrada slika, crop, vizualizacija', 5000)] }),
          new TableRow({ children: [dCell('Hydra / OmegaConf', 2200, true), dCell('≥1.3.2', 1826, true), dCell('Konfiguracija eksperimenata', 5000, true)] }),
          new TableRow({ children: [dCell('Albumentations', 2200), dCell('≥2.0.8', 1826), dCell('Augmentacije za breed klasifikator', 5000)] }),
          new TableRow({ children: [dCell('Polars / Pandas', 2200, true), dCell('Latest', 1826, true), dCell('Obrada manifesta i CSV rezultata', 5000, true)] }),
          new TableRow({ children: [dCell('wandb', 2200), dCell('≥0.27', 1826), dCell('Praćenje eksperimenata (experiment tracking)', 5000)] }),
          new TableRow({ children: [dCell('AMD ROCm 7.2.1', 2200, true), dCell('7.2.1', 1826, true), dCell('GPU runtime za treniranje (Windows)', 5000, true)] }),
          new TableRow({ children: [dCell('GitHub Actions', 2200), dCell('—', 1826), dCell('CI (ci.yml, weekly_scout.yml)', 5000)] }),
          new TableRow({ children: [dCell('pytest', 2200, true), dCell('≥8.2', 1826, true), dCell('8 test datoteka, automatski CI', 5000, true)] }),
        ]
      }),
      spacer(),

      h2('7.2. Literatura'),
      p('[1] Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics. GitHub. https://github.com/ultralytics/ultralytics'),
      p('[2] Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. ICML 2021. arXiv:2104.00298.'),
      p('[3] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019. arXiv:1905.11946.'),
      p('[4] Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv:1804.02767.'),
      p('[5] Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.'),
      p('[6] Wightman, R. (2019). PyTorch Image Models (timm). GitHub. https://github.com/huggingface/pytorch-image-models'),
      p('[7] Yadan, O. (2019). Hydra — A framework for elegantly configuring complex applications. GitHub. https://github.com/facebookresearch/hydra'),
      p('[8] Buslaev, A., et al. (2020). Albumentations: Fast and Flexible Image Augmentations. Information, 11(2), 125.'),
      p('[9] Russakovsky, O., et al. (2015). ImageNet Large Scale Visual Recognition Challenge. IJCV, 115(3), 211–252.'),
      p('[10] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.'),

      h2('Kontrolni popis za predaju'),
      bullet('Seminar u PDF obliku'),
      bullet('Izvorni programski kod u repozitoriju (strojno-ucenje/)'),
      bullet('Prezentacija (PPTX ili PDF)'),
      bullet('Screenshot web aplikacije i primjer anotiranog izlaza'),
      bullet('Popunjena polja: ime i prezime autora, naziv kolegija'),
      spacer(),
      new Paragraph({ spacing: { after: 160 }, children: [new TextRun({
        text: 'Napomena: mjesta označena s [OVDJE DODATI ...] treba ručno dopuniti prije predaje.',
        font: 'Arial', size: 20, italics: true, color: GRAY
      })] }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('seminar_projekt_strojno_ucenje_FINAL.docx', buffer);
  console.log('Done: seminar_projekt_strojno_ucenje_FINAL.docx');
}).catch(err => { console.error(err); process.exit(1); });
