const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, TableOfContents,
  LevelFormat, ImageRun
} = require('docx');
const fs = require('fs');

// ─── palette: muted academic, not corporate-blue ─────────────────────────
const DARK   = '1C1C1C';   // near-black – headings
const MID    = '3D3D3D';   // dark grey – subheadings
const BODY   = '2B2B2B';   // body text
const GRAY   = '6B6B6B';   // captions / meta
const LGRAY  = 'F0EFED';   // alt row shading
const HROW   = 'DDD9D3';   // header row
const BORDER = 'B0AAA0';   // table border
const ACCENT = '5C4033';   // dark brown – sparingly
const FONT   = 'Calibri';
const MONO   = 'Courier New';

// ─── helpers ─────────────────────────────────────────────────────────────
const bdr1 = { style: BorderStyle.SINGLE, size: 3, color: BORDER };
const bdrAll = { top: bdr1, bottom: bdr1, left: bdr1, right: bdr1 };
const noBdr = { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
                left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE } };

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: HROW, space: 6 } },
    children: [new TextRun({ text, font: FONT, size: 30, bold: true, color: DARK })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 220, after: 80 },
    children: [new TextRun({ text, font: FONT, size: 24, bold: true, color: MID })]
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 140, line: 276, lineRule: 'auto' },
    children: [new TextRun({ text, font: FONT, size: 22, color: BODY, ...opts })]
  });
}
function run(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: 22, color: BODY, ...opts });
}
function mixed(...runs) {
  return new Paragraph({ spacing: { after: 140, line: 276, lineRule: 'auto' }, children: runs });
}
function bullet(text, opts = {}) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { after: 70 },
    children: [new TextRun({ text, font: FONT, size: 22, color: BODY, ...opts })]
  });
}
function code(text) {
  return new Paragraph({
    spacing: { after: 40, before: 40 },
    indent: { left: 560 },
    shading: { fill: 'F5F3F0', type: ShadingType.CLEAR },
    children: [new TextRun({ text, font: MONO, size: 19, color: '3A3030' })]
  });
}
function caption(text) {
  return new Paragraph({
    spacing: { after: 60, before: 120 },
    children: [new TextRun({ text, font: FONT, size: 19, italics: true, color: GRAY })]
  });
}
function pageBreak() { return new Paragraph({ children: [new PageBreak()] }); }
function spacer(n = 1) {
  return new Paragraph({ spacing: { after: 80 * n }, children: [new TextRun('')] });
}

// ─── table cells ─────────────────────────────────────────────────────────
function hCell(text, w) {
  return new TableCell({
    borders: bdrAll, width: { size: w, type: WidthType.DXA },
    shading: { fill: HROW, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 110, right: 110 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ children: [new TextRun({ text, font: FONT, size: 20, bold: true, color: DARK })] })]
  });
}
function dCell(text, w, shade = false, bold = false, color = BODY) {
  return new TableCell({
    borders: bdrAll, width: { size: w, type: WidthType.DXA },
    shading: { fill: shade ? LGRAY : 'FFFFFF', type: ShadingType.CLEAR },
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ children: [new TextRun({ text, font: FONT, size: 20, bold, color })] })]
  });
}

// ─── image helper ─────────────────────────────────────────────────────────
function img(path, w, h, description = '') {
  const ext = path.split('.').pop().toLowerCase();
  const type = ext === 'jpg' ? 'jpeg' : ext;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    children: [new ImageRun({
      type,
      data: fs.readFileSync(path),
      transformation: { width: w, height: h },
      altText: { title: description, description, name: description }
    })]
  });
}

// ─── build ────────────────────────────────────────────────────────────────
const BASE = 'C:\\Users\\User\\Desktop\\Strojno ucenje\\';
const PROJ = BASE + 'strojno-ucenje\\';

const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{ level: 0, format: LevelFormat.BULLET, text: '–', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 640, hanging: 320 } } } }]
    }]
  },
  styles: {
    default: { document: { run: { font: FONT, size: 22, color: BODY } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: DARK },
        paragraph: { spacing: { before: 320, after: 120 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: MID },
        paragraph: { spacing: { before: 220, after: 80 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 },
        margin: { top: 1280, right: 1280, bottom: 1280, left: 1500 } }
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        border: { bottom: { style: BorderStyle.SINGLE, size: 3, color: HROW, space: 4 } },
        children: [
          new TextRun({ text: 'Višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina', font: FONT, size: 17, color: GRAY }),
          new TextRun({ text: '\t2026.', font: FONT, size: 17, color: GRAY }),
        ],
        tabStops: [{ type: 'right', position: 8906 }]
      })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        border: { top: { style: BorderStyle.SINGLE, size: 3, color: HROW, space: 4 } },
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: '— ', font: FONT, size: 17, color: GRAY }),
          new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 17, color: GRAY }),
          new TextRun({ text: ' —', font: FONT, size: 17, color: GRAY }),
        ]
      })] })
    },
    children: [

      // ══ NASLOVNICA ═════════════════════════════════════════════════════
      spacer(4),
      new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 80 },
        children: [new TextRun({ text: 'Višerazinski sustav za prepoznavanje', font: FONT, size: 52, bold: true, color: DARK })] }),
      new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 300 },
        children: [new TextRun({ text: 'životinjskih vrsta i pasmina pomoću dubokog učenja', font: FONT, size: 52, bold: true, color: DARK })] }),
      new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 600 },
        children: [new TextRun({ text: 'Projekt iz strojnog učenja i računalnog vida', font: FONT, size: 24, italics: true, color: GRAY })] }),
      new Table({
        width: { size: 5800, type: WidthType.DXA },
        borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
                   left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
                   insideH: { style: BorderStyle.NONE }, insideV: { style: BorderStyle.NONE } },
        columnWidths: [1800, 4000],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders: noBdr, width: { size: 1800, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun({ text: 'Autor', font: FONT, size: 20, bold: true, color: GRAY })] })] }),
            new TableCell({ borders: noBdr, width: { size: 4000, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun({ text: '[OVDJE DODATI IME I PREZIME]', font: FONT, size: 20, color: BODY })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: noBdr, width: { size: 1800, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun({ text: 'Kolegij', font: FONT, size: 20, bold: true, color: GRAY })] })] }),
            new TableCell({ borders: noBdr, width: { size: 4000, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun({ text: '[OVDJE DODATI NAZIV KOLEGIJA]', font: FONT, size: 20, color: BODY })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: noBdr, width: { size: 1800, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun({ text: 'Tehnologije', font: FONT, size: 20, bold: true, color: GRAY })] })] }),
            new TableCell({ borders: noBdr, width: { size: 4000, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun({ text: 'Python · PyTorch · YOLO11 · EfficientNetV2-S · OpenCV · Hydra · wandb', font: FONT, size: 20, color: BODY })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: noBdr, width: { size: 1800, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun({ text: 'Godina', font: FONT, size: 20, bold: true, color: GRAY })] })] }),
            new TableCell({ borders: noBdr, width: { size: 4000, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun({ text: '2025./2026.', font: FONT, size: 20, color: BODY })] })] }),
          ]}),
        ]
      }),
      pageBreak(),

      // ══ SAŽETAK ════════════════════════════════════════════════════════
      h1('Sažetak'),
      p('Ovaj projekt bavi se automatskim prepoznavanjem životinjskih vrsta i pasmina na fotografijama. Sustav je podijeljen u dvije faze: YOLO11s detektor pronalazi vrstu i lokaciju životinje na slici, a zatim EfficientNetV2-S klasifikator dobiva izrezani isječak te određuje pasminu unutar te vrste.'),
      p('Projekt je potpuno završen. Species detektor postiže mAP@50 = 0.679 na finalnom skupu podataka, a breed klasifikatori su istrenirani za svih 16 vrsta s dovoljno podataka. Rezultati se kreću od odličnih (donkey 98,4 %, pig 96,7 %, peafowl 94,6 %) do problematičnih (sheep 35,3 %, dog 48,6 %), ovisno o broju dostupnih primjera i vizualnoj sličnosti pasmina.'),
      p('Kao kontrolni eksperiment, direktni YOLO breed model koji pokušava riješiti cijeli zadatak bez hijerarhije postiže mAP@50 = 0,006 i nije konvergirao — što eksplicitno potvrđuje opravdanost dvostupanjskog pristupa.'),
      p('Uz modele, projekt uključuje infrastrukturu za rudarenje neanotiranih primjera (324 633 kandidata), interaktivne alate za pregled bounding boxova i oznaka, normalizaciju oznaka između različitih dataseta, hard negative mining i lokalnu web aplikaciju za demonstraciju inferencije.'),
      mixed(run('Ključne riječi: ', { bold: true }), run('detekcija objekata, klasifikacija pasmina, YOLO11, EfficientNetV2, višerazinsko učenje, rudarenje podataka, računalni vid', { italics: true })),

      pageBreak(),
      new TableOfContents('Sadržaj', { hyperlink: true, headingStyleRange: '1-2' }),
      pageBreak(),

      // ══ 1. UVOD ════════════════════════════════════════════════════════
      h1('1. Uvod'),
      p('Prepoznavanje životinja na fotografijama je korisno u veterinarskoj dijagnostici, praćenju divljih životinja, upravljanju farmama i katalogizaciji fotografija. Problem nije trivijalan — nije dovoljno samo detektirati da se na slici nalazi životinja, nego razlikovati slične vrste i odrediti pasminu unutar iste vrste.'),
      p('Razlika između pasmina može biti izuzetno suptilna. Labrador i zlatni retriver razlikuju se uglavnom bojom dlake; border collie i australski ovčar sitnim detaljima koji se vide samo pri dobrom osvjetljenju. Uz to, podaci iz prakse su neuravnoteženi, heterogeni i dolaze iz različitih izvora s nekonzistentnim oznakama — to je jednako velik problem kao i sam model.'),
      p('Zbog toga je odabran višerazinski pristup koji problem razdvaja na dvije upravljive faze. Ova odluka bila je ispravna: direktni breed YOLO model koji pokušava riješiti cijeli zadatak odjednom nije konvergirao (mAP@50 = 0,006), dok dvostupanjski pristup daje smislene rezultate za sve istrenirane vrste.'),
      p('Ovaj rad dokumentira arhitekturu sustava, sve odluke u dizajnu pipeline-a podataka, stvarne metrike evaluacije za svaki istrenirani model i pomoćnu infrastrukturu koja čini projekt upotrebljivim izvan trening skripti.'),

      // ══ 2. PODACI ══════════════════════════════════════════════════════
      h1('2. Skup podataka, eksplorativna analiza i predobrada'),
      h2('2.1. Izvori podataka i kanonski manifest'),
      p('Projekt objedinjuje tri kategorije izvora podataka. Javno dostupni YOLO-formatirani skupovi pokrivaju opće kategorije životinja i čine osnovu species detektora. Ručno označeni primjeri sadrže pasminske oznake ali su malobrojniji. Sintetski generirani primjeri nastaju kompozitnom metodom: izrezane životinje postavljaju se na realistične pozadine kako bi se povećao broj primjera za rijetke kategorije.'),
      p('Svi podaci standardizirani su kroz kanonski manifest — tablicu u kojoj svaki redak predstavlja jednu sliku s atributima: image_id, putanja, izvor, canonical_species, canonical_breed (u obliku vrsta:pasmina), dimenzije, bounding box koordinate, split i quality_flags. Normalizacija oznaka između dataseta provodi se automatski (npr. "COW" → "cattle", "Holstein-Friesian breed" → "cattle:holstein_friesian") uz izvještaj o neprepoznatim oznakama za ručni pregled.'),

      h2('2.2. Statistike i neuravnoteženost klasa'),
      p('Glavni skup za detekciju vrsta sadrži 18 klasa i 27 433 slike (82 % trening / 12 % val / 6 % test). Neuravnoteženost između klasa vidljiva je na grafikonu ispod — razlika između najzastupljenije (cat: 4542 slika) i najmanje zastupljene klase (goose: 0, duck: 5) je dramatična.'),

      caption('Slika 1. Raspodjela primjera po vrstama u trening skupu — razlika između najzastupljenije i najmanje zastupljene klase premašuje 900:1.'),
      img(BASE + 'prezentacija_assets\\species_balance.png', 480, 270, 'Raspodjela vrsta'),
      spacer(),

      p('Ta neuravnoteženost ima direktan utjecaj na kvalitetu modela. Vrste poput duck, horse i goose bile su inicijalno gotovo nevidljive za detektor, što je zahtijevalo posebne mjere — kontrolirani sampling, sintetičke primjere i PHR (ručno označene) slike za najslabije zastupljene kategorije.'),

      caption('Tablica 1. Raspodjela skupa podataka'),
      new Table({
        width: { size: 8500, type: WidthType.DXA }, columnWidths: [2200, 1900, 1900, 2500],
        rows: [
          new TableRow({ children: [hCell('Skup', 2200), hCell('Broj slika', 1900), hCell('Udio', 1900), hCell('Napomena', 2500)] }),
          new TableRow({ children: [dCell('Trening', 2200), dCell('22 587', 1900), dCell('82,3 %', 1900), dCell('Za treniranje modela', 2500)] }),
          new TableRow({ children: [dCell('Validacija', 2200, true), dCell('3 375', 1900, true), dCell('12,3 %', 1900, true), dCell('Praćenje konvergencije', 2500, true)] }),
          new TableRow({ children: [dCell('Test', 2200), dCell('1 471', 1900), dCell('5,4 %', 1900), dCell('Konačna evaluacija', 2500)] }),
          new TableRow({ children: [dCell('Ukupno', 2200, true), dCell('27 433', 1900, true), dCell('100 %', 1900, true), dCell('18 klasa vrsta', 2500, true)] }),
        ]
      }),
      spacer(),

      h2('2.3. Eksplorativna analiza i normalizacija'),
      p('Analiza je otkrila nekoliko konkretnih problema. Nekonzistentne oznake: isti pas može biti označen kao "golden retriever", "Golden Retriever" ili "golden_retriever" u različitim izvorima — bez normalizacije, model bi to tretirao kao tri klase. Premali bounding boxevi: sintetički generirani boxovi manji od 20–25 piksela su filtrirani jer ne sadrže dovoljno vizualne informacije. Loša kvaliteta slike: dio fotografija je preeksponiran, presitan objekt ili ima jak šum.'),
      p('Rješenje je sustav normalizacije oznaka (cross_dataset_label_map.yaml) koji automatski prevodi varijante imena u kanonske identifikatore, s izvještajem o neuparenim oznakama za ručni pregled. Alat animal-data-qa provodi automatizirani quality gate koji blokira pokretanje treninga ako podaci ne prolaze definirane pragove.'),

      h2('2.4. Predobrada, sintetički kompoziti i augmentacije'),
      p('Za detekciju, slike se konvertiraju u YOLO format s normaliziranim koordinatama. Za breed klasifikaciju koristi se crop na temelju detekcije uz marginski pomak od 10–15 % kako bi se uključio vizualni kontekst — bez toga model bi dobivao agresivno izrezane isječke koji ponekad zasjeku glavu ili rep životinje.'),

      p('Ključna mjera za suzbijanje neuravnoteženosti klasa su sintetički kompoziti, generirani sustavom HybridSpeciesYoloDataset. Proces je sljedeći: iz trening slika izrezuju se pojedinačne životinje (koristeći postojeće YOLO oznake), a zatim se više takvih izrezaka različitih vrsta postavlja na novu pozadinsku teksturu u raspoređenom gridu s doziranim šumom (jitter ±6 %, skaliranje 0,88–1,20×). Bounding boxovi se automatski generiraju za svaki postavljeni izrezak. Time nastaju slike koje model nikad nije vidio, a koje sadrže baš one kombinacije vrsta koje su rijetke u originalnim podacima.'),
      p('Omjer uzorkovanja: 40 % sintetički kompoziti, 40 % originalne slike, 20 % jako augmentirani pojedinačni uzorci. Prag minimalne veličine boxa je 15–20 piksela — manji se odbacuju jer ne nose dovoljno vizualne informacije.'),

      caption('Slika 2. Primjeri izvornih izrezaka životinja (gore) koji se koriste kao ulaz u sintetički compositor. Dolje: rezultat kompozicije — duck, horse, pig i sheep postavljeni na stvarnu pozadinsku teksturu iz projekta, s automatski generiranim bounding boxovima.'),
      img(BASE + 'synthetic_source_crops.jpg', 440, 115, 'Izvorni izresci'),
      img(BASE + 'synthetic_composite_example.jpg', 340, 340, 'Sinteticki kompozit'),
      spacer(),
      p('Augmentacije primijenjene uz sintetičke slike: horizontalni/vertikalni flip, rotacija ±15°, promjena ekspozicije ±22 %, Gaussov šum (σ = 3 % vrijednosti piksela). Za breed klasifikator dodatno se koriste MixUp (α=0,2) i CutMix (α=1,0). Za YOLO detektor mosaic augmentacija (kombinira 4 slike) gasi se 10 epoha prije kraja kako se ne bi poremetila fina konvergencija.'),

      // ══ 3. METODE ══════════════════════════════════════════════════════
      h1('3. Korištene metode i algoritmi'),
      h2('3.1. YOLO11s species detektor'),
      p('Prva faza koristi YOLO11s — manju varijantu iz generacije YOLO11 tvrtke Ultralytics. YOLO u jednom prolazu kroz sliku predviđa sve bounding boxove i klase, što ga čini brzim i pogodnim za interaktivne primjene. Za razliku od dvodijelnih detektora (poput Faster R-CNN), nema zasebne faze predlaganja regija.'),
      p('Treniranje koristi hibridni dataset (HybridSpeciesYoloDataset) koji kombinira više YOLO izvora s kontroliranim uzorkovanjem po vrstama — ključna razlika od naivnog spajanja, koja sprječava da dominantne klase potisnu rijetke. Hardware je AMD GPU s ROCm 7.2.1 runtimeom pokrenutim kroz izoliranu Python 3.12 embed okolinu.'),

      caption('Tablica 2. Ključni hiperparametri YOLO11s species detektora'),
      new Table({
        width: { size: 8500, type: WidthType.DXA }, columnWidths: [3200, 1800, 3500],
        rows: [
          new TableRow({ children: [hCell('Parametar', 3200), hCell('Vrijednost', 1800), hCell('Napomena', 3500)] }),
          new TableRow({ children: [dCell('Bazni model', 3200), dCell('yolo11s.pt', 1800), dCell('Pretrained na COCO', 3500)] }),
          new TableRow({ children: [dCell('Rezolucija', 3200, true), dCell('640 × 640 px', 1800, true), dCell('Standard za YOLO', 3500, true)] }),
          new TableRow({ children: [dCell('Epohe / patience', 3200), dCell('100 / 25', 1800), dCell('Early stopping', 3500)] }),
          new TableRow({ children: [dCell('Batch / workers', 3200, true), dCell('16 / 8', 1800, true), dCell('', 3500, true)] }),
          new TableRow({ children: [dCell('lr0 / raspored', 3200), dCell('0.01 / cosine', 1800), dCell('cos_lr = true', 3500)] }),
          new TableRow({ children: [dCell('Mosaic / AMP', 3200, true), dCell('close=10 / true', 1800, true), dCell('Gasi se 10 ep. prije kraja', 3500, true)] }),
        ]
      }),
      spacer(),

      h2('3.2. EfficientNetV2-S breed klasifikator'),
      p('Drugi stupanj koristi EfficientNetV2-S, specifično varijantu tf_efficientnetv2_s.in21k_ft_in1k iz timm biblioteke. Ova varijanta je pretrenirana na ImageNet-21k (14 milijuna slika, 21 000 klasa) s daljnjim finim podešavanjem na ImageNet-1k — što joj daje iznimno bogatu inicijalizaciju za fine-grained klasifikaciju detalja poput teksture dlake.'),
      p('Trening je modularan: za svaku vrstu s dovoljno podataka trenira se zaseban model. Time broj klasa ostaje pod kontrolom (npr. 15–30 pasmina pasa umjesto 200+ zajedno). Ulaz nije cijela fotografija nego crop iz prve faze, što eliminira utjecaj pozadine.'),

      caption('Tablica 3. Ključni hiperparametri EfficientNetV2-S breed klasifikatora'),
      new Table({
        width: { size: 8500, type: WidthType.DXA }, columnWidths: [3200, 1800, 3500],
        rows: [
          new TableRow({ children: [hCell('Parametar', 3200), hCell('Vrijednost', 1800), hCell('Napomena', 3500)] }),
          new TableRow({ children: [dCell('Bazni model', 3200), dCell('tf_efficientnetv2_s.in21k_ft_in1k', 1800), dCell('ImageNet-21k → IN-1k', 3500)] }),
          new TableRow({ children: [dCell('Rezolucija', 3200, true), dCell('384 × 384 px', 1800, true), dCell('Nativna rezolucija V2-S', 3500, true)] }),
          new TableRow({ children: [dCell('Epohe / patience', 3200), dCell('60 / 10', 1800), dCell('Min. poboljanje 0.001', 3500)] }),
          new TableRow({ children: [dCell('Batch / optimizer', 3200, true), dCell('48 / AdamW', 1800, true), dCell('Bolja generalizacija od Adam', 3500, true)] }),
          new TableRow({ children: [dCell('lr / weight decay', 3200), dCell('0.0003 / 0.05', 1800), dCell('L2 regularizacija', 3500)] }),
          new TableRow({ children: [dCell('Label smoothing', 3200, true), dCell('0.1', 1800, true), dCell('Smanjuje overconfidence', 3500, true)] }),
          new TableRow({ children: [dCell('MixUp / CutMix α', 3200), dCell('0.2 / 1.0', 1800), dCell('Augmentacije raznolikosti', 3500)] }),
        ]
      }),
      spacer(),

      h2('3.3. Dvostupanjski pipeline'),
      p('Integracija obje faze (src/animal_breed/pipelines/infer_two_stage.py) slijedi ovaj tok: fotografija ulazi u YOLO species detektor → za svaki okvir s pouzdanošću > 0,25 vrši se crop s marginom → crop odlazi u breed klasifikator za tu vrstu → ako model za vrstu nije dostupan, vraća se samo species predikcija → izlaz je JSON s detaljima i anotirana slika.'),

      h2('3.4. Direktni YOLO breed model (kontrolni eksperiment)'),
      p('Kao kontrolni eksperiment implementiran je YOLO breed model koji pokušava direktno prepoznati pasminu bez prethodne faze detekcije vrste — jedan model, jedan korak, ali s 100+ klasa odjednom.'),
      mixed(run('Rezultat: ', { bold: true }), run('model postiže mAP@50 = 0,006 u drugoj epohi i ne konvergira. S tako velikim brojem vizualno sličnih klasa bez ikakve hijerarhijske strukture, YOLO ne može inicijalizirati smislene klasifikacijske glave. Ovaj negativni rezultat je vrijedan jer eksplicitno opravdava hijerarhijski dizajn.')),
      spacer(),

      h2('3.5. Rudarenje neanotiranih životinja'),
      p('Uz treniranje modela, projekt uključuje pipeline za rudarenje neanotiranih životinja iz dostupnih YOLO dataseta (src/animal_breed/mining/unlabeled_animals.py). Ideja je jednostavna: istrenirani species detektor pokreće se na svim dostupnim slikama — uključujući one za koje postoje YOLO oznake za neke klase ali ne sve. Detekcije koje se ne podudaraju s postojećim oznakama postaju "kandidati" — vjerojatno propuštene ili krive oznake.'),
      p('Strategija uzorkovanja ravnomjerno raspoređuje budžet između izvora (ne troši se samo na abecedni prvi dataset), uz opcionalnu TTA konzistentnost provjeru (flip-TTA). Rezultati se pregledaju kroz box review alat i nikad ne mijenjaju originalne oznake automatski.'),
      p('Dosadašnji rezultati: 324 633 ukupnih kandidata, 20 000 visokoprioritnih. Primjeri ispod otkrivaju zanimljive slučajeve — goveda označena kao "horse", konj koji nedostaje u anotaciji, i mačka pogrešno klasificirana kao "rabbit".'),

      caption('Slika 3. Primjeri iz rudarenja — lijevo: detektor pronalazi goveda i označava ih kao "horse" (cross-species konfuzija na višestrukim životinjama); sredina: konj koji nije bio anotiran u originalnom datasetu; desno: mačka pogrešno klasificirana kao "rabbit:0.27" — otkriva granice modela ali i potencijalno propuštenu anotaciju.'),
      img(BASE + 'mining_samples_panel.jpg', 460, 152, 'Mining primjeri panel'),
      spacer(),

      h2('3.6. Interaktivni alati za pregled i reviziju podataka'),
      p('Projekt uključuje dva interaktivna web alata koja su ključna za praktičan rad s podacima. Box review app prikazuje predložene bounding boxove iz mininga i omogućuje korisniku da potvrdi, odbaci ili prilagodi svaki prijedlog — pregledi se čuvaju zasebno, nikad ne mijenjajući originalne oznake. Label audit app prikazuje sliku i pridijeljenu oznaku za brzu identifikaciju pogrešnih ili nekonzistentnih anotacija.'),

      caption('Slika 4. Label audit app — lijevo lista nerecenziranih uzoraka po kategoriji, desno kontaktni list primjera i alat za donošenje odluke. Na slici: pregled kameljih pasminskih oznaka.'),
      img(BASE + 'label-audit-app.png', 420, 220, 'Label audit app'),
      spacer(),
      caption('Slika 5. Box review app — lista kandidata iz mininga s pouzdanošću i statusom (lijevo), panel za pregled i korekciju koordinata okvira (desno). Kandidat je označen kao "ambiguous" — detektor je pronašao ovcu, ali izvorna oznaka na slici je "cattle".'),
      img(BASE + 'selenium-box-review-4.png', 420, 230, 'Box review app'),
      spacer(),

      h2('3.7. Web aplikacija za zaključivanje'),
      p('Lokalna web aplikacija (src/animal_breed/webapp/model_inference_app.py) izlaže REST API i statičko sučelje za demonstraciju. Korisnik uploadira fotografiju, aplikacija pokreće puni pipeline i vraća anotiranu sliku s okvirima i tablicom predikcija. Status panel prikazuje koji breed modeli su učitani i za koje su vrste dostupni.'),

      // ══ 4. EVALUACIJA ══════════════════════════════════════════════════
      h1('4. Evaluacija modela'),
      h2('4.1. Metrike'),
      p('Za YOLO detektor: Precision (udio ispravnih od svih predviđenih detekcija), Recall (udio pronađenih od svih stvarnih objekata), mAP@50 (srednja prosječna preciznost pri IoU=0,50) i mAP@50–95 (striktnija mjera koja kažnjava neprecizne okvire).'),
      p('Za breed klasifikator: Top-1 točnost (točna klasa na prvom mjestu) i Top-5 točnost (točna klasa u prvih 5 prijedloga — relevantno za praktične primjene s mnogo vizualno sličnih klasa).'),

      h2('4.2. Species detektor — iteracije treniranja'),
      p('Provedeno je više iteracija s postupnim poboljanjima u kvaliteti podataka. Napredak između ranih i finalnih eksperimenata pokazuje koliko je priprema podataka važna — rani model (full17_v2) postiže mAP@50 = 0,346, dok finalni dostiže 0,679.'),

      caption('Slika 6. Usporedba finalnih metrika između osnovnog i poboljšanog eksperimenta species detektora.'),
      img(BASE + 'prezentacija_assets\\metrics_compare.png', 440, 220, 'Usporedba metrika'),
      spacer(),

      caption('Slika 7. Trend metrika kroz zadnjih 5 epoha finalnog hybrid species run-a — konvergencija je stabilna, mAP50 i recall rastu monotono.'),
      img(BASE + 'prezentacija_assets\\hybrid_trend.png', 440, 220, 'Trend metrika'),
      spacer(),

      caption('Tablica 4. Usporedba iteracija species detektora'),
      new Table({
        width: { size: 8500, type: WidthType.DXA }, columnWidths: [3200, 1325, 1325, 1325, 1325],
        rows: [
          new TableRow({ children: [hCell('Eksperiment', 3200), hCell('P', 1325), hCell('R', 1325), hCell('mAP@50', 1325), hCell('mAP@50–95', 1325)] }),
          new TableRow({ children: [dCell('hybrid_shared_full17_v2 (rana verzija)', 3200), dCell('0.361', 1325), dCell('0.366', 1325), dCell('0.346', 1325), dCell('0.163', 1325)] }),
          new TableRow({ children: [dCell('hybrid_shared_full18_goose_10more', 3200, true), dCell('0.615', 1325, true), dCell('0.616', 1325, true), dCell('0.617', 1325, true), dCell('0.324', 1325, true)] }),
          new TableRow({ children: [dCell('hybrid_shared_full18_goose_5more_box25', 3200), dCell('0.650', 1325), dCell('0.659', 1325), dCell('0.666', 1325), dCell('0.351', 1325)] }),
          new TableRow({ children: [dCell('hybrid_shared_full18_phr_5more_box25 (final)', 3200, true), dCell('0.644', 1325, true), dCell('0.685', 1325, true), dCell('0.679', 1325, true, true, ACCENT), dCell('0.365', 1325, true)] }),
        ]
      }),
      spacer(),

      h2('4.3. Breed klasifikatori — svi modeli'),
      p('Breed klasifikatori su istrenirani za svih 16 vrsta. Sljedeći grafikon prikazuje finalnu top-1 točnost, a tablica detaljne metrike.'),

      caption('Slika 8. Top-1 točnost breed klasifikatora po vrstama. Zeleno ≥ 80 %, žuto/smeđe 55–80 %, crveno < 55 %.'),
      img(BASE + 'chart_breed_accuracy.png', 470, 280, 'Breed accuracy po vrstama'),
      spacer(),

      caption('Tablica 5. Finalne metrike breed klasifikatora po vrstama (zadnja epoha, early stopping patience=10)'),
      new Table({
        width: { size: 8500, type: WidthType.DXA }, columnWidths: [1700, 800, 1600, 1400, 1400, 1600],
        rows: [
          new TableRow({ children: [hCell('Vrsta', 1700), hCell('Ep.', 800), hCell('val_top1', 1600), hCell('val_top5', 1400), hCell('val_loss', 1400), hCell('train_top1', 1600)] }),
          new TableRow({ children: [dCell('pig', 1700), dCell('10', 800), dCell('96,74 %', 1600, false, true, '375623'), dCell('99,94 %', 1400), dCell('0.596', 1400), dCell('99,95 %', 1600)] }),
          new TableRow({ children: [dCell('donkey', 1700, true), dCell('10', 800, true), dCell('98,35 %', 1600, true, true, '375623'), dCell('98,77 %', 1400, true), dCell('0.667', 1400, true), dCell('99,95 %', 1600, true)] }),
          new TableRow({ children: [dCell('peafowl', 1700), dCell('10', 800), dCell('94,57 %', 1600, false, true, '375623'), dCell('98,45 %', 1400), dCell('0.637', 1400), dCell('99,59 %', 1600)] }),
          new TableRow({ children: [dCell('rabbit', 1700, true), dCell('10', 800, true), dCell('93,75 %', 1600, true), dCell('100,00 %', 1400, true), dCell('0.841', 1400, true), dCell('100,00 %', 1600, true)] }),
          new TableRow({ children: [dCell('chicken', 1700), dCell('10', 800), dCell('89,59 %', 1600), dCell('95,26 %', 1400), dCell('1.325', 1400), dCell('99,63 %', 1600)] }),
          new TableRow({ children: [dCell('pigeon', 1700, true), dCell('10', 800, true), dCell('86,76 %', 1600, true), dCell('87,25 %', 1400, true), dCell('1.473', 1400, true), dCell('99,61 %', 1600, true)] }),
          new TableRow({ children: [dCell('goat', 1700), dCell('10', 800), dCell('86,10 %', 1600), dCell('98,93 %', 1400), dCell('0.992', 1400), dCell('99,88 %', 1600)] }),
          new TableRow({ children: [dCell('goose', 1700, true), dCell('10', 800, true), dCell('78,41 %', 1600, true), dCell('92,05 %', 1400, true), dCell('1.427', 1400, true), dCell('99,50 %', 1600, true)] }),
          new TableRow({ children: [dCell('dromedary', 1700), dCell('10', 800), dCell('77,84 %', 1600), dCell('96,39 %', 1400), dCell('1.024', 1400), dCell('97,48 %', 1600)] }),
          new TableRow({ children: [dCell('cattle', 1700, true), dCell('10', 800, true), dCell('59,84 %', 1600, true), dCell('82,70 %', 1400, true), dCell('2.334', 1400, true), dCell('97,98 %', 1600, true)] }),
          new TableRow({ children: [dCell('horse', 1700), dCell('10', 800), dCell('58,10 %', 1600), dCell('72,38 %', 1400), dCell('2.734', 1400), dCell('98,70 %', 1600)] }),
          new TableRow({ children: [dCell('cat', 1700, true), dCell('10', 800, true), dCell('58,22 %', 1600, true), dCell('81,88 %', 1400, true), dCell('2.246', 1400, true), dCell('98,93 %', 1600, true)] }),
          new TableRow({ children: [dCell('duck', 1700), dCell('10', 800), dCell('55,24 %', 1600), dCell('92,31 %', 1400), dCell('1.633', 1400), dCell('93,84 %', 1600)] }),
          new TableRow({ children: [dCell('dog', 1700, true), dCell('10', 800, true), dCell('48,55 %', 1600, true, true, '8B3A3A'), dCell('78,26 %', 1400, true), dCell('2.676', 1400, true), dCell('98,43 %', 1600, true)] }),
          new TableRow({ children: [dCell('sheep', 1700), dCell('10', 800), dCell('35,32 %', 1600, false, true, '8B3A3A'), dCell('60,32 %', 1400), dCell('3.775', 1400), dCell('98,94 %', 1600)] }),
        ]
      }),
      spacer(),

      h2('4.4. Analiza konvergencije i overfittinga'),
      p('Grafikon ispod ilustrira temeljni problem slabijih modela. Donkey konvergira odlično — train i val loss padaju zajedno i stabiliziraju se. Dog pak pokazuje klasičan overfitting: train loss pada prema nuli dok val loss stagnira na visokim vrijednostima, što znači da model pamti trening primjere umjesto da generalizira.'),

      caption('Slika 9. Lijevo: konvergencija val_top1 kroz epohe za 4 vrste — donkey i chicken konvergiraju brzo i stabilno, dok dog i sheep osciliraju i plateauiraju na nižim vrijednostima. Desno: train vs. val loss za dog (overfitting) i donkey (dobra generalizacija).'),
      img(BASE + 'chart_training_curves.png', 470, 190, 'Training curves'),
      spacer(),
      p('Uzrok overfittinga kod dog i sheep je kombinacija: velik broj pasmina po vrsti, vizualna sličnost između klasa i nedovoljan broj raznolikih primjera za rijetke pasmine. Rješenje nije složeniji model nego više podataka — upravo zato postoji mining pipeline.'),

      // ══ 5. DEMONSTRACIJA ═══════════════════════════════════════════════
      h1('5. Demonstracija rada'),
      h2('5.1. Pokretanje i CLI'),
      p('Projekt ima 17 registriranih CLI naredbi u pyproject.toml. Tipičan tijek:'),
      code('# 1. Priprema podataka'),
      code('animal-build-manifest'),
      code('animal-data-qa data.manifest_path=data/manifests/manifest_v0.csv'),
      code(''),
      code('# 2. Treniranje'),
      code('animal-train-species-aware data.species_aware.train.enabled=true'),
      code('animal-train-hierarchical-classifier'),
      code(''),
      code('# 3. Evaluacija i export'),
      code('animal-eval eval=benchmark'),
      code('animal-export-models eval=benchmark'),
      code(''),
      code('# 4. Rudarenje i pregled'),
      code('animal-mine-unlabeled-animals'),
      code('animal-review-boxes'),
      code(''),
      code('# 5. Inferencija'),
      code('animal-infer-two-stage model/pipeline=two_stage eval=two_stage_infer'),
      spacer(),

      h2('5.2. Primjer JSON izlaza inferencije'),
      code('{'),
      code('  "detections": [{'),
      code('    "species": "dog",  "species_confidence": 0.94,'),
      code('    "bbox": [142, 87, 498, 412],'),
      code('    "breed": "golden_retriever",  "breed_confidence": 0.82,'),
      code('    "breed_top3": ["golden_retriever", "labrador_retriever", "nova_scotia_duck_tolling_retriever"]'),
      code('  }]'),
      code('}'),
      spacer(),

      // ══ 6. ZAKLJUČAK ═══════════════════════════════════════════════════
      h1('6. Zaključak'),
      p('Projekt je potpuno završen. Species detektor je stabilan (mAP@50 = 0,679), svih 16 breed klasifikatora je istrenirano i evaluirano, direktni breed YOLO eksperiment je dao jasan negativni rezultat koji opravdava hijerarhijski dizajn, a cijela infrastruktura (mining, review alati, CI, web app) je operativna.'),
      p('Ključni zaključci:'),
      bullet('Hijerarhijska dekompozicija je ispravna odluka — direktni breed YOLO ne konvergira (mAP = 0,006), dvostupanjski pristup funkcionira za sve vrste'),
      bullet('Kvaliteta podataka je jednako važna kao arhitektura — rani detektor s lošim podacima daje mAP = 0,346, finalni s poboljšanim podacima daje 0,679'),
      bullet('9 od 15 breed modela postiže ≥ 78 % top-1 točnosti, što je korisna razina za praktičnu primjenu'),
      bullet('Dog (48,6 %) i sheep (35,3 %) pate od overfittinga uzrokovanog neuravnoteženim i nedovoljno raznolikim podacima — mining pipeline je direktan odgovor na ovaj problem'),
      bullet('Top-5 točnost je visoka za gotovo sve vrste (pig 99,9 %, rabbit 100 %, duck 92 %) — model zna odgovor, samo ne uvijek s dovoljnom sigurnošću na prvom mjestu'),
      spacer(),
      p('Buduće mjere: integracija mining kandidata u breed trening skupove (posebno za sheep i dog), istraživanje few-shot learning za vrste s malo podataka, finije podešavanje slabih modela s agresivnijom regularizacijom.'),

      // ══ 7. LITERATURA ══════════════════════════════════════════════════
      h1('7. Korištene tehnologije i literatura'),
      h2('7.1. Tehnologije'),
      caption('Tablica 6. Pregled korištenih tehnologija'),
      new Table({
        width: { size: 8500, type: WidthType.DXA }, columnWidths: [2000, 1500, 5000],
        rows: [
          new TableRow({ children: [hCell('Tehnologija', 2000), hCell('Verzija', 1500), hCell('Korištenje', 5000)] }),
          new TableRow({ children: [dCell('Python', 2000), dCell('≥ 3.11', 1500), dCell('Primarni programski jezik', 5000)] }),
          new TableRow({ children: [dCell('PyTorch', 2000, true), dCell('≥ 2.4', 1500, true), dCell('Breed klasifikator, trening loop', 5000, true)] }),
          new TableRow({ children: [dCell('Ultralytics YOLO11', 2000), dCell('≥ 8.4.48', 1500), dCell('Species detektor i direktni breed model', 5000)] }),
          new TableRow({ children: [dCell('timm (EfficientNetV2-S)', 2000, true), dCell('≥ 1.0.27', 1500, true), dCell('Breed klasifikator — pretrained backbone', 5000, true)] }),
          new TableRow({ children: [dCell('OpenCV', 2000), dCell('≥ 4.10', 1500), dCell('Obrada slika, crop, vizualizacija', 5000)] }),
          new TableRow({ children: [dCell('Hydra / OmegaConf', 2000, true), dCell('≥ 1.3.2', 1500, true), dCell('Konfiguracija eksperimenata', 5000, true)] }),
          new TableRow({ children: [dCell('Albumentations', 2000), dCell('≥ 2.0.8', 1500), dCell('Augmentacije za breed klasifikator', 5000)] }),
          new TableRow({ children: [dCell('wandb', 2000, true), dCell('≥ 0.27', 1500, true), dCell('Praćenje eksperimenata (experiment tracking)', 5000, true)] }),
          new TableRow({ children: [dCell('AMD ROCm', 2000), dCell('7.2.1', 1500), dCell('GPU runtime za treniranje (Windows)', 5000)] }),
          new TableRow({ children: [dCell('pytest / GitHub Actions', 2000, true), dCell('≥ 8.2', 1500, true), dCell('8 test datoteka, automatski CI', 5000, true)] }),
        ]
      }),
      spacer(),

      h2('7.2. Literatura'),
      p('[1] Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics. GitHub: github.com/ultralytics/ultralytics'),
      p('[2] Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. ICML 2021. arXiv:2104.00298.'),
      p('[3] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML 2019. arXiv:1905.11946.'),
      p('[4] Paszke, A. et al. (2019). PyTorch: An Imperative Style Deep Learning Library. NeurIPS 2019.'),
      p('[5] Wightman, R. (2019). PyTorch Image Models (timm). GitHub: github.com/huggingface/pytorch-image-models'),
      p('[6] Yadan, O. (2019). Hydra — A framework for elegantly configuring complex applications. GitHub: github.com/facebookresearch/hydra'),
      p('[7] Buslaev, A. et al. (2020). Albumentations: Fast and Flexible Image Augmentations. Information, 11(2), 125.'),
      p('[8] Russakovsky, O. et al. (2015). ImageNet Large Scale Visual Recognition Challenge. IJCV, 115(3), 211–252.'),

      h2('Kontrolni popis za predaju'),
      bullet('Seminar u PDF obliku'),
      bullet('Izvorni programski kod u repozitoriju (strojno-ucenje/)'),
      bullet('Prezentacija (PPTX ili PDF)'),
      bullet('Popunjena polja: ime i prezime autora, naziv kolegija'),
      spacer(),
      p('[OVDJE DODATI IME I PREZIME] i [OVDJE DODATI NAZIV KOLEGIJA] treba ručno popuniti prije predaje.', { italics: true, color: GRAY }),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('seminar_projekt_strojno_ucenje_FINAL_v2.docx', buf);
  console.log('Done: seminar_projekt_strojno_ucenje_FINAL_v2.docx');
}).catch(e => { console.error(e); process.exit(1); });
