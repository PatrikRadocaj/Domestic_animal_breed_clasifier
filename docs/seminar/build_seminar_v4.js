const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, TableOfContents,
  LevelFormat, ImageRun, UnderlineType
} = require('docx');
const fs = require('fs');

// ─── style tokens — Times New Roman, black, academic ─────────────────────
const F    = 'Times New Roman';
const MONO = 'Courier New';
const BLK  = '000000';
const DGRY = '333333';
const GRAY = '666666';
const PT12 = 24;  // half-points
const PT11 = 22;
const PT10 = 20;
const PT14 = 28;
const PT16 = 32;
const PT18 = 36;
const PT24 = 48;
const LINE15 = 360;  // 1.5× line spacing in twips

const thinBdr = { style: BorderStyle.SINGLE, size: 4, color: BLK };
const tblBdr  = { top: thinBdr, bottom: thinBdr, left: thinBdr, right: thinBdr };
const noBdr   = { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
                  left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE } };

// ─── helpers ─────────────────────────────────────────────────────────────
function para(children, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: LINE15, lineRule: 'auto' },
    alignment: AlignmentType.JUSTIFIED,
    ...opts,
    children,
  });
}
function p(text, opts = {}) {
  return para([new TextRun({ text, font: F, size: PT12, color: BLK, ...opts })]);
}
function h1(num, text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 280, after: 140, line: LINE15, lineRule: 'auto' },
    children: [new TextRun({ text: `${num}. ${text}`, font: F, size: PT14, bold: true, color: BLK })]
  });
}
function h2(num, text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 200, after: 100, line: LINE15, lineRule: 'auto' },
    children: [new TextRun({ text: `${num} ${text}`, font: F, size: PT12, bold: true, color: BLK })]
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 100, before: 60 },
    children: [new TextRun({ text, font: F, size: PT10, italics: true, color: BLK })]
  });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { after: 60, line: LINE15, lineRule: 'auto' },
    children: [new TextRun({ text, font: F, size: PT12, color: BLK })]
  });
}
function code(text) {
  return new Paragraph({
    spacing: { after: 40, before: 40 },
    indent: { left: 720 },
    children: [new TextRun({ text, font: MONO, size: 20, color: BLK })]
  });
}
function pageBreak() { return new Paragraph({ children: [new PageBreak()] }); }
function spacer()    { return new Paragraph({ spacing: { after: 120 }, children: [new TextRun('')] }); }
function centeredBold(text, size = PT12) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
    children: [new TextRun({ text, font: F, size, bold: true, color: BLK })]
  });
}
function centered(text, size = PT12) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
    children: [new TextRun({ text, font: F, size, color: BLK })]
  });
}

// ─── table cells ─────────────────────────────────────────────────────────
function hCell(text, w) {
  return new TableCell({
    borders: tblBdr, width: { size: w, type: WidthType.DXA },
    shading: { fill: 'FFFFFF', type: ShadingType.CLEAR },
    margins: { top: 70, bottom: 70, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, font: F, size: PT11, bold: true, color: BLK })]
    })]
  });
}
function dCell(text, w, center = false, bold = false) {
  return new TableCell({
    borders: tblBdr, width: { size: w, type: WidthType.DXA },
    shading: { fill: 'FFFFFF', type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text, font: F, size: PT11, bold, color: BLK })]
    })]
  });
}

// ─── image helper ─────────────────────────────────────────────────────────
function img(path, w, h, desc = '') {
  const ext = path.split('.').pop().toLowerCase();
  const type = ext === 'jpg' ? 'jpeg' : ext;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 40 },
    children: [new ImageRun({
      type, data: fs.readFileSync(path),
      transformation: { width: w, height: h },
      altText: { title: desc, description: desc, name: desc }
    })]
  });
}

const BASE = 'C:\\Users\\User\\Desktop\\Strojno ucenje\\';
const PROJ = BASE + 'strojno-ucenje\\';

// ─── document ─────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{ level: 0, format: LevelFormat.BULLET, text: '•',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
    }]
  },
  styles: {
    default: {
      document: { run: { font: F, size: PT12, color: BLK } }
    },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: F, size: PT14, bold: true, color: BLK },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: F, size: PT12, bold: true, color: BLK },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1418, right: 1418, bottom: 1418, left: 1701 }  // 2.5cm / 3cm left
      }
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: F, size: PT10, color: BLK })]
      })] })
    },
    children: [

      // ══ NASLOVNICA ═══════════════════════════════════════════════════════
      spacer(), spacer(), spacer(),
      centeredBold('SVEUČILIŠTE JOSIPA JURJA STROSSMAYERA U OSIJEKU', PT12),
      centeredBold('FAKULTET ELEKTROTEHNIKE, RAČUNARSTVA I', PT12),
      centeredBold('INFORMACIJSKIH TEHNOLOGIJA OSIJEK', PT12),
      spacer(), spacer(),
      centeredBold('Diplomski studij Računarstvo', PT12),
      spacer(), spacer(),
      centeredBold('Višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina pomoću dubokog učenja', PT14),
      spacer(),
      para([new TextRun({ text: 'Strojno učenje', font: F, size: PT12, bold: true, color: BLK })],
           { alignment: AlignmentType.CENTER, spacing: { after: 80 } }),
      centeredBold('SEMINARSKI RAD', PT12),
      spacer(), spacer(), spacer(), spacer(),
      centeredBold('Patrik Hrvoje Radočaj', PT12),
      spacer(),
      centeredBold('Osijek, 2026.', PT12),
      pageBreak(),

      // ══ SAŽETAK ══════════════════════════════════════════════════════════
      h1('', 'Sažetak'),
      p('U ovom projektu razvijen je višerazinski sustav za automatsko prepoznavanje životinjskih vrsta i pasmina na fotografijama. Sustav je podijeljen u dvije sekvencijalne faze: YOLO11s detektor identificira vrstu životinje i njenu lokaciju unutar slike, a zatim EfficientNetV2-S klasifikator prima izrezani isječak životinje i određuje pasminu unutar prepoznate vrste.'),
      p('Projekt je potpuno završen. Species detektor postiže mAP@50 = 0,781 (40 epoha, srpanj 2026.) na finalnom skupu podataka. Breed klasifikatori su istrenirani za svih 16 vrsta s dovoljno podataka, s rezultatima od 98,4 % (donkey) do 35,3 % (sheep), ovisno o broju dostupnih primjera i vizualnoj sličnosti pasmina. Direktni YOLO breed model koji pokušava riješiti cijeli zadatak bez hijerarhije dostigao je mAP@50 = 0,228 nakon 25 epoha treniranja (20 + 5 nastavak) — model pokazuje određeno učenje, ali ostaje daleko ispod dvostupanjskog pristupa.'),
      p('Uz modele, projekt uključuje infrastrukturu za rudarenje neanotiranih primjera (324 633 kandidata), interaktivne alate za pregled i reviziju podataka, sustav za normalizaciju oznaka između heterogenih dataseta te lokalnu web aplikaciju za demonstraciju inferencije.'),
      para([
        new TextRun({ text: 'Ključne riječi: ', font: F, size: PT12, bold: true, color: BLK }),
        new TextRun({ text: 'detekcija objekata, klasifikacija pasmina, YOLO11, EfficientNetV2, višerazinsko učenje, računalni vid, duboko učenje', font: F, size: PT12, italics: true, color: BLK })
      ]),
      pageBreak(),

      // ══ SADRŽAJ ══════════════════════════════════════════════════════════
      new TableOfContents('Sadržaj', { hyperlink: true, headingStyleRange: '1-2' }),
      pageBreak(),

      // ══ 1. UVOD ══════════════════════════════════════════════════════════
      h1(1, 'Uvod'),
      p('Automatsko prepoznavanje životinja na fotografijama primjenjuje se u veterinarskoj dijagnostici, praćenju divljih životinja, upravljanju farmama te katalogizaciji i pretraživanju fotografija. Problem nije trivijalan — nije dovoljno samo detektirati vrstu, nego i razlikovati pasmine unutar iste vrste, što može biti izuzetno suptilno. Labrador i zlatni retriver razlikuju se uglavnom bojom dlake, dok border collie i australski ovčar dijele gotovo identičnu siluetu.'),
      p('U praktičnim uvjetima problem je dodatno otežan heterogenošću podataka: slike dolaze iz različitih izvora s nekonzistentnim oznakama, raznolikim pozadinama i velikom neuravnoteženošću između klasa. Razlika između najzastupljenije (cat: 4 542 primjera) i najmanje zastupljene klase (duck: 5 primjera, goose: 0) u ovom projektu premašuje 900:1.'),
      p('Zbog toga je odabran višerazinski pristup koji problem razdvaja na dvije upravljive faze. Ispravnost ove odluke potvrđena je kontrolnim eksperimentom: direktni YOLO breed model koji pokušava riješiti cijeli zadatak odjednom dostigao je mAP@50 = 0,228 nakon 25 epoha treniranja (20 + 5 nastavak) — model pokazuje određeno učenje ali uz recall ≤ 0,22 i preciznost koja pada s brojem epoha zbog nemogućnosti razlikovanja 100+ vizualno sličnih klasa odjednom — dok dvostupanjski pristup daje smislene rezultate za sve istrenirane vrste.'),
      p('Ovaj rad dokumentira arhitekturu sustava, sve odluke u dizajnu pipeline-a, stvarne metrike za svaki model te pomoćnu infrastrukturu koja projekt čini upotrebljivim izvan trening skripti.'),

      // ══ 2. PODACI ════════════════════════════════════════════════════════
      h1(2, 'Skup podataka, eksplorativna analiza i predobrada'),
      h2('2.1.', 'Izvori podataka i kanonski manifest'),
      p('Projekt objedinjuje tri kategorije izvora podataka. Javno dostupni YOLO-formatirani skupovi pokrivaju opće kategorije životinja i čine osnovu species detektora. Ručno označeni primjeri sadrže pasminske oznake, ali su malobrojniji i zahtijevaju dodatnu normalizaciju. Sintetički generirani primjeri nastaju kompozitnom metodom opisanom u odjeljku 2.3.'),
      p('Svi podaci standardizirani su kroz kanonski manifest — tablicu u kojoj svaki redak predstavlja jednu sliku s atributima: jedinstveni identifikator, putanja do datoteke, izvor i vrsta izvora, kanonska oznaka vrste (canonical_species), kanonska oznaka pasmine (canonical_breed) u obliku vrsta:pasmina, dimenzije slike, koordinate bounding boxa, split (train/val/test) i oznake kvalitete. Normalizacija oznaka između dataseta provodi se automatski kroz sustav cross_dataset_label_map.yaml.'),

      h2('2.2.', 'Statistike i neuravnoteženost klasa'),
      p('Glavni skup za detekciju vrsta sadrži 18 klasa i 27 433 slike. Distribucija po skupovima prikazana je u Tablici 1.'),

      spacer(),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
        children: [new TextRun({ text: 'Tablica 1. Raspodjela skupa podataka', font: F, size: PT11, bold: true, color: BLK })] }),
      new Table({
        width: { size: 7200, type: WidthType.DXA }, alignment: AlignmentType.CENTER,
        columnWidths: [2000, 1600, 1600, 2000],
        rows: [
          new TableRow({ children: [hCell('Skup', 2000), hCell('Broj slika', 1600), hCell('Udio', 1600), hCell('Napomena', 2000)] }),
          new TableRow({ children: [dCell('Trening', 2000, false), dCell('22 587', 1600, true), dCell('82,3 %', 1600, true), dCell('Za treniranje modela', 2000)] }),
          new TableRow({ children: [dCell('Validacija', 2000), dCell('3 375', 1600, true), dCell('12,3 %', 1600, true), dCell('Praćenje konvergencije', 2000)] }),
          new TableRow({ children: [dCell('Test', 2000), dCell('1 471', 1600, true), dCell('5,4 %', 1600, true), dCell('Konačna evaluacija', 2000)] }),
          new TableRow({ children: [dCell('Ukupno', 2000, false, true), dCell('27 433', 1600, true, true), dCell('100 %', 1600, true, true), dCell('18 klasa vrsta', 2000, false, true)] }),
        ]
      }),
      spacer(),
      p('Neuravnoteženost između klasa je izražena i neposredno utječe na kvalitetu modela. Slika 1 prikazuje raspodjelu primjera po vrstama u trening skupu — razlika između mačaka (4 542) i guske (0) ili patke (5) je dramatična.'),
      caption('Slika 1. Raspodjela primjera po vrstama u trening skupu.'),
      img(BASE + 'prezentacija_assets\\species_balance.png', 430, 242, 'Raspodjela vrsta'),
      spacer(),
      p('Vrste poput duck, horse i goose bile su inicijalno gotovo nevidljive za detektor, što je zahtijevalo posebne mjere — kontrolirani sampling, sintetičke primjere i ručno označene (PHR) slike.'),

      h2('2.3.', 'Sintetički kompoziti i augmentacije'),
      p('Ključna mjera za suzbijanje neuravnoteženosti klasa su sintetički kompoziti, generirani sustavom HybridSpeciesYoloDataset. Proces je sljedeći: iz trening slika izrezuju se pojedinačne životinje korištenjem postojećih YOLO oznaka, a zatim se više takvih izrezaka različitih vrsta postavlja na novu pozadinsku teksturu u raspoređenom gridu s doziranim šumom (pozicijski jitter ±6 %, skaliranje 0,88–1,20×). Bounding boxovi se automatski generiraju za svaki postavljeni izrezak. Omjer uzorkovanja: 40 % sintetički kompoziti, 40 % originalne slike, 20 % jako augmentirani pojedinačni uzorci. Prag minimalne veličine boxa je 15–20 piksela.'),
      p('Slike 2 i 3 prikazuju stvarne primjere iz projekta. Slika 2 prikazuje izvorne izrezake koji ulaze kao komponente, a Slika 3 prikazuje rezultirajući sintetički kompozit s automatski generiranim bounding boxovima.'),
      caption('Slika 2. Izvorni izresci životinja (duck, horse, pig, sheep) iz trening skupa koji ulaze u sintetički compositor.'),
      img(BASE + 'synthetic_source_crops.jpg', 420, 110, 'Izvorni izresci'),
      spacer(),
      caption('Slika 3. Sintetički kompozit: izresci postavljeni na stvarnu pozadinsku teksturu iz projekta s automatski generiranim bounding boxovima. Slika nastaje on-the-fly za svaku trening epohu.'),
      img(BASE + 'synthetic_composite_example.jpg', 300, 300, 'Sinteticki kompozit'),
      spacer(),
      p('Augmentacije primijenjene uz sintetičke slike: horizontalni i vertikalni flip, rotacija ±15°, promjena ekspozicije ±22 %, Gaussov šum (σ = 3 %). Za breed klasifikator koriste se MixUp (α = 0,2) i CutMix (α = 1,0). Za YOLO detektor mosaic augmentacija (kombinira 4 slike) gasi se 10 epoha prije kraja treninga.'),

      h2('2.4.', 'Rudarenje neanotiranih podataka'),
      p('Istrenirani species detektor koristi se i za rudarenje neanotiranih životinja iz dostupnih dataseta — pretražuje slike za koje postoje YOLO oznake za neke klase, ali ne i za sve moguće životinje u kadru. Detekcije koje se ne podudaraju s postojećim oznakama postaju kandidati za manualnu reviziju i potencijalnu integraciju.'),
      p('Dosadašnji rezultati: 324 633 ukupnih kandidata, 20 000 visokoprioritnih. Slika 4 prikazuje tri karakteristična slučaja iz mininga: goveda označena kao "horse" (cross-species konfuzija), neunotiran konj u sceni i mačka klasificirana kao "rabbit".'),
      caption('Slika 4. Primjeri iz mininga neanotiranih životinja — lijevo: cross-species konfuzija (govedo → horse), sredina: neunotiran konj, desno: mačka → rabbit.'),
      img(BASE + 'mining_samples_panel.jpg', 430, 141, 'Mining primjeri'),
      spacer(),

      // ══ 3. METODE ════════════════════════════════════════════════════════
      h1(3, 'Korištene metode i algoritmi'),
      h2('3.1.', 'YOLO11s species detektor'),
      p('Prva faza koristi YOLO11s (small varijanta iz generacije YOLO11 tvrtke Ultralytics). YOLO dijeli ulaznu sliku na mrežu ćelija i u jednom prolazu predviđa sve bounding boxove s pridruženim klasama — jednofazna arhitektura bez zasebne faze predlaganja regija.'),
      p('Treniranje koristi hibridni dataset (HybridSpeciesYoloDataset) koji kombinira više YOLO izvora s kontroliranim uzorkovanjem po vrstama. Hardware: AMD GPU s ROCm 7.2.1 u izoliranoj Python 3.12 embed okolini. Ključni hiperparametri prikazani su u Tablici 2.'),
      spacer(),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
        children: [new TextRun({ text: 'Tablica 2. Hiperparametri YOLO11s species detektora', font: F, size: PT11, bold: true, color: BLK })] }),
      new Table({
        width: { size: 8000, type: WidthType.DXA }, alignment: AlignmentType.CENTER,
        columnWidths: [3000, 1800, 3200],
        rows: [
          new TableRow({ children: [hCell('Parametar', 3000), hCell('Vrijednost', 1800), hCell('Napomena', 3200)] }),
          new TableRow({ children: [dCell('Bazni model', 3000), dCell('yolo11s.pt', 1800, true), dCell('Pretreniran na COCO', 3200)] }),
          new TableRow({ children: [dCell('Ulazna rezolucija', 3000), dCell('640 × 640 px', 1800, true), dCell('Standard za YOLO', 3200)] }),
          new TableRow({ children: [dCell('Epohe / patience', 3000), dCell('100 / 25', 1800, true), dCell('Early stopping', 3200)] }),
          new TableRow({ children: [dCell('Batch size / workers', 3000), dCell('16 / 8', 1800, true), dCell('', 3200)] }),
          new TableRow({ children: [dCell('lr0 / raspored', 3000), dCell('0,01 / cosine', 1800, true), dCell('cos_lr = true', 3200)] }),
          new TableRow({ children: [dCell('Mosaic / AMP', 3000), dCell('close=10 / true', 1800, true), dCell('Gasi se 10 ep. pred kraj', 3200)] }),
        ]
      }),
      spacer(),

      h2('3.2.', 'EfficientNetV2-S breed klasifikator'),
      p('Drugi stupanj koristi EfficientNetV2-S, varijanta tf_efficientnetv2_s.in21k_ft_in1k iz timm biblioteke, pretrenirana na ImageNet-21k (14 milijuna slika, 21 000 klasa) uz daljnje fino podešavanje na ImageNet-1k. Ova dvofazna inicijalizacija osigurava iznimno bogatu inicijalizaciju za fine-grained klasifikaciju vizualnih detalja poput teksture i obojenosti dlake.'),
      p('Napomena: koristi se EfficientNetV2-S (22 M parametara, 384 × 384 px nativna rezolucija), a ne EfficientNetB2 koji se pojavljuje u ranijim verzijama dokumentacije. Trening je modularan: zasebni model po vrsti, s brojem klasa 15–30 umjesto 200+ zajedno. Ulaz modelu je crop iz prve faze, čime se eliminira pozadina. Ključni hiperparametri prikazani su u Tablici 3.'),
      spacer(),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
        children: [new TextRun({ text: 'Tablica 3. Hiperparametri EfficientNetV2-S breed klasifikatora', font: F, size: PT11, bold: true, color: BLK })] }),
      new Table({
        width: { size: 8000, type: WidthType.DXA }, alignment: AlignmentType.CENTER,
        columnWidths: [3000, 1800, 3200],
        rows: [
          new TableRow({ children: [hCell('Parametar', 3000), hCell('Vrijednost', 1800), hCell('Napomena', 3200)] }),
          new TableRow({ children: [dCell('Bazni model', 3000), dCell('tf_efficientnetv2_s.in21k_ft_in1k', 1800, true), dCell('ImageNet-21k → IN-1k', 3200)] }),
          new TableRow({ children: [dCell('Ulazna rezolucija', 3000), dCell('384 × 384 px', 1800, true), dCell('Nativna rezolucija V2-S', 3200)] }),
          new TableRow({ children: [dCell('Epohe / patience', 3000), dCell('60 / 10', 1800, true), dCell('Min. poboljšanje 0,001', 3200)] }),
          new TableRow({ children: [dCell('Batch size / optimizer', 3000), dCell('48 / AdamW', 1800, true), dCell('L2 reg. weight_decay=0,05', 3200)] }),
          new TableRow({ children: [dCell('Stopa učenja', 3000), dCell('0,0003', 1800, true), dCell('Cosine annealing', 3200)] }),
          new TableRow({ children: [dCell('Label smoothing', 3000), dCell('0,1', 1800, true), dCell('Smanjuje overconfidence', 3200)] }),
          new TableRow({ children: [dCell('MixUp / CutMix α', 3000), dCell('0,2 / 1,0', 1800, true), dCell('Augmentacije raznolikosti', 3200)] }),
        ]
      }),
      spacer(),

      h2('3.3.', 'Dvostupanjski pipeline'),
      p('Inferencijski tok: ulazna fotografija → YOLO species detektor → za svaki okvir s pouzdanošću > 0,25 vrši se crop s marginom 10–15 % → izrezani isječak odlazi u breed klasifikator za detektiranu vrstu → ako model za vrstu nije dostupan, vraća se samo species predikcija → izlaz: JSON s detaljima i anotirana slika.'),

      h2('3.4.', 'Direktni YOLO breed model — kontrolni eksperiment'),
      p('Kao kontrolni eksperiment implementiran je direktni YOLO breed model koji pokušava detektirati pasminu bez prethodnog stupnja detekcije vrste — jedan model, 100+ klasa odjednom. Model je treniran u dva koraka: 20 epoha (nastavak od prethodnih 5) i zatim +5 epoha nastavak od najboljeg checkpointa. Ukupno 25 epoha, best mAP@50 = 0,228 (epoha 15 prvog koraka), Precision = 0,480, Recall = 0,209. Model pokazuje jasno učenje (0,012 → 0,228) ali s 100+ vizualno sličnih klasa bez hijerarhijske strukture preciznost pada s daljnjim treningom (val/cls_loss raste od epohe 15 nadalje), a recall ostaje ispod 0,22. Ovaj eksperiment pokazuje da direktni pristup nije potpuno beskoristan, ali hijerarhijska dekompozicija ostaje jedino rješenje koje daje praktično upotrebljive rezultate za sve vrste.'),

      h2('3.5.', 'Alati za pregled i reviziju podataka'),
      p('Projekt uključuje dva interaktivna web alata. Box review app (Slika 5) prikazuje kandidate iz mininga i omogućuje ručni pregled — potvrdu, odbacivanje ili korekciju svakog bounding boxa, bez mijenjanja originalnih oznaka. Label audit app (Slika 6) prikazuje sliku uz pridijeljenu oznaku za brzu identifikaciju pogrešnih anotacija.'),
      caption('Slika 5. Box review app — lista kandidata iz mininga s pouzdanošću (lijevo) i panel za pregled koordinata (desno).'),
      img(BASE + 'selenium-box-review-4.png', 380, 208, 'Box review app'),
      spacer(),
      caption('Slika 6. Label audit app — pregled kameljih pasminskih oznaka s kontaktnim listom primjera.'),
      img(BASE + 'label-audit-app.png', 380, 200, 'Label audit app'),
      spacer(),

      // ══ 4. EVALUACIJA ════════════════════════════════════════════════════
      h1(4, 'Evaluacija modela'),
      h2('4.1.', 'Korištene metrike'),
      p('Za YOLO detektor koriste se standardne metrike detekcije objekata: Precision (P) — udio ispravnih detekcija; Recall (R) — udio pronađenih stvarnih objekata; mAP@50 — srednja prosječna preciznost pri IoU pragu 0,50; mAP@50–95 — stroža mjera za IoU pragove 0,50–0,95 u koraku 0,05.'),
      p('Za breed klasifikator: Top-1 točnost (točna klasa na prvom mjestu) i Top-5 točnost (točna klasa u prvih 5 predikcija).'),

      h2('4.2.', 'Species detektor — iteracije treniranja'),
      p('Provedeno je više iteracija s postupnim poboljanjima u kvaliteti podataka. Tablica 4 prikazuje ključne iteracije, a Slika 7 usporedbu finalnih metrika između osnovnog i poboljšanog eksperimenta.'),
      spacer(),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
        children: [new TextRun({ text: 'Tablica 4. Usporedba iteracija species detektora', font: F, size: PT11, bold: true, color: BLK })] }),
      new Table({
        width: { size: 8500, type: WidthType.DXA }, alignment: AlignmentType.CENTER,
        columnWidths: [3500, 1250, 1250, 1250, 1250],
        rows: [
          new TableRow({ children: [hCell('Eksperiment', 3500), hCell('P', 1250), hCell('R', 1250), hCell('mAP@50', 1250), hCell('mAP@50–95', 1250)] }),
          new TableRow({ children: [dCell('hybrid_shared_full17_v2 (rana)', 3500), dCell('0,361', 1250, true), dCell('0,366', 1250, true), dCell('0,346', 1250, true), dCell('0,163', 1250, true)] }),
          new TableRow({ children: [dCell('hybrid_shared_full18_goose_10more', 3500), dCell('0,615', 1250, true), dCell('0,616', 1250, true), dCell('0,617', 1250, true), dCell('0,324', 1250, true)] }),
          new TableRow({ children: [dCell('hybrid_shared_full18_goose_5more_box25', 3500), dCell('0,650', 1250, true), dCell('0,659', 1250, true), dCell('0,666', 1250, true), dCell('0,351', 1250, true)] }),
          new TableRow({ children: [dCell('hybrid_shared_full18_phr_5more_box25', 3500), dCell('0,644', 1250, true), dCell('0,685', 1250, true), dCell('0,679', 1250, true), dCell('0,365', 1250, true)] }),
          new TableRow({ children: [dCell('hybrid_shared_fresh_qa_full_20260706 (final)', 3500, false, true), dCell('0,788', 1250, true, true), dCell('0,755', 1250, true, true), dCell('0,781', 1250, true, true), dCell('0,477', 1250, true, true)] }),
        ]
      }),
      spacer(),
      caption('Slika 7. Usporedba finalnih metrika između osnovnog i poboljšanog eksperimenta species detektora.'),
      img(BASE + 'prezentacija_assets\\metrics_compare.png', 400, 200, 'Usporedba metrika'),
      spacer(),
      caption('Slika 8. Trend metrika kroz zadnjih 5 epoha finalnog hybrid species run-a.'),
      img(BASE + 'prezentacija_assets\\hybrid_trend.png', 400, 200, 'Trend metrika'),
      spacer(),

      h2('4.3.', 'Breed klasifikatori — rezultati'),
      p('Breed klasifikatori su istrenirani za svih 16 vrsta. Tablica 5 prikazuje finalne metrike, a Slika 9 vizualnu usporedbu top-1 točnosti po vrstama.'),
      spacer(),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
        children: [new TextRun({ text: 'Tablica 5. Finalne metrike breed klasifikatora (zadnja epoha, early stopping patience = 10)', font: F, size: PT11, bold: true, color: BLK })] }),
      new Table({
        width: { size: 8500, type: WidthType.DXA }, alignment: AlignmentType.CENTER,
        columnWidths: [1600, 800, 1600, 1400, 1400, 1700],
        rows: [
          new TableRow({ children: [hCell('Vrsta', 1600), hCell('Ep.', 800), hCell('val_top1', 1600), hCell('val_top5', 1400), hCell('val_loss', 1400), hCell('train_top1', 1700)] }),
          new TableRow({ children: [dCell('donkey', 1600), dCell('10', 800, true), dCell('98,35 %', 1600, true, true), dCell('98,77 %', 1400, true), dCell('0,667', 1400, true), dCell('99,95 %', 1700, true)] }),
          new TableRow({ children: [dCell('pig', 1600), dCell('10', 800, true), dCell('96,74 %', 1600, true, true), dCell('99,94 %', 1400, true), dCell('0,596', 1400, true), dCell('99,95 %', 1700, true)] }),
          new TableRow({ children: [dCell('peafowl', 1600), dCell('10', 800, true), dCell('94,57 %', 1600, true, true), dCell('98,45 %', 1400, true), dCell('0,637', 1400, true), dCell('99,59 %', 1700, true)] }),
          new TableRow({ children: [dCell('rabbit', 1600), dCell('10', 800, true), dCell('93,75 %', 1600, true), dCell('100,00 %', 1400, true), dCell('0,841', 1400, true), dCell('100,00 %', 1700, true)] }),
          new TableRow({ children: [dCell('chicken', 1600), dCell('10', 800, true), dCell('89,59 %', 1600, true), dCell('95,26 %', 1400, true), dCell('1,325', 1400, true), dCell('99,63 %', 1700, true)] }),
          new TableRow({ children: [dCell('pigeon', 1600), dCell('10', 800, true), dCell('86,76 %', 1600, true), dCell('87,25 %', 1400, true), dCell('1,473', 1400, true), dCell('99,61 %', 1700, true)] }),
          new TableRow({ children: [dCell('goat', 1600), dCell('10', 800, true), dCell('86,10 %', 1600, true), dCell('98,93 %', 1400, true), dCell('0,992', 1400, true), dCell('99,88 %', 1700, true)] }),
          new TableRow({ children: [dCell('goose', 1600), dCell('10', 800, true), dCell('78,41 %', 1600, true), dCell('92,05 %', 1400, true), dCell('1,427', 1400, true), dCell('99,50 %', 1700, true)] }),
          new TableRow({ children: [dCell('dromedary', 1600), dCell('10', 800, true), dCell('77,84 %', 1600, true), dCell('96,39 %', 1400, true), dCell('1,024', 1400, true), dCell('97,48 %', 1700, true)] }),
          new TableRow({ children: [dCell('cattle', 1600), dCell('10', 800, true), dCell('59,84 %', 1600, true), dCell('82,70 %', 1400, true), dCell('2,334', 1400, true), dCell('97,98 %', 1700, true)] }),
          new TableRow({ children: [dCell('horse', 1600), dCell('10', 800, true), dCell('58,10 %', 1600, true), dCell('72,38 %', 1400, true), dCell('2,734', 1400, true), dCell('98,70 %', 1700, true)] }),
          new TableRow({ children: [dCell('cat', 1600), dCell('10', 800, true), dCell('58,22 %', 1600, true), dCell('81,88 %', 1400, true), dCell('2,246', 1400, true), dCell('98,93 %', 1700, true)] }),
          new TableRow({ children: [dCell('duck', 1600), dCell('10', 800, true), dCell('55,24 %', 1600, true), dCell('92,31 %', 1400, true), dCell('1,633', 1400, true), dCell('93,84 %', 1700, true)] }),
          new TableRow({ children: [dCell('dog', 1600), dCell('10', 800, true), dCell('48,55 %', 1600, true), dCell('78,26 %', 1400, true), dCell('2,676', 1400, true), dCell('98,43 %', 1700, true)] }),
          new TableRow({ children: [dCell('sheep', 1600), dCell('10', 800, true), dCell('35,32 %', 1600, true), dCell('60,32 %', 1400, true), dCell('3,775', 1400, true), dCell('98,94 %', 1700, true)] }),
        ]
      }),
      spacer(),
      caption('Slika 9. Top-1 točnost breed klasifikatora po vrstama.'),
      img(BASE + 'chart_breed_accuracy.png', 420, 250, 'Breed accuracy'),
      spacer(),

      h2('4.4.', 'Analiza konvergencije i overfittinga'),
      p('Slika 10 prikazuje konvergenciju kroz epohe za četiri odabrane vrste. Donkey i chicken konvergiraju brzo i stabilno, dok dog i sheep osciliraju i plateauiraju na niskim vrijednostima. Desni panel prikazuje razliku između train i val lossa — donkey pokazuje dobar fit (train i val zajedno padaju), dok dog pokazuje overfitting (train loss pada prema nuli, val loss stagnira).'),
      p('Uzrok overfittinga kod dog i sheep je kombinacija: velik broj pasmina, vizualna sličnost između klasa i nedovoljan broj raznolikih primjera. Rješenje nije složeniji model nego više raznolikih podataka — upravo zato postoji mining pipeline.'),
      caption('Slika 10. Lijevo: konvergencija val_top1 kroz epohe za dog, donkey, sheep, chicken. Desno: train vs. val loss za dog (overfitting) i donkey (dobra generalizacija).'),
      img(BASE + 'chart_training_curves.png', 430, 175, 'Training curves'),
      spacer(),

      // ══ 5. DEMONSTRACIJA ═════════════════════════════════════════════════
      h1(5, 'Demonstracija rada'),
      h2('5.1.', 'Pokretanje i CLI ulazne točke'),
      p('Projekt definira 17 CLI naredbi u pyproject.toml. Tipičan tijek rada:'),
      code('animal-build-manifest          # izgradnja kanonskog manifesta'),
      code('animal-data-qa                 # automatizirani quality gate'),
      code('animal-train-species-aware     # treniranje species detektora'),
      code('animal-train-hierarchical-classifier  # breed modeli za sve vrste'),
      code('animal-mine-unlabeled-animals  # rudarenje neanotiranih primjera'),
      code('animal-review-boxes            # interaktivni pregled kandidata'),
      code('animal-infer-two-stage         # inferencija na slikama'),
      spacer(),

      h2('5.2.', 'Web aplikacija za inferenciju'),
      p('Lokalna web aplikacija (src/animal_breed/webapp/model_inference_app.py) izlaže REST API i statičko web sučelje. Korisnik uploadira fotografiju, aplikacija pokreće puni pipeline i vraća anotiranu sliku s okvirima i tablicom predikcija. Primjer JSON izlaza:'),
      code('{ "detections": [{'),
      code('    "species": "dog",  "species_confidence": 0.94,'),
      code('    "breed": "golden_retriever",  "breed_confidence": 0.82,'),
      code('    "breed_top3": ["golden_retriever", "labrador_retriever", "..."]'),
      code('}] }'),
      spacer(),

      // ══ 6. ZAKLJUČAK ═════════════════════════════════════════════════════
      h1(6, 'Zaključak'),
      p('Projekt je potpuno završen. Species detektor postiže mAP@50 = 0,781 (40 epoha, srpanj 2026., P=0,788, R=0,755), svih 16 breed klasifikatora je istrenirano i evaluirano, a kontrolni eksperiment s direktnim breed YOLO modelom (25 epoha ukupno, best mAP@50 = 0,228) pokazuje da model uči ali ostaje praktično neupotrebljiv za sve 100+ klasa odjednom.'),
      p('Ključni zaključci: (1) hijerarhijska dekompozicija je nužna — direktni YOLO breed model postiže tek mAP@50 = 0,228 s recallom ≤ 0,22; (2) kvaliteta i balansiranje podataka su jednako važni kao arhitektura — rani detektor daje mAP@50 = 0,346, finalni s poboljšanim podacima 0,781 (+126%); (3) 9 od 15 breed modela postiže ≥ 78 % top-1 točnosti; (4) dog (48,6 %) i sheep (35,3 %) pate od overfittinga uzrokovanog nedovoljnim brojem raznolikih primjera — mining pipeline je direktan odgovor na ovaj problem.'),
      p('Budući smjerovi: integracija mining kandidata u breed trening skupove, agresivnija regularizacija za slabe modele, istraživanje few-shot learning pristupa za rijetke pasmine.'),

      // ══ 7. LITERATURA ════════════════════════════════════════════════════
      h1(7, 'Korištene tehnologije i literatura'),
      h2('7.1.', 'Korištene tehnologije'),
      spacer(),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
        children: [new TextRun({ text: 'Tablica 6. Pregled korištenih tehnologija', font: F, size: PT11, bold: true, color: BLK })] }),
      new Table({
        width: { size: 8000, type: WidthType.DXA }, alignment: AlignmentType.CENTER,
        columnWidths: [1800, 1400, 4800],
        rows: [
          new TableRow({ children: [hCell('Tehnologija', 1800), hCell('Verzija', 1400), hCell('Uloga u projektu', 4800)] }),
          new TableRow({ children: [dCell('Python', 1800), dCell('≥ 3.11', 1400, true), dCell('Primarni programski jezik', 4800)] }),
          new TableRow({ children: [dCell('PyTorch', 1800), dCell('≥ 2.4', 1400, true), dCell('Breed klasifikator, trening loop', 4800)] }),
          new TableRow({ children: [dCell('Ultralytics YOLO11', 1800), dCell('≥ 8.4.48', 1400, true), dCell('Species detektor, direktni breed model', 4800)] }),
          new TableRow({ children: [dCell('timm (EfficientNetV2-S)', 1800), dCell('≥ 1.0.27', 1400, true), dCell('Breed klasifikator — pretreniran backbone', 4800)] }),
          new TableRow({ children: [dCell('OpenCV', 1800), dCell('≥ 4.10', 1400, true), dCell('Obrada slika, crop, vizualizacija', 4800)] }),
          new TableRow({ children: [dCell('Hydra / OmegaConf', 1800), dCell('≥ 1.3.2', 1400, true), dCell('Konfiguracija eksperimenata', 4800)] }),
          new TableRow({ children: [dCell('Albumentations', 1800), dCell('≥ 2.0.8', 1400, true), dCell('Augmentacije', 4800)] }),
          new TableRow({ children: [dCell('wandb', 1800), dCell('≥ 0.27', 1400, true), dCell('Praćenje eksperimenata', 4800)] }),
          new TableRow({ children: [dCell('AMD ROCm', 1800), dCell('7.2.1', 1400, true), dCell('GPU runtime za treniranje (Windows)', 4800)] }),
        ]
      }),
      spacer(),

      h2('7.2.', 'Literatura'),
      p('[1] Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics. https://github.com/ultralytics/ultralytics'),
      p('[2] Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. ICML 2021. arXiv:2104.00298.'),
      p('[3] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019. arXiv:1905.11946.'),
      p('[4] Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.'),
      p('[5] Wightman, R. (2019). PyTorch Image Models (timm). https://github.com/huggingface/pytorch-image-models'),
      p('[6] Yadan, O. (2019). Hydra — A framework for elegantly configuring complex applications. https://github.com/facebookresearch/hydra'),
      p('[7] Buslaev, A. et al. (2020). Albumentations: Fast and Flexible Image Augmentations. Information, 11(2), 125.'),
      p('[8] Russakovsky, O. et al. (2015). ImageNet Large Scale Visual Recognition Challenge. IJCV, 115(3), 211–252.'),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('seminar_projekt_strojno_ucenje_v4.docx', buf);
  console.log('Done: seminar_projekt_strojno_ucenje_v4.docx');
}).catch(e => { console.error(e); process.exit(1); });
