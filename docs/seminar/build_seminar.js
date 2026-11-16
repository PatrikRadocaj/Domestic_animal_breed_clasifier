const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, TableOfContents,
  LevelFormat, ExternalHyperlink
} = require('docx');
const fs = require('fs');

// ─── helpers ────────────────────────────────────────────────────────────────
const BLUE  = '17365D';
const LGRAY = 'F2F2F2';
const LBLUE = 'D9EAF7';
const MBLUE = 'BDD7EE';
const GRAY  = '555555';

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
  return new Paragraph({
    spacing: { after: 160 },
    children: [new TextRun({ text, font: 'Arial', size: 22, ...opts })]
  });
}
function bold(text) { return new TextRun({ text, font: 'Arial', size: 22, bold: true }); }
function italic(text) { return new TextRun({ text, font: 'Arial', size: 22, italics: true }); }
function mixed(...runs) {
  return new Paragraph({ spacing: { after: 160 }, children: runs });
}
function bullet(text, opts = {}) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: 'Arial', size: 22, ...opts })]
  });
}
function code(text) {
  return new Paragraph({
    spacing: { after: 80, before: 80 },
    indent: { left: 720 },
    children: [new TextRun({ text, font: 'Courier New', size: 20, color: '1F3864' })]
  });
}
function pageBreak() { return new Paragraph({ children: [new PageBreak()] }); }
function spacer() { return new Paragraph({ spacing: { after: 160 }, children: [new TextRun('')] }); }

const border1 = { style: BorderStyle.SINGLE, size: 4, color: 'AAAAAA' };
const cellBorders = { top: border1, bottom: border1, left: border1, right: border1 };

function hdrCell(text, w) {
  return new TableCell({
    borders: cellBorders,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: MBLUE, type: ShadingType.CLEAR },
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ children: [bold(text)] })]
  });
}
function dataCell(text, w, shade = false) {
  return new TableCell({
    borders: cellBorders,
    width: { size: w, type: WidthType.DXA },
    shading: shade ? { fill: LGRAY, type: ShadingType.CLEAR } : { fill: 'FFFFFF', type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ children: [new TextRun({ text, font: 'Arial', size: 20 })] })]
  });
}

// ─── document ───────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
      }]
    }]
  },
  styles: {
    default: { document: { run: { font: 'Arial', size: 22 } } },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: 'Arial', color: BLUE },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: MBLUE, space: 4 } } }
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: 'Arial', color: BLUE },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 }
      },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: MBLUE, space: 4 } },
          children: [
            new TextRun({ text: 'Višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina', font: 'Arial', size: 18, color: GRAY }),
            new TextRun({ text: '\t', font: 'Arial', size: 18 }),
            new TextRun({ text: '2026.', font: 'Arial', size: 18, color: GRAY }),
          ],
          tabStops: [{ type: 'right', position: 9026 }]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: MBLUE, space: 4 } },
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: 'Stranica ', font: 'Arial', size: 18, color: GRAY }),
            new TextRun({ children: [PageNumber.CURRENT], font: 'Arial', size: 18, color: GRAY }),
            new TextRun({ text: ' od ', font: 'Arial', size: 18, color: GRAY }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], font: 'Arial', size: 18, color: GRAY }),
          ]
        })]
      })
    },
    children: [

      // ── NASLOVNICA ──────────────────────────────────────────────────────
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 1200, after: 240 },
        children: [new TextRun({
          text: 'Višerazinski sustav za prepoznavanje životinjskih vrsta',
          font: 'Arial', size: 44, bold: true, color: BLUE
        })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 480 },
        children: [new TextRun({
          text: 'i pasmina pomoću dubokog učenja',
          font: 'Arial', size: 44, bold: true, color: BLUE
        })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 600 },
        children: [new TextRun({
          text: 'Projekt iz strojnog učenja i računalnog vida',
          font: 'Arial', size: 26, italics: true, color: GRAY
        })]
      }),

      new Table({
        width: { size: 6000, type: WidthType.DXA },
        alignment: AlignmentType.CENTER,
        columnWidths: [2200, 3800],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, width: { size: 2200, type: WidthType.DXA },
              shading: { fill: LBLUE, type: ShadingType.CLEAR },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [bold('Autor:')] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3800, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [new TextRun({ text: '[OVDJE DODATI IME I PREZIME]', font: 'Arial', size: 22 })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, width: { size: 2200, type: WidthType.DXA },
              shading: { fill: LBLUE, type: ShadingType.CLEAR },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [bold('Kolegij:')] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3800, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [new TextRun({ text: '[OVDJE DODATI NAZIV KOLEGIJA]', font: 'Arial', size: 22 })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, width: { size: 2200, type: WidthType.DXA },
              shading: { fill: LBLUE, type: ShadingType.CLEAR },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [bold('Tehnologije:')] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3800, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [new TextRun({ text: 'Python, PyTorch, YOLO, EfficientNetB2, OpenCV', font: 'Arial', size: 22 })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, width: { size: 2200, type: WidthType.DXA },
              shading: { fill: LBLUE, type: ShadingType.CLEAR },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [bold('Akademska godina:')] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3800, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [new TextRun({ text: '2025./2026.', font: 'Arial', size: 22 })] })] }),
          ]}),
        ]
      }),

      pageBreak(),

      // ── SAŽETAK ────────────────────────────────────────────────────────────
      h1('Sažetak'),
      p('U ovom projektu razvijen je višerazinski sustav za automatsko prepoznavanje životinjskih vrsta i pasmina na fotografijama. Sustav je arhitekturalno podijeljen u dvije sekvencijalne faze: u prvoj fazi YOLO11s detektor identificira vrstu životinje i lokalizira je unutar slike pomoću okvira detekcije (engl. bounding box), a u drugoj fazi EfficientNetB2 klasifikator prima izrezani isječak životinje i provodi finiju klasifikaciju pasmine unutar prepoznate vrste.'),
      p('Kao kontrolni eksperiment paralelno je implementiran i direktni YOLO model koji pokušava prepoznati pasminu bez prethodne faze detekcije vrste. Taj model služi kao mjerni okvir za procjenu koliko hijerarhijska dekompozicija zadatka zaista pomaže u praksi.'),
      p('Posebna pažnja posvećena je kvaliteti i pripremi podataka. Projekt objedinjuje više heterogenih izvora: javno dostupne YOLO skupove podataka, ručno označene primjere i sintetski generirane slike. Svi podaci standardizirani su kroz jedinstven kanonski manifest, a provedena je detaljna eksplorativna analiza s naglaskom na neuravnoteženost razreda, nekonzistentne oznake i premale okvire detekcije.'),
      p('Na razini infrastrukture, projekt uključuje CLI ulazne točke za sve ključne korake (izgradnja manifesta, priprema podataka, treniranje, inferencija) te lokalnu web aplikaciju koja korisniku omogućuje interaktivni upload slike, prikaz anotiranih detekcija i pregled predikcija za vrstu i pasminu kroz pregledno sučelje.'),
      mixed(bold('Ključne riječi: '), italic('detekcija objekata, klasifikacija pasmina, YOLO, EfficientNet, višerazinsko učenje, računalni vid, duboko učenje')),

      pageBreak(),

      // ── SADRŽAJ ─────────────────────────────────────────────────────────────
      new TableOfContents('Sadržaj', {
        hyperlink: true,
        headingStyleRange: '1-2',
        stylesWithLevels: [
          { styleName: 'Heading 1', level: 1 },
          { styleName: 'Heading 2', level: 2 },
        ]
      }),

      pageBreak(),

      // ── 1. UVOD ──────────────────────────────────────────────────────────
      h1('1. Uvod'),
      p('Automatsko prepoznavanje životinja na fotografijama jedan je od klasičnih i praktično korisnih zadataka računalnog vida. Primjene su široke: veterinarska dijagnostika, praćenje divljih životinja u prirodi, upravljanje farmama i stočarskim operacijama, edukativne aplikacije, digitalna katalogizacija zbirki i automatizirano označavanje fotografija na velikim platformama. Problem, međutim, nije trivijalan — nije dovoljno samo detektirati da se na slici nalazi neka životinja, nego je potrebno razlikovati slične vrste, a potom još finije odrediti pasminu unutar iste vrste.'),
      p('Razlika između vrsta ponekad je vizualno gruba (pas nasuprot ptice), ali razlika između pasmina unutar iste vrste može biti izuzetno suptilna. Labrador retriever i zlatni retriver imaju gotovo identičan oblik tijela, ali različitu boju dlake; border collie i australski ovčar razlikuju se sitnim detaljima koji su vidljivi tek pri dobrom osvjetljenju i kadru. To čini klasifikaciju pasmina znatno zahtjevnijim zadatkom od klasične višekategorične klasifikacije.'),
      p('Motivacija za projektni pristup bila je izraditi sustav koji je robustan na stvarne uvjete: varijabilnu kvalitetu fotografija, male objekte, djelomična zaklanjanja i raznolike pozadine. Zbog toga je odabran višerazinski pristup (engl. hierarchical pipeline) umjesto jednog monolitnog modela koji bi morao rješavati sve odjednom. U ovom pristupu, prvi stupanj se bavi grubom detekcijom i lokalizacijom, dok drugi stupanj rješava finiju klasifikaciju na temelju već izoliranog objekta.'),
      p('Ovaj rad dokumentira arhitekturu sustava, izazove u pripremi podataka, metodologiju treniranja, postignute rezultate i buduće smjerove razvoja. Poseban naglasak stavljen je na transparentnost: uz pozitivne rezultate, dokumentiraju se i ograničenja, nepotpunost nekih skupova i mjesta gdje sustav još nije potpuno stabilan.'),

      // ── 2. PODACI ────────────────────────────────────────────────────────
      h1('2. Opis korištenog skupa podataka, eksplorativna analiza i predobrada podataka'),
      h2('2.1. Izvori podataka i organizacija'),
      p('Projekt koristi kombinaciju više heterogenih izvora podataka. Ovakav pristup nužan je zbog toga što ne postoji jedan objedinjeni javni skup koji bi pokrivao sve željene vrste i pasmine uz dovoljan broj primjera po kategoriji.'),
      p('Prvi izvor su javno dostupni YOLO-formatirani skupovi podataka koji pokrivaju opće kategorije životinja. Ovaj skup daje široku pokrivenost vrsta i dobru količinu podataka za treniranje detektora prve faze, ali je ograničen na razinu vrste bez informacija o pasminama.'),
      p('Drugi izvor su ručno označeni primjeri u CSV i ZIP formatima. Ovi podaci su skupocjeniji jer sadrže informacije o pasminama, ali su malobrojniji i zahtijevaju dodatnu normalizaciju oznaka.'),
      p('Treći izvor su sintetski generirani primjeri. Koristi se kompozitna metoda pakiranja: više izrezanih životinja postavlja se na realistične pozadine, čime se povećava raznolikost vizualnih uvjeta u trening skupu, posebno za rijetke pasmine.'),
      p('Svi podaci objedinjeni su kroz kanonski manifest — tablicu u kojoj svaki redak predstavlja jednu sliku. Manifest uključuje: jedinstven identifikator slike, putanju do datoteke, izvor i vrstu izvora (YOLO, CSV, ZIP, sintetika), licencu, kanonsku oznaku vrste (canonical_species), kanonsku oznaku pasmine (canonical_breed) u obliku vrsta:pasmina, dimenzije slike, koordinate okvira detekcije ako postoje, grupu za podjelu skupa (train/val/test) i oznake kvalitete (quality_flags).'),
      p('Ovakva organizacija omogućuje potpunu reproducibilnost eksperimenta: iz manifesta se uvijek mogu rekonstruirati točno oni podaci koji su korišteni za određeni trening, bez rizika od slučajnog prepisivanja originalnih datoteka.'),

      h2('2.2. Statistike skupa podataka'),
      p('Glavni skup za YOLO detekciju vrsta sadrži 18 klasa i ukupno 27 433 slike. Distribucija po skupovima je sljedeća: 22 587 slika u trening skupu (82,3 %), 3 375 u validacijskom skupu (12,3 %) i 1 471 u testnom skupu (5,4 %).'),

      // Data distribution table
      new Paragraph({ spacing: { after: 80, before: 160 }, children: [bold('Tablica 1. Raspodjela slika po skupovima podataka')] }),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [2500, 2100, 2100, 2326],
        rows: [
          new TableRow({ children: [hdrCell('Skup', 2500), hdrCell('Broj slika', 2100), hdrCell('Udio (%)', 2100), hdrCell('Napomena', 2326)] }),
          new TableRow({ children: [dataCell('Trening', 2500), dataCell('22 587', 2100), dataCell('82,3 %', 2100), dataCell('Za treniranje modela', 2326)] }),
          new TableRow({ children: [dataCell('Validacija', 2500, true), dataCell('3 375', 2100, true), dataCell('12,3 %', 2100, true), dataCell('Praćenje konvergencije', 2326, true)] }),
          new TableRow({ children: [dataCell('Test', 2500), dataCell('1 471', 2100), dataCell('5,4 %', 2100), dataCell('Konačna evaluacija', 2326)] }),
          new TableRow({ children: [dataCell('Ukupno', 2500, true), dataCell('27 433', 2100, true), dataCell('100 %', 2100, true), dataCell('18 klasa vrsta', 2326, true)] }),
        ]
      }),
      spacer(),
      p('Neuravnoteženost između klasa je izražena. Neke vrste kao što su mačka (cat), svinja (pig) i kokoš (chicken) imaju višestruko više primjera od rijetko zastupljenih vrsta poput patke (duck), konja (horse), goveda (cattle) i zeča (rabbit). Ta asimetrija je bitan izazov jer model sklono naučiti da ignorira rijetke klase u korist češćih.'),
      p('Za pasmine, situacija je još značajno složenije. Dok neke popularne pasmine pasa i mačaka imaju stotine primjera, za egzotičnije pasmine ili za vrste poput goveda i koze ukupni broj primjera s točnom pasminskom oznakom može biti manji od 50 po kategoriji.'),

      h2('2.3. Oznake, klase i struktura datoteka'),
      p('Podaci su označeni na dvije razine. Na razini vrste koristi se kanonski identifikator vrste (npr. dog, cat, horse), a na razini pasmine koristi se kompozitni identifikator u obliku vrsta:pasmina (npr. dog:labrador_retriever, cat:persian). Ovaj format izbjegava kolizije — pasmina „mixed“ može postojati za različite vrste bez zabune.'),
      p('Struktura direktorija projekta organizirana je prema načelu nepromjenjivih originalnih podataka:'),
      bullet('data/raw/ — originalni, nepromijenjeni ulazni podaci iz svih izvora'),
      bullet('data/raw/yolo_sources/ — YOLO-formatirani skupovi podataka'),
      bullet('data/processed/species_aware/ — obrađeni podaci za treniranje prvog stupnja'),
      bullet('data/processed/breed_runs/ — privremeni skupovi za treniranje drugog stupnja, po vrstama'),
      bullet('runs/detect/ — rezultati treniranja YOLO modela (checkpoint, metrike, konfiguracija)'),
      bullet('results/reports/ — metrike evaluacije, izvještaji i vizualizacije'),
      bullet('src/animal_breed/ — izvorni kod projekta (pipeline, webapp, konfiguracije)'),
      spacer(),

      h2('2.4. Eksplorativna analiza'),
      p('Eksplorativna analiza provedena je na razini cijelog skupa i na razini pojedinih izvora. Identificirani su sljedeći ključni obrasci i problemi:'),
      p('Neuravnoteženost razreda: Omjer između najzastupljenije i najmanje zastupljene klase iznosi više od 10:1 za neke parove vrsta. To je dovoljno da model „nauči“ ignorirat rijetke kategorije bez eksplicitnih mjera za balansiranje.'),
      p('Nekonzistentne oznake: Isti pas životinjske pasmine pojavljuje se pod različitim imenima u različitim izvorima (npr. „golden retriever“, „Golden Retriever“, „golden_retriever“). Bez normalizacije, model bi te varijante tretirao kao različite klase.'),
      p('Premali okviri detekcije: U sintetski generiranim slikama dio okvira je manji od 20–25 piksela u jednoj dimenziji. Takvi okviri ne sadrže dovoljno vizualne informacije za smisleno učenje i izbačeni su iz trening skupa.'),
      p('Kvaliteta slike: Dio fotografija je loše kvalitete: preeksponiran, presitan objekt, snažan šum. Implementirani su kriteriji za filtriranje na temelju veličine okvira i rezolucije slike.'),

      h2('2.5. Predobrada i augmentacije'),
      p('Predobrada je bila jedan od vremenski najintenzivnijih dijelova projekta. Za detekciju, slike se konvertiraju u YOLO format (normalizirane koordinate okvira). Za klasifikaciju pasmina, koristi se crop na temelju detekcije iz prve faze uz dodatni marginski pomak od 10–15 % kako bi se uključio kontekst oko životinje.'),
      p('Sintetske slike generirane su kompozitnom metodom. Izrezane životinje iz poznatih slika postavljaju se na raznolike pozadine pri čemu su testirani različiti parametri: broj objekata po slici (1–4), međusobni razmak objekata i različiti kutovi skaliranja. Ova metoda značajno povećava veličinu trening skupa za rijetke kategorije.'),
      p('Augmentacije korištene tijekom treniranja uključuju:'),
      bullet('Horizontalni i vertikalni flip'),
      bullet('Slučajna rotacija (±15°)'),
      bullet('Promjena ekspozicije i kontrasta (±20 %)'),
      bullet('Slučajno izrezivanje i promjena mjerila (scale jitter)'),
      bullet('Mosaic augmentacija (YOLO-specifična, kombinira 4 slike u jednu)'),
      bullet('MixUp i CutMix za klasifikacijski model'),
      spacer(),

      // ── 3. METODE ────────────────────────────────────────────────────────
      h1('3. Opis korištenih metoda i algoritama'),
      h2('3.1. YOLO detekcija vrsta životinja'),
      p('Prva faza sustava temelji se na YOLO (You Only Look Once) arhitekturi za detekciju objekata u realnom vremenu. Koristi se verzija YOLO11s (small variant iz YOLO11 generacije), koja nudi dobar kompromis između točnosti i brzine zaključivanja (engl. inference).'),
      p('YOLO dijeli ulaznu sliku na mrežu ćelija i za svaku ćeliju predviđa više okvira kandidata zajedno s klasifikacijskim distribucijama. Ključne prednosti su: jednofazna arhitektura (ne treba zasebni region proposal network), brzo zaključivanje pogodno za interaktivne primjene, i dobra generalizacija uz dovoljno podataka.'),
      p('Treniranje je provedeno s parametrima:'),

      new Paragraph({ spacing: { after: 80, before: 160 }, children: [bold('Tablica 2. Ključni hiperparametri YOLO species detektora')] }),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [3500, 2763, 2763],
        rows: [
          new TableRow({ children: [hdrCell('Parametar', 3500), hdrCell('Vrijednost', 2763), hdrCell('Napomena', 2763)] }),
          new TableRow({ children: [dataCell('Bazni model', 3500), dataCell('yolo11s.pt', 2763), dataCell('Pretrained na COCO', 2763)] }),
          new TableRow({ children: [dataCell('Ulazna rezolucija', 3500, true), dataCell('640 × 640 px', 2763, true), dataCell('Standard za YOLO', 2763, true)] }),
          new TableRow({ children: [dataCell('Broj epoha', 3500), dataCell('100', 2763), dataCell('', 2763)] }),
          new TableRow({ children: [dataCell('Batch size', 3500, true), dataCell('16', 2763, true), dataCell('', 2763, true)] }),
          new TableRow({ children: [dataCell('Workers', 3500), dataCell('8', 2763), dataCell('Paralelno učitavanje', 2763)] }),
          new TableRow({ children: [dataCell('Patience (early stop)', 3500, true), dataCell('25', 2763, true), dataCell('', 2763, true)] }),
          new TableRow({ children: [dataCell('Optimizer', 3500), dataCell('auto (SGD/Adam)', 2763), dataCell('YOLO automatski odabire', 2763)] }),
          new TableRow({ children: [dataCell('Početna stopa učenja', 3500, true), dataCell('0.01', 2763, true), dataCell('lr0', 2763, true)] }),
          new TableRow({ children: [dataCell('Raspored stope učenja', 3500), dataCell('cosine annealing', 2763), dataCell('cos_lr = true', 2763)] }),
          new TableRow({ children: [dataCell('Mosaic zatvaranje', 3500, true), dataCell('close_mosaic = 10', 2763, true), dataCell('Isključuje se 10 epoha prije kraja', 2763, true)] }),
          new TableRow({ children: [dataCell('Mješovita preciznost', 3500), dataCell('amp = true', 2763), dataCell('Brze treniranje', 2763)] }),
        ]
      }),
      spacer(),
      p('Na izlazu, detektor daje skup okvira s pridruženim klasama i ocjenama pouzdanosti (engl. confidence scores). Primjenjuje se NMS (Non-Maximum Suppression) kako bi se uklonila preklapajuća predviđanja, a koristi se prag pouzdanosti od 0.25 za filtriranje slabih detekcija.'),

      h2('3.2. EfficientNetB2 za klasifikaciju pasmina'),
      p('Drugi stupanj sustava koristi EfficientNetB2 arhitekturu, specifično pretrained varijantu tf_efficientnet_b2.in1k iz timm biblioteke (PyTorch Image Models). EfficientNet obitelji modeli koriste tehniku compound scaling — istovremeno skaliranje dubine, širine i rezolucije mreže — što im omogućuje ostvarivanje visoke točnosti uz relativno mali broj parametara.'),
      p('EfficientNetB2 ima oko 9,1 milijuna parametara, što ga čini znatno kompaktnijim od alternativa kao što su ResNet-50 (25,6 M) ili ViT-B/16 (86 M), uz usporedive rezultate na standardnim benchmarkovima. Ova kompaktnost posebno je bitna u ovom projektu jer se trenira zasebni model za svaku vrstu.'),
      p('Ulaz u model nije cijela fotografija, nego crop koji dolazi iz detekcije prve faze. Time se eliminira utjecaj pozadine i model se može fokusirati isključivo na vizualne detalje životinje.'),

      new Paragraph({ spacing: { after: 80, before: 160 }, children: [bold('Tablica 3. Hiperparametri EfficientNetB2 breed klasifikatora')] }),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [3500, 2763, 2763],
        rows: [
          new TableRow({ children: [hdrCell('Parametar', 3500), hdrCell('Vrijednost', 2763), hdrCell('Obrazloženje', 2763)] }),
          new TableRow({ children: [dataCell('Bazni model', 3500), dataCell('tf_efficientnet_b2.in1k', 2763), dataCell('Pretrained na ImageNet-1k', 2763)] }),
          new TableRow({ children: [dataCell('Ulazna rezolucija', 3500, true), dataCell('320 × 320 px', 2763, true), dataCell('Nativna rezolucija B2', 2763, true)] }),
          new TableRow({ children: [dataCell('Broj epoha', 3500), dataCell('60', 2763), dataCell('Sa early stopping', 2763)] }),
          new TableRow({ children: [dataCell('Batch size', 3500, true), dataCell('48', 2763, true), dataCell('', 2763, true)] }),
          new TableRow({ children: [dataCell('Optimizer', 3500), dataCell('AdamW', 2763), dataCell('Bolja generalizacija od Adam', 2763)] }),
          new TableRow({ children: [dataCell('Stopa učenja', 3500, true), dataCell('0.0003', 2763, true), dataCell('', 2763, true)] }),
          new TableRow({ children: [dataCell('Weight decay', 3500), dataCell('0.05', 2763), dataCell('L2 regularizacija', 2763)] }),
          new TableRow({ children: [dataCell('Raspored', 3500, true), dataCell('cosine annealing', 2763, true), dataCell('', 2763, true)] }),
          new TableRow({ children: [dataCell('Label smoothing', 3500), dataCell('0.1', 2763), dataCell('Smanjuje overconfidence', 2763)] }),
          new TableRow({ children: [dataCell('MixUp α', 3500, true), dataCell('0.2', 2763, true), dataCell('Interpolacija primjera', 2763, true)] }),
          new TableRow({ children: [dataCell('CutMix α', 3500), dataCell('1.0', 2763), dataCell('Izrezivanje i spajanje primjera', 2763)] }),
          new TableRow({ children: [dataCell('Early stopping patience', 3500, true), dataCell('10 epoha', 2763, true), dataCell('Min. poboljanje 0.001', 2763, true)] }),
        ]
      }),
      spacer(),
      p('Trening je modularan: za svaku vrstu koja ima dovoljno podataka trenira se zasebni breed model. Time se znatno smanjuje broj klasa po modelu (npr. 15–30 pasmina pasa umjesto 200+ pasmina svih vrsta zajedno), što olakšava učenje i smanjuje konfuziju između vizualno sličnih pasmina različitih vrsta.'),

      h2('3.3. Direktni YOLO model za pasmine'),
      p('Uz dvostupanjski pipeline, u projektu je implementiran i direktni YOLO model za pasmine koji služi kao kontrolni eksperiment. Ovaj model pokušava riješiti cijeli zadatak u jednom koraku: detekciju objekta i klasifikaciju pasmine istovremeno, bez prethodne faze prepoznavanja vrste.'),
      p('Prednost ovog pristupa je konceptualna jednostavnost — jedan model, jedan korak zaključivanja. Međutim, nedostaci su značajni:'),
      bullet('Puno veći broj klasa (pasmine > vrste), što otježava treniranje'),
      bullet('Slabija početna separacija klasa jer vizualne razlike između pasmina su suptilnije'),
      bullet('Veća osjetljivost na neuravnoteženost jer rijetke pasmine imaju još manje primjera'),
      bullet('Nema koristi od hijerarhijske strukture taksonomije'),
      spacer(),
      p('Konfiguracijski, ovaj model koristi yolo11s.pt kao početnu točku, ulaznu rezoluciju 640 × 640 piksela i 150 epoha treniranja (više od species modela zbog veće složenosti zadatka). Rezultati ovog eksperimenta usporeduju se s dvostupanjskim pristupom u poglavlju 4.'),

      h2('3.4. Višerazinski pipeline — arhitektura integracije'),
      p('Cjelokupni pipeline integrira obje faze u koherentnu cjelinu. Tok podataka pri inferenciji je sljedeći:'),
      bullet('Ulazna fotografija proslijeđuje se YOLO species detektoru'),
      bullet('Za svaki detektirani okvir s pouzdanošću iznad praga (0.25), vrši se crop s dodatnim marginom'),
      bullet('Izrezani isječak proslijeđuje se odgovarajućem breed klasifikatoru za tu vrstu'),
      bullet('Ako breed model za tu vrstu nije dostupan, vraća se samo species predikcija'),
      bullet('Izlaz sustava su anotirana slika i strukturirani JSON s detaljima svake detekcije'),
      spacer(),
      p('Ovaj dizajn omogućuje postupno dodavanje breed modela bez promjene prve faze, što je važno za iterativni razvoj kakav se prakticira u ovom projektu.'),

      h2('3.5. Lokalna web aplikacija za inferenciju'),
      p('Uz CLI ulazne točke, projekt uključuje lokalnu web aplikaciju smještenu u src/animal_breed/webapp/. Aplikacija je implementirana kao jednostavan Python HTTP server koji izlaže REST API i statiku za preglednik.'),
      p('Ključni API endpointovi:'),
      bullet('/api/status (GET) — vraća informacije o učitanim modelima, broj dostupnih breed modela i popis vrsta za koje je breed klasifikacija dostupna'),
      bullet('/api/infer (POST) — prima multipart/form-data s uploadanom slikom, pokreće full pipeline i vraća JSON s detekcijama i anotiranom slikom (base64)'),
      spacer(),
      p('Prednje sučelje (frontend) prikazuje uploadanu sliku s iscrtanim okvirima, boju-kodiranim labelama i tablicom predikcija. Tehnički, aplikacija koristi postojeći shared species detector singleton i, ako su dostupni, species-specifične breed modele koji se učitavaju pri pokretanju.'),
      p('Ovaj sloj je važan jer projekt sada nije samo kolekcija skripti i model checkpoint datoteka, nego sustav koji se može demonstrirati i koristiti kroz pregledno sučelje bez poznavanja interne strukture koda.'),

      // ── 4. EVALUACIJA ────────────────────────────────────────────────────
      h1('4. Evaluacija izgrađenih modela i algoritama'),
      h2('4.1. Korištene metrike'),
      p('Za YOLO detektor korištene su standardne metrike za detekciju objekata:'),
      bullet('Precision (P): udio ispravnih detekcija među svim predviđenim detekcijama. Visoka preciznost znači malo lažno pozitivnih.'),
      bullet('Recall (R): udio stvarnih objekata koje je model uspio pronaći. Visoki recall znači malo lažno negativnih.'),
      bullet('mAP@50: srednja prosječna preciznost (mean Average Precision) pri pragu preklopa (IoU) od 0,50. Standardna metrika za praćenje napretka.'),
      bullet('mAP@50–95: mAP izračunata za IoU pragove od 0,50 do 0,95 u koraku 0,05. Strožija i realnija mjera koja kažniava neprecizne okvire.'),
      spacer(),
      p('Za EfficientNetB2 klasifikator korištene su klasifikacijske metrike:'),
      bullet('Top-1 točnost: udio uzoraka gdje je točna klasa na prvom mjestu u poretku predikcija.'),
      bullet('Top-5 točnost: udio uzoraka gdje je točna klasa u prvih 5 predikcija. Relevantno za situacije s velikim brojem vizualno sličnih klasa.'),
      bullet('Križnoa entropija gubitak (cross-entropy loss): praćena zasebno za trening i validacijski skup kroz epohe.'),
      bullet('Matrica zabune (confusion matrix): omogućuje vizualnu identifikaciju koje pasmine model međusobno zamjenjuje.'),
      spacer(),

      h2('4.2. Rezultati species detektora'),
      p('Provedena su dva eksperimenta s različitim verzijama skupa podataka:'),

      new Paragraph({ spacing: { after: 80, before: 160 }, children: [bold('Tablica 4. Rezultati species detektora — usporedba eksperimenata')] }),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [3026, 1500, 1500, 1500, 1500],
        rows: [
          new TableRow({ children: [hdrCell('Eksperiment', 3026), hdrCell('Precision', 1500), hdrCell('Recall', 1500), hdrCell('mAP@50', 1500), hdrCell('mAP@50–95', 1500)] }),
          new TableRow({ children: [dataCell('Inicijalni trening (v1)', 3026), dataCell('0.6735', 1500), dataCell('0.6051', 1500), dataCell('0.6144', 1500), dataCell('0.3941', 1500)] }),
          new TableRow({ children: [dataCell('Prošireni skup, bolje balansiranje (v2)', 3026, true), dataCell('0.7069', 1500, true), dataCell('0.6822', 1500, true), dataCell('0.7236', 1500, true), dataCell('0.4073', 1500, true)] }),
          new TableRow({ children: [dataCell('Poboljanje (v2 − v1)', 3026), dataCell('+0.0334', 1500), dataCell('+0.0771', 1500), dataCell('+0.1092', 1500), dataCell('+0.0132', 1500)] }),
        ]
      }),
      spacer(),
      p('Rezultati pokazuju konzistentno poboljanje u svim metrikama između prve i druge verzije. Najveći skok je u mAP@50 (+10,9 postotnih bodova), što ukazuje da su poboljšana kvaliteta i balansiranje skupa podataka imali značajan učinak. Recall je također značajno porastao, što sugerira da model u drugoj verziji rjeđe greši propustići stvarne objekte.'),
      p('Apsolutne vrijednosti metrika su solidne za problem s 18 klasa, ali ne izvrsne. Precision od oko 0.71 i recall od oko 0.68 ukazuju da postoji prostor za poboljanje, posebno na zahtjevnijim klasama s malo podataka. Strožija mjera mAP@50–95 oko 0.40 odražava činjenicu da okviri detekcija nisu uvijek perfektno poravnati s rubovima životinja, što je očekivano za slike iz prirode.'),

      h2('4.3. Rezultati drugog stupnja — breed klasifikacija'),
      p('Drugi stupanj sada je pokrenuta i trenira se modularno po vrstama. Za svaku vrstu s dovoljno podataka trenira se zaseban breed model. Ovo je ključna razlika u odnosu na jedan monolitni klasifikator: broj klasa po modelu je kontroliran, što olakšava učenje.'),
      p('Najraniji dostupni rezultat je za vrstu koza (goat), koji pokazuje brzu konvergenciju već u prvoj epohi:'),

      new Paragraph({ spacing: { after: 80, before: 160 }, children: [bold('Tablica 5. Rezultati breed klasifikatora za vrstu „goat“ (epoha 1)')] }),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [2256, 1693, 1693, 1692, 1692],
        rows: [
          new TableRow({ children: [hdrCell('Metrika', 2256), hdrCell('Trening loss', 1693), hdrCell('Trening top-1', 1693), hdrCell('Val. loss', 1692), hdrCell('Val. top-1', 1692)] }),
          new TableRow({ children: [dataCell('Vrijednost (epoha 1)', 2256), dataCell('1.6262', 1693), dataCell('62.78 %', 1693), dataCell('1.2387', 1692), dataCell('78.07 %', 1692)] }),
        ]
      }),
      spacer(),
      p('Val. top-1 od 78 % već u prvoj epohi je vrlo dobar signal. Najvažnije, val. top-5 iznosi 98.13 %, što znači da model gotovo uvijek ima točnu pasminu među prvih 5 prijedloga. Ovo je posebno bitno za aplikacije gdje je breed klasifikacija pomoćni alat, a ne konačna presuda.'),
      p('Konačne breed metrike za sve dostupne vrste još nisu zatvorene — trening je u tijeku. Ovo je eksplicitno dokumentirano kako bi se izbjeglo navijavanje neutemeljenih rezultata. Kada trening završi, ovaj odjeljak treba nadopuniti tablicama početnih, srednji i konačnih metrika za svaki species model.'),

      h2('4.4. Usporedba dvostupanjskog i direktnog pristupa'),
      p('Dvostupanjski pristup i direktni YOLO breed model usporeduju se na temelju kompleksnosti treniranja i kvalitete rezultata. Ključne razlike su:'),

      new Paragraph({ spacing: { after: 80, before: 160 }, children: [bold('Tablica 6. Usporedba dvostupanjskog i direktnog pristupa')] }),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [2500, 3263, 3263],
        rows: [
          new TableRow({ children: [hdrCell('Dimenzija', 2500), hdrCell('Dvostupanjski pipeline', 3263), hdrCell('Direktni YOLO breed', 3263)] }),
          new TableRow({ children: [dataCell('Broj klasa', 2500), dataCell('18 (species) + ~15-30 (breed/vrsti)', 3263), dataCell('Sve pasmine odjednom (100+)', 3263)] }),
          new TableRow({ children: [dataCell('Kompleksnost učenja', 2500, true), dataCell('Manja po modelu', 3263, true), dataCell('Visoka', 3263, true)] }),
          new TableRow({ children: [dataCell('Interpretabilnost', 2500), dataCell('Visoka (znamo vrstu i pasminu)', 3263), dataCell('Srednja (samo pasmina)', 3263)] }),
          new TableRow({ children: [dataCell('Infrastruktura', 2500, true), dataCell('Složenija (više modela)', 3263, true), dataCell('Jednostavnija (1 model)', 3263, true)] }),
          new TableRow({ children: [dataCell('Robusnost na rijetke klase', 2500), dataCell('Bolja (kondicioniranje na vrstu)', 3263), dataCell('Slabija', 3263)] }),
          new TableRow({ children: [dataCell('Brzina inferencije', 2500, true), dataCell('Dva prolaza kroz mrežu', 3263, true), dataCell('Jedan prolaz', 3263, true)] }),
        ]
      }),
      spacer(),
      p('Preliminarni zaključak je da dvostupanjski pristup ima značajnu prednost u robusnosti i interpretabilnosti, a cijena je nešto složenija infrastruktura i duže zaključivanje. Za interaktivne demonstracijske primjene, obje brzine su sasvim prihvatljive.'),

      // ── 5. DEMONSTRACIJA ─────────────────────────────────────────────────
      h1('5. Demonstracija rada'),
      p('Sustav je namijenjen za korištenje na dva načina: kroz naredbeni redak (CLI) i kroz lokalnu web aplikaciju. Oba načina temelje se na istom kodu za inferenciju.'),

      h2('5.1. CLI ulazne točke'),
      p('U pyproject.toml definirane su ključne CLI naredbe koje pokrivaju sve korake pipeline-a:'),
      code('# Izgradnja kanonskog manifesta iz sirovih izvora'),
      code('animal-breed build-manifest'),
      code(''),
      code('# Validacija taksonomije i konzistentnosti oznaka'),
      code('animal-breed validate-taxonomy'),
      code(''),
      code('# Priprema YOLO-formatiranog skupa za species detektor'),
      code('animal-breed prepare-yolo-dataset'),
      code(''),
      code('# Treniranje species detektora'),
      code('animal-breed train-species-detector'),
      code(''),
      code('# Treniranje breed klasifikatora za sve vrste'),
      code('animal-breed train-breed-classifiers'),
      code(''),
      code('# Inferencija na jednoj slici (dvostupanjski)'),
      code('animal-breed infer --image path/to/image.jpg'),
      code(''),
      code('# Pokretanje web aplikacije'),
      code('animal-breed serve'),
      spacer(),

      h2('5.2. Web aplikacija — tok rada'),
      p('Korisnik pokreće lokalnu aplikaciju naredbom animal-breed serve i otvara preglednik na localhost:8080. Tok rada je sljedeći:'),
      bullet('Odabir fotografije klikom na polje za upload ili povlačenjem datoteke'),
      bullet('Klik na „Run Inference“ pokreće pipeline na serveru'),
      bullet('Povrat anotiranih rezultata: slika s iscrtanim okvirima i labelama'),
      bullet('Tablica detekcija prikazuje: detektiranu vrstu, pouzdanost, koordinate okvira i, ako je dostupno, pasminu s njezinim top-3 predikcijama'),
      bullet('Status panel prikazuje koji breed modeli su učitani i za koje su vrste dostupni'),
      spacer(),
      p('Ako breed model za detektiranu vrstu nije dostupan, aplikacija to jasno prikazuje („Breed model not available for this species“) umjesto lažne predikcije. Ovo je bitna dizajnerska odluka — bolje je eksplicitno reći da ne znamo nego nagađati.'),
      mixed(italic('Napomena: ovdje preporučljivo umetnuti screenshot web aplikacije i primjer anotiranog izlaza u konačnoj verziji predaje.')),

      h2('5.3. Primjer analize jedne slike'),
      p('Za testnu fotografiju koja prikazuje dva psa u parku, očekivani izlaz sustava je:'),
      code('Detection 1:'),
      code('  species: dog (confidence: 0.94)'),
      code('  bounding_box: [142, 87, 498, 412]'),
      code('  breed: golden_retriever (confidence: 0.82)'),
      code('  breed_top3: [golden_retriever, labrador_retriever, nova_scotia_duck_tolling_retriever]'),
      code(''),
      code('Detection 2:'),
      code('  species: dog (confidence: 0.91)'),
      code('  bounding_box: [601, 124, 889, 445]'),
      code('  breed: border_collie (confidence: 0.76)'),
      code('  breed_top3: [border_collie, australian_shepherd, shetland_sheepdog]'),
      spacer(),

      // ── 6. ZAKLJUČAK ───────────────────────────────────────────────────────
      h1('6. Zaključak'),
      p('U ovom projektu razvijen je višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina koji demonstrira više bitnih aspekata modernog strojnog učenja: hijerarhijsku dekompoziciju složenog zadatka, integraciju heterogenih izvora podataka, modularan dizajn pipeline-a i praktičnu demonstraciju kroz web sučelje.'),
      p('Ključni postignuti ciljevi su:'),
      bullet('Funkcionalan species detektor s mAP@50 od 0.724 na proširenom skupu podataka, što je solidan rezultat za 18-klasni problem detekcije'),
      bullet('Modularan breed klasifikator koji pokazuje brzu konvergenciju (val. top-1 > 78 % u prvoj epohi za kozu)'),
      bullet('Kanonski manifest koji standardizira podatke iz više heterogenih izvora i omogućuje reproducibilnost'),
      bullet('Operativna web aplikacija koja sustav čini dostupnim bez poznavanja interne arhitekture koda'),
      bullet('Usporedni eksperiment s direktnim YOLO breed modelom koji kvantificira vrijednost hijerarhijskog pristupa'),
      spacer(),
      p('Glavna ograničenja koja su identificirana su: neuravnoteženost razreda u izvornim podacima (omjer do 10:1 između klasa), ograničen broj pasminski označenih primjera za rijetke vrste i činjenica da konačne breed metrike za sve vrste još nisu zatvorene. Ova ograničenja su dokumentirana transparentno jer je iskrena procjena stanja projekta jednako vrijedno kao i sami rezultati.'),
      p('Budući pravci razvoja uključuju: završetak treniranja breed modela za sve dostupne vrste s dovoljno podataka, evaluaciju na neovisnom testnom skupu koji nije korišten u razvoju, prikupljanje dodatnih podataka za slabo zastupljene pasmine, te istraživanje korištenja većih modela (YOLO11m/l) ili fine-tuninga ViT modela za breed klasifikaciju.'),
      p('Projekt demonstrira kako je u računalnom vidu kvaliteta podataka i arhitekturalna struktura jednako važna kao i odabir konkretnog algoritma. Studentski napredak u ovom projektu nije mjerljiv samo konačnim metrikama, nego razumijevanjem kako postaviti reproducibilan ML pipeline, kako interpretirati metrike i kako argumentirano dokumentirati i ono što radi dobro i ono što još treba doraditi.'),

      // ── 7. LITERATURA ────────────────────────────────────────────────────
      h1('7. Korištene tehnologije i literatura'),
      h2('7.1. Korištene tehnologije'),
      new Paragraph({ spacing: { after: 80, before: 160 }, children: [bold('Tablica 7. Pregled korištenih tehnologija')] }),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [2500, 2500, 4026],
        rows: [
          new TableRow({ children: [hdrCell('Tehnologija', 2500), hdrCell('Verzija', 2500), hdrCell('Korištenje u projektu', 4026)] }),
          new TableRow({ children: [dataCell('Python', 2500), dataCell('≥3.10', 2500), dataCell('Primarni programski jezik', 4026)] }),
          new TableRow({ children: [dataCell('PyTorch', 2500, true), dataCell('≥2.0', 2500, true), dataCell('Duboko učenje, breed klasifikator', 4026, true)] }),
          new TableRow({ children: [dataCell('Ultralytics YOLO11', 2500), dataCell('yolo11s', 2500), dataCell('Species detektor i direktni breed model', 4026)] }),
          new TableRow({ children: [dataCell('timm (EfficientNetB2)', 2500, true), dataCell('tf_efficientnet_b2.in1k', 2500, true), dataCell('Breed klasifikator', 4026, true)] }),
          new TableRow({ children: [dataCell('OpenCV', 2500), dataCell('≥4.0', 2500), dataCell('Obrada slika, crop, vizualizacija', 4026)] }),
          new TableRow({ children: [dataCell('NumPy / Pandas', 2500, true), dataCell('Latest', 2500, true), dataCell('Obrada podataka i manifesta', 4026, true)] }),
          new TableRow({ children: [dataCell('AMD ROCm', 2500), dataCell('Latest', 2500), dataCell('GPU runtime za treniranje', 4026)] }),
          new TableRow({ children: [dataCell('Python http.server', 2500, true), dataCell('Stdlib', 2500, true), dataCell('Lokalna web aplikacija', 4026, true)] }),
        ]
      }),
      spacer(),

      h2('7.2. Literatura'),
      p('[1] Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics. Ultralytics. Dostupno na: https://github.com/ultralytics/ultralytics'),
      p('[2] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Proceedings of the 36th International Conference on Machine Learning (ICML 2019). arXiv:1905.11946.'),
      p('[3] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. Proceedings of CVPR 2016.'),
      p('[4] Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.'),
      p('[5] Rw Wightman. (2019). PyTorch Image Models (timm). GitHub. Dostupno na: https://github.com/rwightman/pytorch-image-models'),
      p('[6] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.'),
      p('[7] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS 2012.'),
      p('[8] Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The Pascal Visual Object Classes (VOC) Challenge. International Journal of Computer Vision, 88(2), 303–338.'),

      h2('Kontrolni popis za predaju'),
      bullet('Seminar u PDF obliku'),
      bullet('Izvorni programski kod u repozitoriju'),
      bullet('Prezentacija (PPTX ili PDF)'),
      bullet('Screenshot web aplikacije i primjer anotiranog izlaza'),
      bullet('Konačne breed metrike kada trening završi'),
      bullet('Popunjena polja: ime i prezime autora, naziv kolegija'),
      spacer(),
      new Paragraph({
        spacing: { after: 160 },
        children: [new TextRun({
          text: 'Napomena: mjesta označena s [OVDJE DODATI ...] treba ručno dopuniti prije predaje.',
          font: 'Arial', size: 20, italics: true, color: GRAY
        })]
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('seminar_projekt_strojno_ucenje_EXPANDED.docx', buffer);
  console.log('Done: seminar_projekt_strojno_ucenje_EXPANDED.docx');
}).catch(err => { console.error(err); process.exit(1); });
