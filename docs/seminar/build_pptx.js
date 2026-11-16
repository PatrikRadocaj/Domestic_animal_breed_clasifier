// ── Višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina ──────
// Full presentation — rebuilt from scratch
// ──────────────────────────────────────────────────────────────────────────
const pptxgen = require('pptxgenjs');
const fs      = require('fs');
const path    = require('path');

const pres = new pptxgen();
pres.layout  = 'LAYOUT_16x9';
pres.title   = 'Višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina';
pres.author  = 'Projekt iz strojnog učenja';

// ── colour / font constants ──────────────────────────────────────────────
const NAVY   = '0F172A';  // dark bg
const NAVYM  = '1E293B';  // slightly lighter navy
const WHITE  = 'FFFFFF';
const TEAL   = '0D9488';
const TEAL2  = '14B8A6';
const AMBER  = 'F59E0B';
const SLATE  = '334155';  // body text
const MUTED  = '64748B';  // secondary text
const LIGHT  = 'F8FAFC';  // slide bg
const CARD   = 'FFFFFF';
const BORD   = 'E2E8F0';
const GREEN  = '16A34A';
const RED2   = '991B1B';

const FF = 'Calibri';

const BASE = __dirname + path.sep;

// ── helpers ──────────────────────────────────────────────────────────────
function imgPath(name) { return BASE + name; }

// Card box (white rect + optional left accent bar)
function card(slide, x, y, w, h, accentColor) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: CARD },
    line: { color: BORD, width: 0.5 },
    shadow: { type: 'outer', color: '000000', opacity: 0.06, blur: 8, offset: 3, angle: 135 }
  });
  if (accentColor) {
    slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h, fill: { color: accentColor }, line: { color: accentColor, width: 0 } });
  }
}

// Dark section divider
function darkCard(slide, x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: NAVYM }, line: { color: NAVYM, width: 0 } });
}

// Stat box: big number + label
function statBox(slide, x, y, w, h, value, label, accent) {
  card(slide, x, y, w, h, accent);
  slide.addText(value, { x: x + 0.12, y: y + 0.10, w: w - 0.18, h: 0.55,
    fontFace: FF, fontSize: 28, bold: true, color: SLATE, margin: 0 });
  slide.addText(label, { x: x + 0.12, y: y + 0.60, w: w - 0.18, h: 0.50,
    fontFace: FF, fontSize: 10, color: MUTED, margin: 0 });
}

// Small tag / chip
function tag(slide, x, y, label, bg, fg) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: label.length * 0.11 + 0.18, h: 0.24, fill: { color: bg }, line: { color: bg, width: 0 } });
  slide.addText(label, { x, y, w: label.length * 0.11 + 0.18, h: 0.24,
    fontFace: FF, fontSize: 9, bold: true, color: fg, align: 'center', valign: 'middle', margin: 0 });
}

// Section pill at top-left on content slides
function sectionPill(slide, label) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 0.18, w: label.length * 0.10 + 0.28, h: 0.27, fill: { color: TEAL }, line: { color: TEAL, width: 0 } });
  slide.addText(label, { x: 0.35, y: 0.18, w: label.length * 0.10 + 0.28, h: 0.27,
    fontFace: FF, fontSize: 9.5, bold: true, color: WHITE, align: 'center', valign: 'middle', margin: 0 });
}

// Slide title on light slides
function slideTitle(slide, title) {
  slide.addText(title, { x: 0.35, y: 0.55, w: 9.3, h: 0.55,
    fontFace: FF, fontSize: 22, bold: true, color: SLATE, margin: 0 });
  slide.addShape(pres.shapes.LINE, { x: 0.35, y: 1.10, w: 9.3, h: 0, line: { color: BORD, width: 0.5 } });
}

// Bullet text helper
function bullets(slide, x, y, w, h, items, size = 12, color = SLATE) {
  const runs = items.map((txt, i) => ({
    text: txt,
    options: { bullet: true, fontSize: size, fontFace: FF, color, paraSpaceAfter: 4,
                breakLine: i < items.length - 1 }
  }));
  slide.addText(runs, { x, y, w, h, valign: 'top' });
}

// Arrow between pipeline boxes
function arrow(slide, x, y, color = AMBER) {
  slide.addShape(pres.shapes.LINE, { x, y, w: 0.35, h: 0, line: { color, width: 2 } });
  // arrowhead as a small triangle
  slide.addShape(pres.shapes.RECTANGLE, { x: x + 0.30, y: y - 0.07, w: 0.08, h: 0.14, fill: { color }, line: { color, width: 0 } });
}

// ═══════════════════════════════════════════════════════════════════════
// SLIDE 1 — TITLE
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // decorative teal bar top
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.07, fill: { color: TEAL }, line: { color: TEAL, width: 0 } });

  // left accent column
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0.07, w: 0.08, h: 5.555, fill: { color: TEAL2 }, line: { color: TEAL2, width: 0 } });

  // main title
  s.addText('Višerazinski sustav za\nprepoznavanje životinjskih vrsta\ni pasmina pomoću dubokog učenja', {
    x: 0.50, y: 0.65, w: 5.8, h: 2.5,
    fontFace: FF, fontSize: 28, bold: true, color: WHITE, valign: 'top'
  });

  // subtitle
  s.addText('Projekt iz strojnog učenja i računalnog vida', {
    x: 0.50, y: 3.20, w: 5.8, h: 0.45,
    fontFace: FF, fontSize: 14, italic: true, color: TEAL2, margin: 0
  });

  // tech tags
  const tags = ['Python', 'YOLO11', 'EfficientNetV2-S', 'PyTorch', 'OpenCV'];
  let tx = 0.50;
  tags.forEach(t => { tag(s, tx, 3.78, t, NAVYM, TEAL2); tx += t.length * 0.11 + 0.28; });

  // right side — pipeline visual
  // boxes
  const boxes = [
    { label: 'Ulazna slika', sub: 'fotografija', col: '1E3A5F', tx: 6.20, ty: 0.90 },
    { label: 'YOLO11s', sub: 'species detector\n640 px', col: '0F766E', tx: 6.20, ty: 1.82 },
    { label: 'Crop životinje', sub: 'izrez + kontekst\n10-15% margina', col: '0F766E', tx: 6.20, ty: 2.88 },
    { label: 'EfficientNetV2-S', sub: 'breed classifier\n384 px', col: '065F46', tx: 6.20, ty: 3.94 },
  ];
  boxes.forEach(b => {
    s.addShape(pres.shapes.RECTANGLE, { x: b.tx, y: b.ty, w: 3.45, h: 0.78, fill: { color: b.col }, line: { color: WHITE, width: 0.3 } });
    s.addText(b.label, { x: b.tx + 0.14, y: b.ty + 0.05, w: 3.2, h: 0.32, fontFace: FF, fontSize: 13, bold: true, color: WHITE, margin: 0 });
    s.addText(b.sub, { x: b.tx + 0.14, y: b.ty + 0.36, w: 3.2, h: 0.38, fontFace: FF, fontSize: 10, color: TEAL2, margin: 0 });
  });
  // arrows between boxes
  [1.68, 2.74, 3.80].forEach(ay => {
    s.addShape(pres.shapes.LINE, { x: 7.60, y: ay, w: 0, h: 0.14, line: { color: AMBER, width: 2 } });
    s.addText('↓', { x: 7.52, y: ay + 0.03, w: 0.36, h: 0.22, fontFace: FF, fontSize: 12, bold: true, color: AMBER, align: 'center', margin: 0 });
  });

  // meta
  s.addText('[IME I PREZIME]  ·  2026.', {
    x: 0.50, y: 5.15, w: 5.0, h: 0.30, fontFace: FF, fontSize: 10, color: MUTED, margin: 0
  });
}

// ═══════════════════════════════════════════════════════════════════════
// SLIDE 2 — PROBLEM I MOTIVACIJA
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  sectionPill(s, '1. Motivacija');
  slideTitle(s, 'Zašto višerazinski pristup?');

  // Left — problem description cards
  const leftCards = [
    { icon: '🔍', title: 'Nije dovoljno samo detektirati', body: 'Treba razlikovati vrste I pasmine unutar vrste — vizualne razlike mogu biti samo boja dlake.', accent: TEAL },
    { icon: '⚖️', title: 'Neuravnoteženost 900:1', body: 'Cat: 4 542 primjera vs. Duck: 5 primjera — model uči ignorirati rijetke klase.', accent: AMBER },
    { icon: '📂', title: 'Heterogeni podaci', body: 'Više izvora, nekonzistentne oznake, sintetičke slike — čišćenje pipeline-a je pola posla.', accent: '7C3AED' },
  ];
  leftCards.forEach((c, i) => {
    const cy = 1.25 + i * 1.28;
    card(s, 0.35, cy, 4.30, 1.10, c.accent);
    s.addText(c.icon + '  ' + c.title, { x: 0.55, y: cy + 0.08, w: 4.0, h: 0.32, fontFace: FF, fontSize: 12, bold: true, color: SLATE, margin: 0 });
    s.addText(c.body, { x: 0.55, y: cy + 0.40, w: 4.0, h: 0.60, fontFace: FF, fontSize: 10.5, color: MUTED, margin: 0 });
  });

  // Right — the critical experiment result
  darkCard(s, 5.00, 1.18, 4.70, 1.50);
  s.addText('Kontrolni eksperiment', { x: 5.16, y: 1.28, w: 4.40, h: 0.30, fontFace: FF, fontSize: 11, bold: true, color: TEAL2, margin: 0 });
  s.addText('Direktni YOLO breed model', { x: 5.16, y: 1.58, w: 4.40, h: 0.28, fontFace: FF, fontSize: 10.5, color: WHITE, margin: 0 });
  s.addText('mAP@50 = 0.228', { x: 5.16, y: 1.88, w: 2.50, h: 0.52, fontFace: FF, fontSize: 32, bold: true, color: AMBER, margin: 0 });
  s.addText('25 epoha, Recall ≤ 0.22 — daleko ispod dvostupanjskog', { x: 7.56, y: 2.12, w: 2.0, h: 0.30, fontFace: FF, fontSize: 11, italic: true, color: MUTED, margin: 0 });

  // Two stage vs single stage
  card(s, 5.00, 2.82, 4.70, 2.40, TEAL);
  s.addText('Dvostupanjski pristup', { x: 5.18, y: 2.94, w: 4.30, h: 0.30, fontFace: FF, fontSize: 12, bold: true, color: SLATE, margin: 0 });

  const steps = ['1. YOLO11s detektira vrstu i bounding box', '2. Crop životinje s marginom 10-15%', '3. EfficientNetV2-S klasificira pasminu', '4. Rezultat: species + breed + confidence'];
  steps.forEach((st, i) => {
    s.addText(st, { x: 5.18, y: 3.28 + i * 0.42, w: 4.30, h: 0.35, fontFace: FF, fontSize: 10.5, color: SLATE, margin: 0 });
  });
}

// ═══════════════════════════════════════════════════════════════════════
// SLIDE 3 — PODACI / CLASS IMBALANCE
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  sectionPill(s, '2. Podaci');
  slideTitle(s, 'Neuravnoteženost klasa — ključni izazov');

  // Three stat boxes — updated to July 2026 final dataset
  statBox(s, 0.35, 1.25, 2.80, 1.00, '39 615', 'slika ukupno u species skupu', AMBER);
  statBox(s, 3.30, 1.25, 2.80, 1.00, '19', 'klasa vrsta (19 kategorija životinja)', TEAL);
  statBox(s, 6.25, 1.25, 3.40, 1.00, '82% / 11% / 7%', 'train / val / test podjela', '7C3AED');

  // Chart image — regenerated July 2026
  s.addImage({ path: imgPath('species_balance_new.png'), x: 0.35, y: 2.40, w: 7.20, h: 2.95 });

  // Annotation box on the right
  card(s, 7.70, 2.40, 2.00, 2.95, TEAL);
  s.addText('📊 Finalni skup', { x: 7.82, y: 2.52, w: 1.74, h: 0.30, fontFace: FF, fontSize: 11, bold: true, color: SLATE, margin: 0 });
  const issues = ['Cat: 6 476 slika', 'Llama: 347 slika', 'Omjer ~19:1', '(bio 900:1 ✓)', '', 'Rješenje:', 'sintetički kompoziti', '+ PHR slike', '+ balansiran', '  sampling'];
  issues.forEach((t, i) => {
    const isBold = t === 'Rješenje:';
    const isGreen = ['sintetički kompoziti', '+ PHR slike', '+ balansiran', '  sampling'].includes(t);
    const isFixed = t === '(bio 900:1 ✓)';
    s.addText(t, { x: 7.82, y: 2.88 + i * 0.25, w: 1.74, h: 0.25,
      fontFace: FF, fontSize: 9.5, bold: isBold, color: isGreen ? TEAL : (isFixed ? GREEN : SLATE), margin: 0 });
  });
}

// ═══════════════════════════════════════════════════════════════════════
// SLIDE 4 — SINTETIČKI KOMPOZITI
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  sectionPill(s, '2. Podaci');
  slideTitle(s, 'Sintetički kompoziti — HybridSpeciesYoloDataset');

  // Left: explanation
  card(s, 0.35, 1.25, 3.90, 4.00, TEAL);
  s.addText('Proces generiranja', { x: 0.53, y: 1.35, w: 3.60, h: 0.32, fontFace: FF, fontSize: 12, bold: true, color: SLATE, margin: 0 });

  const steps = [
    ['1', 'Izreži životinje iz trening slika koristeći YOLO oznake'],
    ['2', 'Postavi izreske na realnu pozadinsku teksturu u grid rasporedu'],
    ['3', 'Dodaj pozicijski jitter ±6% i skaliranje 0.88–1.20×'],
    ['4', 'Automatski generiraj bounding boxove za svaki isječak'],
  ];
  steps.forEach(([n, t], i) => {
    const cy = 1.78 + i * 0.78;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.53, y: cy, w: 0.32, h: 0.32, fill: { color: TEAL }, line: { color: TEAL, width: 0 } });
    s.addText(n, { x: 0.53, y: cy, w: 0.32, h: 0.32, fontFace: FF, fontSize: 11, bold: true, color: WHITE, align: 'center', valign: 'middle', margin: 0 });
    s.addText(t, { x: 0.96, y: cy, w: 3.16, h: 0.60, fontFace: FF, fontSize: 10.5, color: SLATE, margin: 0 });
  });

  // Ratio
  s.addShape(pres.shapes.RECTANGLE, { x: 0.53, y: 4.90, w: 3.60, h: 0.24, fill: { color: NAVYM }, line: { color: NAVYM, width: 0 } });
  s.addText('40% sintetika  ·  40% real  ·  20% augmentirano', {
    x: 0.53, y: 4.90, w: 3.60, h: 0.24, fontFace: FF, fontSize: 9.5, bold: true, color: TEAL2, align: 'center', valign: 'middle', margin: 0
  });

  // Right: images
  s.addText('Ulazni izresci:', { x: 4.45, y: 1.25, w: 5.20, h: 0.28, fontFace: FF, fontSize: 10.5, color: MUTED, margin: 0 });
  s.addImage({ path: imgPath('synthetic_source_crops.jpg'), x: 4.45, y: 1.56, w: 5.20, h: 1.36 });
  s.addText('↓  compositor  ↓', { x: 4.45, y: 2.97, w: 5.20, h: 0.30, fontFace: FF, fontSize: 10.5, color: AMBER, bold: true, align: 'center', margin: 0 });
  s.addText('Sintetička trening slika s automatskim bbox:', { x: 4.45, y: 3.28, w: 5.20, h: 0.28, fontFace: FF, fontSize: 10.5, color: MUTED, margin: 0 });
  s.addImage({ path: imgPath('synthetic_composite_example.jpg'), x: 4.45, y: 3.58, w: 3.20, h: 1.75 });
  s.addText('Svaka trening iteracija\ngenerira novu kombinaciju', {
    x: 7.74, y: 3.65, w: 1.90, h: 0.90, fontFace: FF, fontSize: 10, italic: true, color: MUTED, margin: 0
  });
}

// ═══════════════════════════════════════════════════════════════════════
// SLIDE 5 — PIPELINE ARHITEKTURA (dark slide)
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  s.addText('3. Pipeline — arhitektura', {
    x: 0.40, y: 0.22, w: 6.0, h: 0.32, fontFace: FF, fontSize: 11, color: TEAL2, margin: 0
  });
  s.addText('Dvostupanjski inferencijski pipeline', {
    x: 0.40, y: 0.58, w: 9.2, h: 0.52, fontFace: FF, fontSize: 24, bold: true, color: WHITE, margin: 0
  });

  // Pipeline boxes — 5 stages in a row
  const stages = [
    { title: 'Ulazna\nfotografija', sub: 'JPG / PNG\nlijeban format', bg: '1E3A5F', accent: null },
    { title: 'YOLO11s\nSpecies Detector', sub: 'yolo11s.pt\n640×640 px', bg: '0F4C3A', accent: TEAL },
    { title: 'Crop\n+ Margina', sub: '10-15%\nkontekst', bg: '1E3A5F', accent: null },
    { title: 'EfficientNetV2-S\nBreed Classifier', sub: 'tf_efficientnetv2_s\n384×384 px', bg: '0F4C3A', accent: TEAL2 },
    { title: 'Rezultat', sub: 'species + breed\n+ confidence', bg: '3B1A5F', accent: AMBER },
  ];

  const bw = 1.58, bh = 1.65, startX = 0.20, by = 1.40;
  stages.forEach((st, i) => {
    const bx = startX + i * (bw + 0.25);
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: by, w: bw, h: bh, fill: { color: st.bg }, line: { color: st.accent || NAVYM, width: st.accent ? 1.5 : 0.5 } });
    if (st.accent) s.addShape(pres.shapes.RECTANGLE, { x: bx, y: by, w: bw, h: 0.05, fill: { color: st.accent }, line: { color: st.accent, width: 0 } });
    s.addText(st.title, { x: bx + 0.10, y: by + 0.14, w: bw - 0.20, h: 0.70, fontFace: FF, fontSize: 12.5, bold: true, color: WHITE, align: 'center', margin: 0 });
    s.addText(st.sub, { x: bx + 0.10, y: by + 0.85, w: bw - 0.20, h: 0.65, fontFace: FF, fontSize: 9.5, color: TEAL2, align: 'center', margin: 0 });
    // arrow after (not last)
    if (i < stages.length - 1) {
      const ax = bx + bw + 0.04;
      s.addText('→', { x: ax, y: by + 0.65, w: 0.22, h: 0.35, fontFace: FF, fontSize: 18, bold: true, color: AMBER, align: 'center', margin: 0 });
    }
  });

  // Below — key config boxes
  const cfgCards = [
    { title: 'YOLO — detektor vrsta', lines: ['100 epoha, batch 16', 'cos_lr, close_mosaic=10', 'Hybrid dataset (40/40/20)'] },
    { title: 'Breed — klasifikator pasmina', lines: ['60 epoha, AdamW, lr=0.0003', 'label_smoothing=0.1, MixUp/CutMix', 'Modularan — 1 model po vrsti'] },
    { title: 'Kontrolni eksperiment', lines: ['Direktni YOLO breed model', 'Sve pasmine odjednom (100+ klasa)', 'Best mAP50=0.228 (25 ep), Recall≤0.22 — daleko ispod dvostupanjskog ✗'] },
  ];
  cfgCards.forEach((c, i) => {
    const cx = 0.30 + i * 3.27, cy = 3.28;
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: cy, w: 3.10, h: 1.90, fill: { color: NAVYM }, line: { color: '334155', width: 0.5 } });
    s.addText(c.title, { x: cx + 0.14, y: cy + 0.14, w: 2.82, h: 0.30, fontFace: FF, fontSize: 11, bold: true, color: i === 2 ? RED2 : TEAL2, margin: 0 });
    c.lines.forEach((l, j) => {
      s.addText(l, { x: cx + 0.14, y: cy + 0.50 + j * 0.38, w: 2.82, h: 0.32,
        fontFace: FF, fontSize: 10, color: i === 2 && j === 2 ? RED2 : '94A3B8', margin: 0 });
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════
// SLIDE 6 — SPECIES DETEKTOR REZULTATI
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  sectionPill(s, '4. Evaluacija — Species');
  slideTitle(s, 'Species detektor — iterativno poboljšanje');

  // Stat boxes
  statBox(s, 0.35, 1.25, 2.20, 1.00, '0.781', 'mAP@50 — finalni run (Jul 2026)', TEAL);
  statBox(s, 2.68, 1.25, 2.20, 1.00, '0.788 / 0.755', 'Precision / Recall', AMBER);
  statBox(s, 4.98, 1.25, 2.20, 1.00, '0.477', 'mAP@50–95 (stroga)', NAVYM);
  statBox(s, 7.23, 1.25, 2.42, 1.00, '0.346 → 0.781', 'rani → finalni run (+126%)', GREEN);

  // Chart left
  s.addImage({ path: imgPath('prezentacija_assets\\metrics_compare.png'), x: 0.35, y: 2.42, w: 5.40, h: 2.90 });

  // Iterations table right
  card(s, 5.95, 2.42, 3.70, 2.90, TEAL);
  s.addText('Ključne iteracije', { x: 6.12, y: 2.54, w: 3.35, h: 0.28, fontFace: FF, fontSize: 11, bold: true, color: SLATE, margin: 0 });

  const rows = [
    { label: 'full17_v2 (rana)',   val: '0.346', col: RED2 },
    { label: 'full18_goose_10more', val: '0.617', col: AMBER },
    { label: 'full18_goose_box25', val: '0.666', col: AMBER },
    { label: 'phr_5more_box25',    val: '0.679', col: AMBER },
    { label: 'fresh_qa_20260706 ✓', val: '0.781', col: GREEN },
  ];
  rows.forEach((r, i) => {
    const ry = 2.90 + i * 0.56;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.95, y: ry, w: 3.70, h: 0.52,
      fill: { color: i % 2 === 0 ? 'F8FAFC' : WHITE }, line: { color: BORD, width: 0 } });
    s.addText(r.label, { x: 6.07, y: ry + 0.10, w: 2.50, h: 0.30, fontFace: FF, fontSize: 10, color: SLATE, margin: 0 });
    s.addText(r.val, { x: 8.97, y: ry + 0.08, w: 0.65, h: 0.34, fontFace: FF, fontSize: 13, bold: true, color: r.col, align: 'right', margin: 0 });
  });
}

// ═══════════════════════════════════════════════════════════════════════
// SLIDE 7 — BREED KLASIFIKATORI
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  sectionPill(s, '4. Evaluacija — Breed');
  slideTitle(s, 'Breed klasifikatori — 16 vrsta, svi završeni');

  // Stat row
  statBox(s, 0.35, 1.25, 2.10, 0.90, '16', 'istrenirano breed modela', TEAL);
  statBox(s, 2.58, 1.25, 2.10, 0.90, '98.4%', 'best: donkey top-1', GREEN);
  statBox(s, 4.78, 1.25, 2.10, 0.90, '50.3%', 'worst: sheep top-1', RED2);
  statBox(s, 6.98, 1.25, 2.67, 0.90, '11 / 16 ≥ 78%', 'modela s dobrim performansama', AMBER);

  // Main chart
  s.addImage({ path: imgPath('chart_breed_accuracy.png'), x: 0.35, y: 2.28, w: 9.30, h: 3.10 });
}

// ═══════════════════════════════════════════════════════════════════════
// SLIDE 8 — ANALIZA KONVERGENCIJE / OVERFITTING
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  sectionPill(s, '4. Evaluacija — Analiza');
  slideTitle(s, 'Konvergencija i overfitting — zašto sheep i dog zaostaju');

  // Main chart
  s.addImage({ path: imgPath('chart_training_curves.png'), x: 0.35, y: 1.25, w: 7.00, h: 2.85 });

  // Diagnosis cards right
  card(s, 7.55, 1.25, 2.10, 1.32, RED2);
  s.addText('Dog — 48.6%', { x: 7.72, y: 1.35, w: 1.80, h: 0.28, fontFace: FF, fontSize: 11, bold: true, color: SLATE, margin: 0 });
  s.addText('train: 98% → val: 49%\nvelik broj sličnih\npasmina, premalo\nraznolikih primjera', { x: 7.72, y: 1.64, w: 1.80, h: 0.88, fontFace: FF, fontSize: 9.5, color: MUTED, margin: 0 });

  card(s, 7.55, 2.70, 2.10, 1.30, RED2);
  s.addText('Sheep — 35.3%', { x: 7.72, y: 2.80, w: 1.80, h: 0.28, fontFace: FF, fontSize: 11, bold: true, color: SLATE, margin: 0 });
  s.addText('val_loss=3.78\nNajgori overfit.\nPasmine se razlikuju\ngotovo samo bojom', { x: 7.72, y: 3.10, w: 1.80, h: 0.84, fontFace: FF, fontSize: 9.5, color: MUTED, margin: 0 });

  // Bottom — solution
  card(s, 0.35, 4.20, 9.30, 1.10, TEAL);
  s.addText('Rješenje: Mining pipeline', { x: 0.55, y: 4.30, w: 3.0, h: 0.28, fontFace: FF, fontSize: 12, bold: true, color: SLATE, margin: 0 });
  s.addText('324 633 kandidata rudareno — neanotirane životinje iz dataseta koji nemaju breed oznake. Integracija ovih kandidata direktno adresira nedostatak raznolikih primjera za problematične pasmine.', {
    x: 0.55, y: 4.60, w: 9.00, h: 0.58, fontFace: FF, fontSize: 10.5, color: MUTED, margin: 0
  });
}

// ═══════════════════════════════════════════════════════════════════════
// SLIDE 9 — ALATI I INFRASTRUKTURA
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  sectionPill(s, '5. Alati i infrastruktura');
  slideTitle(s, 'Projekt koji se može koristiti — ne samo trenirati');

  // Mining section
  card(s, 0.35, 1.28, 4.55, 2.20, TEAL);
  s.addText('Rudarenje neanotiranih podataka', { x: 0.53, y: 1.38, w: 4.22, h: 0.30, fontFace: FF, fontSize: 11.5, bold: true, color: SLATE, margin: 0 });
  s.addImage({ path: imgPath('mining_samples_panel.jpg'), x: 0.53, y: 1.72, w: 4.22, h: 1.46 });
  s.addText('Detektor pronalazi životinje koje su propuštene ili pogrešno označene u originalnim datasetima.', {
    x: 0.53, y: 3.24, w: 4.22, h: 0.40, fontFace: FF, fontSize: 9.5, italic: true, color: MUTED, margin: 0
  });

  // Box review
  card(s, 5.10, 1.28, 4.55, 2.20, AMBER);
  s.addText('Box review & Label audit app', { x: 5.28, y: 1.38, w: 4.22, h: 0.30, fontFace: FF, fontSize: 11.5, bold: true, color: SLATE, margin: 0 });
  s.addImage({ path: imgPath('selenium-box-review-4.png'), x: 5.28, y: 1.72, w: 4.22, h: 1.46 });
  s.addText('Interaktivni alati za provjeru i reviziju kandidata iz mininga bez mijenjanja originalnih oznaka.', {
    x: 5.28, y: 3.24, w: 4.22, h: 0.40, fontFace: FF, fontSize: 9.5, italic: true, color: MUTED, margin: 0
  });

  // Bottom — tools grid
  const tools = [
    { label: '17 CLI naredbi', sub: 'cijeli pipeline pokriven\nanimal-build-manifest, train...', col: TEAL },
    { label: 'Web aplikacija', sub: 'upload slika → inferencija\nspecies + breed u browseru', col: AMBER },
    { label: 'AMD ROCm GPU', sub: 'izolirani ROCm 7.2.1 runtime\nbez mijenjanja globalnog PATH', col: '7C3AED' },
    { label: 'GitHub CI + pytest', sub: '8 test datoteka, automatski\nci.yml + weekly_scout.yml', col: GREEN },
  ];
  tools.forEach((t, i) => {
    const tx = 0.35 + i * 2.40;
    s.addShape(pres.shapes.RECTANGLE, { x: tx, y: 3.70, w: 2.28, h: 1.45, fill: { color: NAVYM }, line: { color: NAVYM, width: 0 } });
    s.addShape(pres.shapes.RECTANGLE, { x: tx, y: 3.70, w: 2.28, h: 0.05, fill: { color: t.col }, line: { color: t.col, width: 0 } });
    s.addText(t.label, { x: tx + 0.12, y: 3.80, w: 2.04, h: 0.30, fontFace: FF, fontSize: 11, bold: true, color: WHITE, margin: 0 });
    s.addText(t.sub, { x: tx + 0.12, y: 4.12, w: 2.04, h: 0.90, fontFace: FF, fontSize: 9.5, color: '94A3B8', margin: 0 });
  });
}

// ═══════════════════════════════════════════════════════════════════════
// SLIDE 10 — ZAKLJUČAK (dark)
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.07, fill: { color: TEAL }, line: { color: TEAL, width: 0 } });

  s.addText('Zaključak', { x: 0.50, y: 0.24, w: 4.0, h: 0.36, fontFace: FF, fontSize: 13, color: TEAL2, margin: 0 });
  s.addText('Pet ključnih nalaza', { x: 0.50, y: 0.65, w: 9.0, h: 0.52, fontFace: FF, fontSize: 26, bold: true, color: WHITE, margin: 0 });

  const points = [
    { n: '01', title: 'Hijerarhija je nužna, ne izborna', body: 'Direktni breed YOLO: best mAP=0.228, Recall≤0.22 nakon 25 epoha — daleko ispod dvostupanjskog koji funkcionira za svih 16 vrsta.' , col: TEAL },
    { n: '02', title: 'Podaci > arhitektura', body: 'Isti model s poboljšanim podacima: mAP 0.346 → 0.781 (+126%). Normalizacija, balansiranje i sintetički kompoziti odlučujući su.', col: AMBER },
    { n: '03', title: '11 od 16 breed modela ≥ 78% top-1', body: 'Donkey 98.4%, pig 98.0%, peafowl 95.3%. Moduli s dovoljno raznolikih podataka konvergiraju stabilno.', col: GREEN },
    { n: '04', title: 'Dog/sheep: overfitting, ne arhitektura', body: 'Train 99% vs val 51%/38%. Mining pipeline (324k kandidata) direktno adresira nedostatak raznolikosti.', col: RED2 },
    { n: '05', title: 'Sustav je operativan, ne samo treniran', body: 'CLI, web app, mining alati, review alati, CI — projekt se može koristiti i demonstrirati.', col: '7C3AED' },
  ];

  points.forEach((p, i) => {
    const col = i < 3 ? 0 : 1;
    const row = i < 3 ? i : i - 3;
    const px = col === 0 ? 0.42 : 5.25;
    const py = 1.34 + row * 1.26;
    const pw = 4.60;

    s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: pw, h: 1.10, fill: { color: NAVYM }, line: { color: '334155', width: 0.5 } });
    s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: 0.06, h: 1.10, fill: { color: p.col }, line: { color: p.col, width: 0 } });
    s.addText(p.n, { x: px + 0.16, y: py + 0.08, w: 0.38, h: 0.30, fontFace: FF, fontSize: 10, bold: true, color: p.col, margin: 0 });
    s.addText(p.title, { x: px + 0.55, y: py + 0.08, w: pw - 0.65, h: 0.30, fontFace: FF, fontSize: 11, bold: true, color: WHITE, margin: 0 });
    s.addText(p.body, { x: px + 0.16, y: py + 0.42, w: pw - 0.26, h: 0.60, fontFace: FF, fontSize: 9.5, color: '94A3B8', margin: 0 });
  });

  // Bottom right — submission list
  s.addShape(pres.shapes.RECTANGLE, { x: 5.25, y: 4.70, w: 4.60, h: 0.70, fill: { color: NAVYM }, line: { color: TEAL, width: 0.8 } });
  s.addText('📦  Predaja: seminar (PDF)  ·  izvorni kod  ·  ova prezentacija', {
    x: 5.42, y: 4.82, w: 4.28, h: 0.36, fontFace: FF, fontSize: 9.5, color: TEAL2, margin: 0
  });
}

// ── Save ────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: 'prezentacija_projekt_v3.pptx' })
  .then(() => console.log('Done: prezentacija_projekt_v3.pptx'))
  .catch(e => { console.error(e); process.exit(1); });
