const pptxgen = require("pptxgenjs");
const path = require("path");

const BASE = "C:\\Users\\User\\Desktop\\Strojno ucenje\\";
const A = (f) => path.join(BASE, f);

// ── design tokens ─────────────────────────────────────────────────────────
const INK   = "0B2027";   // deep ink (dark bg)
const INK2  = "103641";   // panel on dark
const PAPER = "F5F7F8";   // light bg
const CARD  = "FFFFFF";
const LINE  = "E2E8EA";   // hairline
const TEAL  = "13A89E";   // primary accent
const TEALD = "0E7C74";
const AMBER = "E0A33E";   // secondary accent
const TEXT  = "17313A";   // body text on light
const MUTE  = "6A7B83";   // muted text
const GREEN = "4F9D77";
const RED   = "C45D54";
const MIDA  = "C98A3C";
const WHITE = "FFFFFF";
const FOG   = "9DB2B8";   // muted on dark

const HF = "Segoe UI Semibold";
const BF = "Segoe UI";

const W = 13.33, H = 7.5;
const MX = 0.62;                 // side margin
const CW = W - 2 * MX;           // content width

const pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: W, height: H });
pres.layout = "WIDE";
pres.author = "Patrik Hrvoje Radočaj";
pres.title = "Višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina";

const shadow = () => ({ type: "outer", color: "0B2027", blur: 9, offset: 3, angle: 90, opacity: 0.10 });

// ── shared chrome ─────────────────────────────────────────────────────────
function lightBase(slide) {
  slide.background = { color: PAPER };
}
function darkBase(slide) {
  slide.background = { color: INK };
}
function header(slide, kicker, title, n) {
  // accent square + kicker
  slide.addShape(pres.shapes.RECTANGLE, { x: MX, y: 0.46, w: 0.16, h: 0.16, fill: { color: TEAL }, line: { type: "none" } });
  slide.addText(kicker.toUpperCase(), {
    x: MX + 0.28, y: 0.40, w: 9, h: 0.3, fontFace: HF, fontSize: 11, bold: true,
    color: TEALD, charSpacing: 2, align: "left", valign: "middle", margin: 0,
  });
  slide.addText(title, {
    x: MX, y: 0.74, w: CW - 1.0, h: 0.62, fontFace: HF, fontSize: 25, bold: true,
    color: TEXT, align: "left", valign: "middle", margin: 0,
  });
  // page number top-right
  slide.addText(String(n).padStart(2, "0"), {
    x: W - MX - 1.0, y: 0.40, w: 1.0, h: 0.3, fontFace: BF, fontSize: 11,
    color: MUTE, align: "right", valign: "middle", margin: 0,
  });
}
function footer(slide) {
  slide.addShape(pres.shapes.LINE, { x: MX, y: H - 0.46, w: CW, h: 0, line: { color: LINE, width: 1 } });
  slide.addText("Višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina", {
    x: MX, y: H - 0.42, w: 9, h: 0.28, fontFace: BF, fontSize: 8.5, color: MUTE, align: "left", valign: "middle", margin: 0,
  });
  slide.addText("FERIT · Strojno učenje · 2026.", {
    x: W - MX - 4, y: H - 0.42, w: 4, h: 0.28, fontFace: BF, fontSize: 8.5, color: MUTE, align: "right", valign: "middle", margin: 0,
  });
}
function card(slide, x, y, w, h, fill = CARD) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: LINE, width: 1 }, shadow: shadow() });
}
// accent-rail card
function railCard(slide, x, y, w, h, accent) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: shadow() });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.07, h, fill: { color: accent }, line: { type: "none" } });
}
function imgFit(slide, file, ratio, x, y, w, h, frame = true) {
  // contain inside box, centered
  let iw = w, ih = w / ratio;
  if (ih > h) { ih = h; iw = h * ratio; }
  const ix = x + (w - iw) / 2, iy = y + (h - ih) / 2;
  if (frame) slide.addShape(pres.shapes.RECTANGLE, { x: ix - 0.04, y: iy - 0.04, w: iw + 0.08, h: ih + 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: shadow() });
  slide.addImage({ path: A(file), x: ix, y: iy, w: iw, h: ih });
  return { ix, iy, iw, ih };
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 1 — TITLE
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); darkBase(s);
  // motif: large faint ring + teal accent block
  s.addShape(pres.shapes.OVAL, { x: 9.6, y: -1.6, w: 5.6, h: 5.6, fill: { type: "none" }, line: { color: INK2, width: 28 } });
  s.addShape(pres.shapes.OVAL, { x: 10.8, y: 3.4, w: 3.4, h: 3.4, fill: { type: "none" }, line: { color: TEALD, width: 2 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.18, h: H, fill: { color: TEAL }, line: { type: "none" } });

  s.addText("PROJEKT IZ STROJNOG UČENJA I RAČUNALNOG VIDA", {
    x: 1.0, y: 1.5, w: 10, h: 0.35, fontFace: HF, fontSize: 12.5, bold: true, color: TEAL, charSpacing: 3, margin: 0,
  });
  s.addText("Višerazinski sustav za prepoznavanje\nživotinjskih vrsta i pasmina", {
    x: 0.96, y: 2.0, w: 11.4, h: 1.9, fontFace: HF, fontSize: 41, bold: true, color: WHITE, lineSpacingMultiple: 1.02, margin: 0,
  });
  s.addText("pomoću dubokog učenja", {
    x: 1.0, y: 3.85, w: 11, h: 0.6, fontFace: BF, fontSize: 23, italic: true, color: FOG, margin: 0,
  });

  // bottom meta row
  s.addShape(pres.shapes.LINE, { x: 1.0, y: 5.55, w: 8.4, h: 0, line: { color: INK2, width: 1.5 } });
  s.addText([
    { text: "Autor   ", options: { color: TEAL, bold: true, fontFace: HF } },
    { text: "Patrik Hrvoje Radočaj", options: { color: WHITE, fontFace: BF } },
  ], { x: 1.0, y: 5.75, w: 6, h: 0.34, fontSize: 13, margin: 0, valign: "middle" });
  s.addText([
    { text: "Tehnologije   ", options: { color: TEAL, bold: true, fontFace: HF } },
    { text: "YOLO11 · EfficientNetV2-S · PyTorch · Hydra", options: { color: FOG, fontFace: BF } },
  ], { x: 1.0, y: 6.15, w: 9, h: 0.34, fontSize: 12, margin: 0, valign: "middle" });
  s.addText("Osijek, 2026.", { x: 1.0, y: 6.55, w: 6, h: 0.34, fontFace: BF, fontSize: 12, color: MUTE, margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 2 — SADRŽAJ
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); lightBase(s);
  header(s, "Pregled", "Sadržaj izlaganja", 2);
  const items = [
    ["01", "Problem i motivacija", "Zašto fina klasifikacija i hijerarhijski pristup"],
    ["02", "Arhitektura sustava", "Dvostupanjski pipeline i kontrolni eksperiment"],
    ["03", "Podaci i predobrada", "Neuravnoteženost, sintetički kompoziti, rudarenje"],
    ["04", "Modeli i metode", "YOLO11s detektor i EfficientNetV2-S klasifikator"],
    ["05", "Rezultati", "Species detektor i 15 breed klasifikatora"],
    ["06", "Analiza i zaključci", "Overfitting, kontrolni eksperiment, smjerovi"],
  ];
  const colW = (CW - 0.5) / 2, gap = 0.5;
  const rowH = 1.18, y0 = 1.75;
  items.forEach((it, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = MX + col * (colW + gap), y = y0 + row * (rowH + 0.18);
    railCard(s, x, y, colW, rowH, TEAL);
    s.addText(it[0], { x: x + 0.22, y: y + 0.12, w: 1.0, h: 0.9, fontFace: HF, fontSize: 34, bold: true, color: "DCE6E6", align: "left", valign: "middle", margin: 0 });
    s.addText(it[1], { x: x + 1.35, y: y + 0.20, w: colW - 1.55, h: 0.42, fontFace: HF, fontSize: 15.5, bold: true, color: TEXT, align: "left", valign: "middle", margin: 0 });
    s.addText(it[2], { x: x + 1.35, y: y + 0.62, w: colW - 1.55, h: 0.42, fontFace: BF, fontSize: 11, color: MUTE, align: "left", valign: "middle", margin: 0 });
  });
  footer(s);
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 3 — PROBLEM I MOTIVACIJA
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); lightBase(s);
  header(s, "01 · Uvod", "Problem i motivacija", 3);
  // left: narrative
  const lx = MX, lw = 6.7;
  s.addText("Cilj nije samo otkriti da je na slici životinja — nego razlikovati vrstu, a zatim i pasminu unutar te vrste.", {
    x: lx, y: 1.7, w: lw, h: 0.95, fontFace: HF, fontSize: 17.5, bold: true, color: TEXT, lineSpacingMultiple: 1.12, margin: 0, valign: "top",
  });
  const pts = [
    ["Fina granularnost", "Pasmine se razlikuju tek u boji dlake i sitnim detaljima — labrador i zlatni retriver dijele gotovo identičnu siluetu."],
    ["Realni uvjeti", "Promjenjiva kvaliteta, mali objekti, zaklanjanja i raznolike pozadine otežavaju jednostavnu klasifikaciju."],
    ["Heterogeni podaci", "Izvori s nekonzistentnim oznakama i izraženom neuravnoteženošću klasa (omjer ~24:1 nakon balansiranja)."],
  ];
  let yy = 2.85;
  pts.forEach((p) => {
    s.addShape(pres.shapes.RECTANGLE, { x: lx, y: yy + 0.06, w: 0.13, h: 0.13, fill: { color: AMBER }, line: { type: "none" } });
    s.addText(p[0], { x: lx + 0.3, y: yy - 0.02, w: lw - 0.3, h: 0.3, fontFace: HF, fontSize: 13.5, bold: true, color: TEXT, margin: 0 });
    s.addText(p[1], { x: lx + 0.3, y: yy + 0.28, w: lw - 0.3, h: 0.62, fontFace: BF, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.08, margin: 0 });
    yy += 1.12;
  });

  // right: insight card with the key decision
  const rx = MX + lw + 0.5, rw = CW - lw - 0.5;
  railCard(s, rx, 1.7, rw, 4.55, TEAL);
  s.addText("RJEŠENJE", { x: rx + 0.35, y: 1.95, w: rw - 0.7, h: 0.3, fontFace: HF, fontSize: 11, bold: true, color: TEALD, charSpacing: 2, margin: 0 });
  s.addText("Hijerarhijska dekompozicija", { x: rx + 0.35, y: 2.28, w: rw - 0.7, h: 0.5, fontFace: HF, fontSize: 19, bold: true, color: TEXT, margin: 0 });
  s.addText("Problem se dijeli na dvije upravljive faze umjesto jednog monolitnog modela: prvo detekcija vrste i lokalizacija, zatim fina klasifikacija pasmine na izoliranom isječku.", {
    x: rx + 0.35, y: 2.85, w: rw - 0.7, h: 1.2, fontFace: BF, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.15, margin: 0,
  });
  s.addShape(pres.shapes.LINE, { x: rx + 0.35, y: 4.2, w: rw - 0.7, h: 0, line: { color: LINE, width: 1 } });
  s.addText([
    { text: "Potvrđeno eksperimentom:  ", options: { bold: true, color: TEXT, fontFace: HF } },
    { text: "jednofazni breed YOLO doseže tek ", options: { color: MUTE, fontFace: BF } },
    { text: "mAP@50 ≈ 0,21", options: { bold: true, color: RED, fontFace: HF } },
    { text: " — daleko ispod dvostupanjskog pristupa (0,743).", options: { color: MUTE, fontFace: BF } },
  ], { x: rx + 0.35, y: 4.4, w: rw - 0.7, h: 1.6, fontSize: 12.5, lineSpacingMultiple: 1.2, valign: "top", margin: 0 });
  footer(s);
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 4 — ARHITEKTURA PIPELINE
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); lightBase(s);
  header(s, "02 · Arhitektura", "Dvostupanjski pipeline", 4);

  const boxY = 2.25, boxH = 1.5, boxW = 2.35;
  const xs = [MX + 0.1, 0, 0, 0];
  const steps = [
    { t: "Ulazna slika", s: "fotografija\nživotinje", fill: CARD, accent: FOG, txt: TEXT },
    { t: "YOLO11s", s: "detekcija vrste\n+ lokalizacija", fill: CARD, accent: TEAL, txt: TEXT },
    { t: "Crop", s: "izrez objekta\n+ kontekst", fill: CARD, accent: AMBER, txt: TEXT },
    { t: "EfficientNetV2-S", s: "klasifikacija\npasmine", fill: CARD, accent: TEALD, txt: TEXT },
  ];
  const gap = 0.62;
  let x = MX + 0.15;
  const centers = [];
  steps.forEach((st, i) => {
    railCard(s, x, boxY, boxW, boxH, st.accent);
    s.addText(st.t, { x: x + 0.18, y: boxY + 0.24, w: boxW - 0.3, h: 0.4, fontFace: HF, fontSize: 14.5, bold: true, color: st.txt, align: "center", margin: 0 });
    s.addText(st.s, { x: x + 0.18, y: boxY + 0.68, w: boxW - 0.3, h: 0.6, fontFace: BF, fontSize: 10.5, color: MUTE, align: "center", lineSpacingMultiple: 1.05, margin: 0 });
    centers.push(x + boxW / 2);
    if (i < steps.length - 1) {
      const ax = x + boxW + 0.10;
      s.addText("→", { x: ax, y: boxY, w: gap - 0.2, h: boxH, fontFace: HF, fontSize: 26, bold: true, color: TEAL, align: "center", valign: "middle", margin: 0 });
    }
    x += boxW + gap;
  });
  // stage labels above
  s.addText("PRVA FAZA — DETEKCIJA VRSTE", { x: MX + 0.15, y: 1.78, w: boxW * 2 + gap, h: 0.3, fontFace: HF, fontSize: 10.5, bold: true, color: TEALD, charSpacing: 1.5, align: "center", margin: 0 });
  s.addText("DRUGA FAZA — KLASIFIKACIJA PASMINE", { x: MX + 0.15 + (boxW + gap) * 2, y: 1.78, w: boxW * 2 + gap, h: 0.3, fontFace: HF, fontSize: 10.5, bold: true, color: TEALD, charSpacing: 1.5, align: "center", margin: 0 });

  // control experiment row
  const cy = 4.55;
  s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.15, y: cy, w: CW - 0.3, h: 1.25, fill: { color: "FBF1E0" }, line: { color: "EAD6AE", width: 1 } });
  s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.15, y: cy, w: 0.07, h: 1.25, fill: { color: AMBER }, line: { type: "none" } });
  s.addText("KONTROLNI EKSPERIMENT", { x: MX + 0.45, y: cy + 0.18, w: 5, h: 0.3, fontFace: HF, fontSize: 10.5, bold: true, color: MIDA, charSpacing: 1.5, margin: 0 });
  s.addText([
    { text: "Direktni YOLO breed model ", options: { bold: true, color: TEXT, fontFace: HF } },
    { text: "— prepoznaje pasminu odjednom, bez faze detekcije vrste. Doseže tek ~0,21 i nakon ~16. epohe regresira (koristi se best.pt, ne last.pt).", options: { color: "7A6435", fontFace: BF } },
  ], { x: MX + 0.45, y: cy + 0.5, w: CW - 3.4, h: 0.65, fontSize: 12, lineSpacingMultiple: 1.1, valign: "top", margin: 0 });
  s.addText("mAP@50 ≈ 0,21", { x: W - MX - 2.7, y: cy + 0.26, w: 2.4, h: 0.5, fontFace: HF, fontSize: 21, bold: true, color: RED, align: "right", valign: "middle", margin: 0 });
  s.addText("najbolji: epoha 15", { x: W - MX - 2.7, y: cy + 0.74, w: 2.4, h: 0.3, fontFace: BF, fontSize: 9.5, color: MIDA, align: "right", valign: "middle", margin: 0 });
  footer(s);
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 5 — PODACI
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); lightBase(s);
  header(s, "03 · Podaci", "Skup podataka i neuravnoteženost", 5);
  // stat row
  const stats = [
    ["19", "vrsta u skupu detektora", TEAL],
    ["26 052", "slika — stage 1 (detektor)", TEAL],
    ["45 223", "mined slika — stage 2", AMBER],
  ];
  const sw = (6.4 - 0.4) / 3, sgap = 0.2, sx0 = MX, sy = 1.78, sh = 1.15;
  stats.forEach((st, i) => {
    const x = sx0 + i * (sw + sgap);
    railCard(s, x, sy, sw, sh, st[2]);
    s.addText(st[0], { x: x + 0.18, y: sy + 0.12, w: sw - 0.3, h: 0.55, fontFace: HF, fontSize: 25, bold: true, color: TEXT, margin: 0, valign: "middle" });
    s.addText(st[1], { x: x + 0.18, y: sy + 0.68, w: sw - 0.3, h: 0.4, fontFace: BF, fontSize: 9.5, color: MUTE, margin: 0, lineSpacingMultiple: 0.95 });
  });
  // narrative under stats
  s.addText("Najzastupljenije su mačka (4 542) i svinja (4 443); najrjeđi su pas (193), llama (347) i guska (395). Ciljanim balansiranjem i rudarenjem nijedna klasa više nije prazna.", {
    x: MX, y: 3.15, w: 6.2, h: 2.0, fontFace: BF, fontSize: 13, color: TEXT, lineSpacingMultiple: 1.3, valign: "top", margin: 0,
  });
  s.addText([
    { text: "Omjer max/min:  ", options: { bold: true, color: TEXT, fontFace: HF } },
    { text: "pao je na ~24:1 (mačka 4 542 / pas 193) — i dalje izražen, pa je balansiranje jednako važno kao i sam model.", options: { color: MUTE, fontFace: BF } },
  ], { x: MX, y: 4.6, w: 6.2, h: 1.3, fontSize: 12, lineSpacingMultiple: 1.2, valign: "top", margin: 0 });

  // right: species balance chart
  imgFit(s, "chart_species_balance_v2.png", 1.232, MX + 6.55, 1.72, CW - 6.55, 4.2, true);
  s.addText("Slika 1 · Raspodjela primjera po vrstama u trening skupu", {
    x: MX + 6.55, y: 6.0, w: CW - 6.55, h: 0.3, fontFace: BF, fontSize: 9, italic: true, color: MUTE, align: "center", margin: 0,
  });
  footer(s);
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 6 — SINTETIČKI KOMPOZITI
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); lightBase(s);
  header(s, "03 · Predobrada", "Sintetički kompoziti za balansiranje", 6);
  // left text
  const lw = 4.5;
  s.addText("Rijetke klase se umjetno pojačavaju kompozitnim generiranjem.", {
    x: MX, y: 1.78, w: lw, h: 0.8, fontFace: HF, fontSize: 16, bold: true, color: TEXT, lineSpacingMultiple: 1.1, margin: 0,
  });
  const steps = [
    "Iz trening slika izrezuju se pojedinačne životinje pomoću postojećih YOLO oznaka.",
    "Više izrezaka različitih vrsta postavlja se na novu pozadinsku teksturu u gridu s jitterom (±6 %) i skaliranjem (0,88–1,20×).",
    "Bounding boxovi se generiraju automatski za svaki postavljeni izrezak.",
  ];
  let yy = 2.7;
  steps.forEach((t, i) => {
    s.addShape(pres.shapes.OVAL, { x: MX, y: yy, w: 0.34, h: 0.34, fill: { color: TEAL }, line: { type: "none" } });
    s.addText(String(i + 1), { x: MX, y: yy, w: 0.34, h: 0.34, fontFace: HF, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
    s.addText(t, { x: MX + 0.5, y: yy - 0.05, w: lw - 0.5, h: 0.75, fontFace: BF, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.1, valign: "top", margin: 0 });
    yy += 0.92;
  });
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 5.6, w: lw, h: 0.62, fill: { color: "E7F4F2" }, line: { type: "none" } });
  s.addText([
    { text: "Omjer uzorkovanja:  ", options: { bold: true, color: TEALD, fontFace: HF } },
    { text: "40 % sintetika · 40 % real · 20 % augmentirano", options: { color: TEXT, fontFace: BF } },
  ], { x: MX + 0.2, y: 5.6, w: lw - 0.4, h: 0.62, fontSize: 11, valign: "middle", margin: 0 });

  // right: composite + source crops
  const rx = MX + lw + 0.55, rw = CW - lw - 0.55;
  // composite (square) on left of right area, source strip on right
  const compW = 3.0;
  imgFit(s, "synthetic_composite_example.jpg", 1.0, rx, 1.85, compW, 3.55, true);
  s.addText("Slika 3 · sintetički kompozit", { x: rx, y: 5.5, w: compW, h: 0.3, fontFace: BF, fontSize: 9, italic: true, color: MUTE, align: "center", margin: 0 });
  const srcX = rx + compW + 0.35, srcW = rw - compW - 0.35;
  imgFit(s, "synthetic_source_crops.jpg", 2.0, srcX, 2.55, srcW, 2.2, true);
  s.addText("Slika 2 · izvorni izresci (duck, horse, pig, sheep)", { x: srcX, y: 4.85, w: srcW, h: 0.3, fontFace: BF, fontSize: 9, italic: true, color: MUTE, align: "center", margin: 0 });
  s.addText("Komponente →", { x: srcX, y: 2.2, w: srcW, h: 0.3, fontFace: HF, fontSize: 10, bold: true, color: TEALD, align: "center", margin: 0 });
  footer(s);
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 7 — RUDARENJE I ALATI
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); lightBase(s);
  header(s, "03 · Infrastruktura", "Rudarenje podataka i alati za reviziju", 7);
  // stat callouts top
  const stat1X = MX, statW = 2.7;
  railCard(s, stat1X, 1.78, statW, 1.1, TEAL);
  s.addText("324 633", { x: stat1X + 0.2, y: 1.9, w: statW - 0.4, h: 0.5, fontFace: HF, fontSize: 23, bold: true, color: TEXT, margin: 0, valign: "middle" });
  s.addText("ukupno kandidata iz rudarenja", { x: stat1X + 0.2, y: 2.42, w: statW - 0.4, h: 0.35, fontFace: BF, fontSize: 9.5, color: MUTE, margin: 0 });
  railCard(s, stat1X + statW + 0.25, 1.78, statW, 1.1, AMBER);
  s.addText("20 000", { x: stat1X + statW + 0.45, y: 1.9, w: statW - 0.4, h: 0.5, fontFace: HF, fontSize: 23, bold: true, color: TEXT, margin: 0, valign: "middle" });
  s.addText("visokoprioritnih za pregled", { x: stat1X + statW + 0.45, y: 2.42, w: statW - 0.4, h: 0.35, fontFace: BF, fontSize: 9.5, color: MUTE, margin: 0 });

  s.addText("Istrenirani detektor pretražuje slike za propuštene životinje; nepodudarne detekcije postaju kandidati za reviziju — originalne oznake se nikad ne mijenjaju automatski.", {
    x: MX, y: 3.05, w: statW * 2 + 0.25, h: 1.5, fontFace: BF, fontSize: 12, color: TEXT, lineSpacingMultiple: 1.25, valign: "top", margin: 0,
  });
  // mining panel image (wide)
  imgFit(s, "mining_samples_panel.jpg", 3.062, MX, 4.55, statW * 2 + 0.25, 1.5, true);
  s.addText("Slika 4 · primjeri iz rudarenja (cross-species konfuzije)", { x: MX, y: 6.05, w: statW * 2 + 0.25, h: 0.3, fontFace: BF, fontSize: 9, italic: true, color: MUTE, align: "left", margin: 0 });

  // right: two app screenshots
  const rx = MX + statW * 2 + 0.6, rw = CW - (statW * 2 + 0.6);
  const halfW = (rw - 0.3) / 2;
  imgFit(s, "selenium-box-review-4.png", 1.623, rx, 1.78, halfW, 3.1, true);
  s.addText("Box review app", { x: rx, y: 4.95, w: halfW, h: 0.28, fontFace: HF, fontSize: 10, bold: true, color: TEXT, align: "center", margin: 0 });
  s.addText("pregled i korekcija okvira", { x: rx, y: 5.2, w: halfW, h: 0.28, fontFace: BF, fontSize: 8.5, color: MUTE, align: "center", margin: 0 });
  imgFit(s, "label-audit-app.png", 1.857, rx + halfW + 0.3, 1.78, halfW, 3.1, true);
  s.addText("Label audit app", { x: rx + halfW + 0.3, y: 4.95, w: halfW, h: 0.28, fontFace: HF, fontSize: 10, bold: true, color: TEXT, align: "center", margin: 0 });
  s.addText("revizija kvalitete oznaka", { x: rx + halfW + 0.3, y: 5.2, w: halfW, h: 0.28, fontFace: BF, fontSize: 8.5, color: MUTE, align: "center", margin: 0 });
  footer(s);
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 8 — MODELI I METODE
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); lightBase(s);
  header(s, "04 · Metode", "Modeli i ključni hiperparametri", 8);
  const cw = (CW - 0.5) / 2, y0 = 1.78, ch = 4.45;
  // card 1 — YOLO11s
  railCard(s, MX, y0, cw, ch, TEAL);
  s.addText("PRVA FAZA", { x: MX + 0.3, y: y0 + 0.25, w: cw - 0.6, h: 0.28, fontFace: HF, fontSize: 10.5, bold: true, color: TEALD, charSpacing: 2, margin: 0 });
  s.addText("YOLO11s — species detektor", { x: MX + 0.3, y: y0 + 0.55, w: cw - 0.6, h: 0.45, fontFace: HF, fontSize: 18, bold: true, color: TEXT, margin: 0 });
  s.addText("Jednofazni detektor: u jednom prolazu predviđa sve okvire i klase. Brz i pogodan za interaktivnu primjenu.", {
    x: MX + 0.3, y: y0 + 1.05, w: cw - 0.6, h: 0.8, fontFace: BF, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.12, margin: 0,
  });
  const yolo = [["Bazni model", "yolo11s.pt (COCO)"], ["Rezolucija", "1280 px (najnovije)"], ["Trening", "10 epoha · batch 4"], ["Raspored", "cosine · AMP · mosaic"]];
  let ry = y0 + 1.95;
  yolo.forEach(([k, v], i) => {
    if (i) s.addShape(pres.shapes.LINE, { x: MX + 0.3, y: ry, w: cw - 0.6, h: 0, line: { color: LINE, width: 0.75 } });
    s.addText(k, { x: MX + 0.3, y: ry + 0.06, w: (cw - 0.6) * 0.5, h: 0.4, fontFace: BF, fontSize: 11.5, color: MUTE, valign: "middle", margin: 0 });
    s.addText(v, { x: MX + 0.3 + (cw - 0.6) * 0.45, y: ry + 0.06, w: (cw - 0.6) * 0.55, h: 0.4, fontFace: HF, fontSize: 11.5, bold: true, color: TEXT, align: "right", valign: "middle", margin: 0 });
    ry += 0.52;
  });
  // card 2 — EfficientNetV2-S
  const x2 = MX + cw + 0.5;
  railCard(s, x2, y0, cw, ch, TEALD);
  s.addText("DRUGA FAZA", { x: x2 + 0.3, y: y0 + 0.25, w: cw - 0.6, h: 0.28, fontFace: HF, fontSize: 10.5, bold: true, color: TEALD, charSpacing: 2, margin: 0 });
  s.addText("EfficientNetV2-S — pasmine", { x: x2 + 0.3, y: y0 + 0.55, w: cw - 0.6, h: 0.45, fontFace: HF, fontSize: 18, bold: true, color: TEXT, margin: 0 });
  s.addText("Pretreniran na ImageNet-21k → 1k. Modularno: zaseban model po vrsti, ulaz je crop iz prve faze.", {
    x: x2 + 0.3, y: y0 + 1.05, w: cw - 0.6, h: 0.8, fontFace: BF, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.12, margin: 0,
  });
  const eff = [["Backbone", "tf_efficientnetv2_s"], ["Rezolucija", "384 px"], ["Optimizer", "AdamW · lr 3e-4"], ["Regularizacija", "MixUp · CutMix · LS 0,1"]];
  ry = y0 + 1.95;
  eff.forEach(([k, v], i) => {
    if (i) s.addShape(pres.shapes.LINE, { x: x2 + 0.3, y: ry, w: cw - 0.6, h: 0, line: { color: LINE, width: 0.75 } });
    s.addText(k, { x: x2 + 0.3, y: ry + 0.06, w: (cw - 0.6) * 0.5, h: 0.4, fontFace: BF, fontSize: 11.5, color: MUTE, valign: "middle", margin: 0 });
    s.addText(v, { x: x2 + 0.3 + (cw - 0.6) * 0.4, y: ry + 0.06, w: (cw - 0.6) * 0.6, h: 0.4, fontFace: HF, fontSize: 11.5, bold: true, color: TEXT, align: "right", valign: "middle", margin: 0 });
    ry += 0.52;
  });
  footer(s);
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 9 — REZULTATI: SPECIES DETEKTOR
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); lightBase(s);
  header(s, "05 · Rezultati", "Species detektor — evolucija kroz iteracije", 9);
  // left: chart
  imgFit(s, "chart_species_evolution.png", 1.53, MX, 1.85, 6.7, 4.2, true);
  s.addText("Slika 5 · mAP@50 kroz tri iteracije treniranja", { x: MX, y: 6.05, w: 6.7, h: 0.3, fontFace: BF, fontSize: 9, italic: true, color: MUTE, align: "center", margin: 0 });

  // right: final metrics
  const rx = MX + 7.0, rw = CW - 7.0;
  s.addText("Najnoviji „strogi“ model", { x: rx, y: 1.85, w: rw, h: 0.4, fontFace: HF, fontSize: 16, bold: true, color: TEXT, margin: 0 });
  s.addText("1280 px · 10 epoha", { x: rx, y: 2.25, w: rw, h: 0.3, fontFace: BF, fontSize: 11, color: MUTE, margin: 0 });
  const m = [["mAP@50", "0,743", TEAL], ["mAP@50–95", "0,430", TEALD], ["Precision", "0,741", TEXT], ["Recall", "0,698", TEXT]];
  const mw = (rw - 0.3) / 2, mh = 1.15;
  m.forEach(([k, v, c], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = rx + col * (mw + 0.3), y = 2.75 + row * (mh + 0.25);
    card(s, x, y, mw, mh);
    s.addText(v, { x: x + 0.2, y: y + 0.12, w: mw - 0.4, h: 0.6, fontFace: HF, fontSize: 27, bold: true, color: c, margin: 0, valign: "middle" });
    s.addText(k, { x: x + 0.2, y: y + 0.72, w: mw - 0.4, h: 0.3, fontFace: BF, fontSize: 11, color: MUTE, margin: 0 });
  });
  s.addText([
    { text: "Napredak:  ", options: { bold: true, color: TEXT, fontFace: HF } },
    { text: "mAP@50 je porastao s 0,346 na 0,743 — kombinacijom boljih podataka i veće ulazne rezolucije.", options: { color: MUTE, fontFace: BF } },
  ], { x: rx, y: 5.55, w: rw, h: 0.8, fontSize: 11.5, lineSpacingMultiple: 1.2, valign: "top", margin: 0 });
  footer(s);
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 10 — REZULTATI: BREED
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); lightBase(s);
  header(s, "05 · Rezultati", "15 breed klasifikatora po vrstama", 10);
  // chart
  imgFit(s, "chart_breed_v2.png", 1.21, MX, 1.78, 7.0, 4.45, true);

  // right: tier legend + insight
  const rx = MX + 7.3, rw = CW - 7.3;
  s.addText("Tri razine kvalitete", { x: rx, y: 1.9, w: rw, h: 0.4, fontFace: HF, fontSize: 15.5, bold: true, color: TEXT, margin: 0 });
  const tiers = [
    [GREEN, "Dobar — ≥ 80 %", "donkey 98,4 % · pig 96,7 % · peafowl 94,6 %"],
    [MIDA, "Prihvatljiv — 55–80 %", "goose, dromedary, cattle, cat, horse, duck"],
    [RED, "Problematičan — < 55 %", "dog 48,6 % · sheep 35,3 %"],
  ];
  let ty = 2.5;
  tiers.forEach(([c, t, d]) => {
    s.addShape(pres.shapes.RECTANGLE, { x: rx, y: ty + 0.04, w: 0.22, h: 0.22, fill: { color: c }, line: { type: "none" } });
    s.addText(t, { x: rx + 0.36, y: ty - 0.04, w: rw - 0.36, h: 0.32, fontFace: HF, fontSize: 12.5, bold: true, color: TEXT, margin: 0 });
    s.addText(d, { x: rx + 0.36, y: ty + 0.28, w: rw - 0.36, h: 0.55, fontFace: BF, fontSize: 10, color: MUTE, lineSpacingMultiple: 1.05, margin: 0 });
    ty += 1.0;
  });
  s.addShape(pres.shapes.RECTANGLE, { x: rx, y: 5.55, w: rw, h: 0.75, fill: { color: "E7F4F2" }, line: { type: "none" } });
  s.addText([
    { text: "9 / 15 ", options: { bold: true, color: TEALD, fontFace: HF, fontSize: 14 } },
    { text: "modela postiže ≥ 78 % top-1 točnosti.", options: { color: TEXT, fontFace: BF, fontSize: 11.5 } },
  ], { x: rx + 0.2, y: 5.55, w: rw - 0.4, h: 0.75, valign: "middle", lineSpacingMultiple: 1.1, margin: 0 });
  footer(s);
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 11 — ANALIZA: OVERFITTING
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); lightBase(s);
  header(s, "06 · Analiza", "Zašto slabi modeli zakazuju: overfitting", 11);
  imgFit(s, "chart_training_curves.png", 2.727, MX, 1.85, CW, 2.7, true);
  s.addText("Slika 6 · konvergencija (lijevo) i train/val gubitak (desno) za odabrane vrste", {
    x: MX, y: 4.6, w: CW, h: 0.3, fontFace: BF, fontSize: 9, italic: true, color: MUTE, align: "center", margin: 0,
  });
  const cards = [
    ["Dobra generalizacija", GREEN, "Donkey: train i val gubitak padaju zajedno i stabiliziraju se — model uči prenosive značajke."],
    ["Overfitting", RED, "Dog/sheep: train top-1 ≈ 99 % dok val ostaje na 35–49 %. Model pamti primjere umjesto da generalizira."],
    ["Rješenje nije veći model", AMBER, "Uzrok je manjak raznolikih podataka. Odgovor je rudarenje i prikupljanje primjera, ne dublja mreža."],
  ];
  const cwd = (CW - 0.6) / 3, cy = 5.05, chh = 1.4;
  cards.forEach((c, i) => {
    const x = MX + i * (cwd + 0.3);
    railCard(s, x, cy, cwd, chh, c[1]);
    s.addText(c[0], { x: x + 0.22, y: cy + 0.16, w: cwd - 0.44, h: 0.35, fontFace: HF, fontSize: 13, bold: true, color: TEXT, margin: 0 });
    s.addText(c[2], { x: x + 0.22, y: cy + 0.52, w: cwd - 0.44, h: 0.8, fontFace: BF, fontSize: 10.5, color: MUTE, lineSpacingMultiple: 1.08, margin: 0 });
  });
  footer(s);
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 12 — ZAKLJUČAK
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); lightBase(s);
  header(s, "06 · Zaključak", "Ključni zaključci i smjerovi", 12);
  const concl = [
    ["Hijerarhija je nužna", "Jednofazni breed YOLO doseže tek mAP 0,21; staged pristup čisto razdvaja lak (vrsta) i težak (pasmina) problem."],
    ["Podaci ≥ arhitektura", "Samo poboljšanjem podataka i rezolucije mAP@50 raste s 0,346 na 0,743."],
    ["Solidna breed pokrivenost", "9 od 15 modela ≥ 78 % top-1; vrh donkey 98,4 %, pig 96,7 %."],
    ["Overfitting kao usko grlo", "Dog i sheep zakazuju zbog manjka podataka — mining pipeline je direktan odgovor."],
  ];
  const cw = (CW - 0.5) / 2, y0 = 1.8, chh = 1.5;
  concl.forEach((c, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = MX + col * (cw + 0.5), y = y0 + row * (chh + 0.25);
    railCard(s, x, y, cw, chh, TEAL);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.4, h: 0.4, fill: { color: "E7F4F2" }, line: { type: "none" } });
    s.addText(String(i + 1), { x: x + 0.25, y: y + 0.25, w: 0.4, h: 0.4, fontFace: HF, fontSize: 15, bold: true, color: TEALD, align: "center", valign: "middle", margin: 0 });
    s.addText(c[0], { x: x + 0.85, y: y + 0.22, w: cw - 1.1, h: 0.4, fontFace: HF, fontSize: 14.5, bold: true, color: TEXT, valign: "middle", margin: 0 });
    s.addText(c[1], { x: x + 0.85, y: y + 0.66, w: cw - 1.1, h: 0.7, fontFace: BF, fontSize: 11, color: MUTE, lineSpacingMultiple: 1.1, valign: "top", margin: 0 });
  });
  // future directions strip
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 5.45, w: CW, h: 0.85, fill: { color: INK }, line: { type: "none" } });
  s.addText("SMJEROVI", { x: MX + 0.3, y: 5.58, w: 1.6, h: 0.6, fontFace: HF, fontSize: 11, bold: true, color: TEAL, charSpacing: 2, valign: "middle", margin: 0 });
  s.addText("Integracija mining kandidata u trening · jača regularizacija slabih modela · few-shot pristup za rijetke pasmine", {
    x: MX + 1.9, y: 5.45, w: CW - 2.2, h: 0.85, fontFace: BF, fontSize: 11.5, color: "D6E2E2", valign: "middle", lineSpacingMultiple: 1.1, margin: 0,
  });
  footer(s);
}

// ════════════════════════════════════════════════════════════════════════
// SLIDE 13 — HVALA
// ════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide(); darkBase(s);
  s.addShape(pres.shapes.OVAL, { x: -1.8, y: 3.6, w: 5.6, h: 5.6, fill: { type: "none" }, line: { color: INK2, width: 26 } });
  s.addShape(pres.shapes.RECTANGLE, { x: W - 0.18, y: 0, w: 0.18, h: H, fill: { color: TEAL }, line: { type: "none" } });
  s.addText("HVALA NA PAŽNJI", { x: 1.0, y: 2.5, w: 11, h: 0.4, fontFace: HF, fontSize: 13, bold: true, color: TEAL, charSpacing: 3, margin: 0 });
  s.addText("Pitanja i rasprava", { x: 0.96, y: 2.95, w: 11, h: 1.0, fontFace: HF, fontSize: 38, bold: true, color: WHITE, margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 1.0, y: 4.3, w: 5.5, h: 0, line: { color: INK2, width: 1.5 } });
  s.addText("Patrik Hrvoje Radočaj · FERIT · Strojno učenje · 2026.", {
    x: 1.0, y: 4.55, w: 11, h: 0.4, fontFace: BF, fontSize: 13, color: FOG, margin: 0,
  });
  s.addText("Višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina pomoću dubokog učenja", {
    x: 1.0, y: 5.0, w: 10.5, h: 0.4, fontFace: BF, fontSize: 11, italic: true, color: MUTE, margin: 0,
  });
}

pres.writeFile({ fileName: A("prezentacija_strojno_ucenje_v3.pptx") }).then((fn) => {
  console.log("Saved:", fn);
});
