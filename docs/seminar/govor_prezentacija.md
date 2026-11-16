# Govorna skripta — Višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina

**Cilj: ~5 minuta.** Za to koristi samo **SKRAĆENO** natuknice iznad svakog slajda.
Puni govor ispod je rezerva / priprema ako želiš reći više.
Brojke su zaokružene da ih je lako izgovoriti; govori u prvom licu, ne čitaj doslovno.

---

## Slajd 1 — Naslovnica  ⏱ ~20 s

**SKRAĆENO:**
- Ime + naslov projekta.
- Iz slike: prvo prepoznaj vrstu, zatim pasminu.

> Dobar dan. Moje ime je Patrik Hrvoje Radočaj i predstavit ću svoj projekt iz
> strojnog učenja i računalnog vida — višerazinski sustav za prepoznavanje
> životinjskih vrsta i pasmina pomoću dubokog učenja.
>
> Ukratko, cilj je iz obične fotografije prvo prepoznati o kojoj se životinji radi,
> a zatim odrediti i njezinu pasminu. Kroz izlaganje ću pokazati kako sam to
> riješio, koje sam podatke koristio i kakve sam rezultate postigao.

---

## Slajd 2 — Sadržaj  ⏱ ~10 s

**SKRAĆENO:**
- Šest cjelina: motivacija → arhitektura → podaci → modeli → rezultati → zaključci.

> Izlaganje ima šest cjelina. Prvo motivacija — zašto je problem težak i zašto
> sam odabrao baš ovakav pristup. Zatim arhitektura sustava, pa podaci i njihova
> priprema. Nakon toga modeli koje sam koristio, rezultati, i na kraju analiza i
> zaključci.

---

## Slajd 3 — Problem i motivacija  ⏱ ~30 s

**SKRAĆENO:**
- Ne samo vrsta — i pasmina; to je teško: fini detalji, realni uvjeti, neuravnoteženi podaci.
- Zato hijerarhija (2 faze). Dokaz: jednofazni model = tek ~0,21 mAP.

> Ključna ideja je da nije dovoljno samo prepoznati da je na slici životinja —
> treba razlikovati vrstu, a onda i pasminu unutar te vrste.
>
> A to je teško iz tri razloga. Prvo, **fina granularnost** — pasmine se često
> razlikuju samo u boji dlake i sitnim detaljima; labrador i zlatni retriver imaju
> gotovo identičnu siluetu. Drugo, **realni uvjeti** — slike su promjenjive kvalitete,
> objekti su mali, zaklonjeni, pozadine raznolike. I treće, **podaci** — dolaze iz
> različitih izvora s nekonzistentnim oznakama i jako neuravnoteženim klasama.
>
> Zato sam odabrao **hijerarhijsku dekompoziciju**: umjesto jednog modela koji
> rješava sve odjednom, problem dijelim u dvije faze. I to nije samo intuicija —
> kontrolni eksperiment to potvrđuje: jednofazni model koji odmah pokušava
> pogoditi pasminu doseže tek mAP oko 0,21, daleko ispod ovog dvostupanjskog
> pristupa.

---

## Slajd 4 — Dvostupanjski pipeline  ⏱ ~30 s

**SKRAĆENO:**
- Faza 1: YOLO11s → vrsta + okvir → crop. Faza 2: EfficientNetV2-S → pasmina.
- Svaka faza rješava lakši zadatak. Direktni YOLO (dolje) ~0,21 → zato razdvajanje.

> Evo kako sustav izgleda. U **prvoj fazi** ulazna slika ide u YOLO11s detektor
> koji prepoznaje vrstu i lokalizira životinju — daje okvir oko nje. Zatim se taj
> dio slike izreže, uz malo konteksta okolo. U **drugoj fazi** taj izrezani isječak
> ide u EfficientNetV2-S klasifikator koji određuje pasminu.
>
> Prednost je što svaka faza rješava lakši, jasnije definiran zadatak.
>
> Dolje je kontrolni eksperiment — direktni YOLO koji preskače fazu vrste i odmah
> traži pasminu. On doseže samo oko 0,21 i nakon šesnaeste epohe počinje
> regresirati, pa se koristi najbolji checkpoint, ne zadnji. To je upravo razlog
> zašto razdvajanje na dvije faze ima smisla.

---

## Slajd 5 — Podaci i neuravnoteženost  ⏱ ~30 s

**SKRAĆENO:**
- 19 vrsta, ~26 tis. slika (detektor) + 45 tis. mined (pasmine).
- Glavni izazov = neuravnoteženost; nakon balansiranja nijedna klasa nije prazna (~24:1).

> Za detektor vrste koristio sam skup od **19 vrsta i nešto preko 26 tisuća slika**.
> Za drugu fazu, klasifikaciju pasmina, dodatnih **45 tisuća slika** prikupljenih
> rudarenjem, o čemu ću još govoriti.
>
> Graf desno pokazuje glavni izazov — neuravnoteženost. Najzastupljenije su mačka
> i svinja s preko četiri tisuće slika, dok pas, llama i guska imaju tek nekoliko
> stotina. Bitno je da nakon ciljanog balansiranja i rudarenja **nijedna klasa više
> nije prazna** — omjer najveće i najmanje klase pao je na otprilike 24 naprema 1.
>
> I dalje je izražen, pa je kvaliteta i balansiranje podataka jednako važno kao i
> sam izbor modela.

---

## Slajd 6 — Sintetički kompoziti  ⏱ ~25 s

**SKRAĆENO:**
- Pojačavam rijetke klase: izreži životinje → posloži na novu pozadinu → okviri automatski.
- Omjer 40 % sintetika / 40 % real / 20 % augmentirano.

> Jedan od načina na koji sam pojačao rijetke klase su **sintetički kompoziti**.
>
> Postupak ima tri koraka. Prvo, iz postojećih slika izrežem pojedinačne životinje
> koristeći već postojeće oznake. Drugo, više tih izrezaka različitih vrsta
> posložim na novu pozadinu, uz nasumično pomicanje i skaliranje. I treće,
> okviri oko svake životinje generiraju se automatski.
>
> Desno vidite rezultat — donji izresci su komponente, a lijeva slika je gotov
> kompozit koji model nikad prije nije vidio. Omjer u treningu je 40 posto
> sintetike, 40 posto stvarnih slika i 20 posto augmentiranih primjera. Tako
> dobivam raznolikost koja posebno pomaže slabo zastupljenim vrstama.

---

## Slajd 7 — Rudarenje i alati  ⏱ ~25 s

**SKRAĆENO:**
- Detektor preko neoznačenih slika → 324 tis. kandidata, 20 tis. prioritetnih.
- Dva alata: Box review (okviri) + Label audit (oznake). Originali se ne mijenjaju.

> Drugi izvor podataka je **rudarenje**. Istrenirani detektor pustim preko velike
> količine neoznačenih slika; tako je nastalo preko **324 tisuće kandidata**, od čega
> sam izdvojio **20 tisuća visokoprioritnih** za pregled. Bitno — originalne oznake se
> nikad ne mijenjaju automatski.
>
> Lijevo dolje su primjeri gdje detektor pronalazi životinje koje su prije bile
> promašene, uključujući zanimljive cross-species zabune. A da bih to mogao
> učinkovito pregledati, izradio sam dva alata: **Box review** za potvrdu i ispravak
> okvira, i **Label audit** za reviziju kvalitete oznaka. To pretvara ručni posao u
> brz, kontroliran proces.

---

## Slajd 8 — Modeli i hiperparametri  ⏱ ~25 s

**SKRAĆENO:**
- Faza 1: YOLO11s, brz jednofazni detektor, 1280 px.
- Faza 2: EfficientNetV2-S, 384 px, modularno — zaseban model po vrsti, ulaz = crop.

> Sada o samim modelima.
>
> **Prva faza** je YOLO11s — jednofazni detektor koji u jednom prolazu predviđa sve
> okvire i klase, pa je brz i pogodan za interaktivnu upotrebu. U najnovijoj verziji
> treniram ga na rezoluciji 1280 piksela.
>
> **Druga faza** je EfficientNetV2-S, pretreniran na ImageNetu, na rezoluciji 384
> piksela. Key detalj je da je **modularan** — treniram zaseban klasifikator po
> svakoj vrsti, a ulaz mu je isječak iz prve faze, ne cijela slika. Time se svaki
> model fokusira na uži, lakši problem.

---

## Slajd 9 — Rezultati: detektor vrste  ⏱ ~30 s

**SKRAĆENO:**
- Evolucija mAP@50: 0,35 → 0,68 → 0,743 (+100 %), bez mijenjanja arhitekture.
- Konačno: mAP 0,74, precision 0,74, recall 0,70 → stabilna osnova.

> Ovo su rezultati detektora vrste i njegova evolucija kroz tri iteracije.
>
> Osnovni model na 640 piksela bio je tek 0,35 mAP. Poboljšanjem podataka skočio
> je na 0,68, a najnoviji „strogi“ model na 1280 piksela doseže **0,743**. To je
> ukupno preko sto posto napretka — i to ne mijenjajući arhitekturu, nego prije
> svega kvalitetu podataka i rezoluciju.
>
> Desno su konačne metrike: mAP na 50 je 0,74, precision 0,74, recall 0,70. Dakle
> detektor vrste je stabilna, pouzdana osnova za drugu fazu.

---

## Slajd 10 — Rezultati: klasifikatori pasmina  ⏱ ~30 s

**SKRAĆENO:**
- 15 modela, tri razine: zeleno >80 % (magarac 99, svinja 97), žuto 55–80 %, crveno pas/ovca.
- 9 od 15 prelazi 78 % → upotrebljivo.

> Za pasmine sam istrenirao **15 zasebnih klasifikatora**. Graf ih dijeli u tri razine.
>
> **Zeleno** su odlični modeli, preko 80 posto — magarac skoro 99, svinja 97, paun 95.
> **Žuto** su prihvatljivi, između 55 i 80 posto. A **crveno** su problematični —
> pas s 49 i ovca s 35 posto.
>
> Ono što je važno: **9 od 15 modela prelazi 78 posto** točnosti, što je sasvim
> upotrebljivo. Slabiji rezultati nisu slučajni i o njima govorim na sljedećem
> slajdu.

---

## Slajd 11 — Analiza: overfitting  ⏱ ~25 s

**SKRAĆENO:**
- Pas/ovca zakazuju zbog overfittinga: train ~99 %, val 35–49 %.
- Rješenje nije veći model nego više podataka → mining.

> Zašto pas i ovca zakazuju? Odgovor je **overfitting**.
>
> Lijevi graf pokazuje konvergenciju, desni gubitak na treningu i validaciji. Kod
> magarca — zelena priča — gubitak na treningu i validaciji padaju zajedno; model
> uči nešto prenosivo. Kod psa i ovce, suprotno: točnost na treningu ide prema 99
> posto, dok validacija ostaje na 35 do 49. Dakle model **pamti primjere umjesto da
> uči**.
>
> I ključno — rješenje nije veći ili dublji model. Uzrok je manjak raznolikih
> podataka, a odgovor je upravo rudarenje i prikupljanje primjera o kojem sam
> govorio ranije.

---

## Slajd 12 — Zaključci i smjerovi  ⏱ ~30 s

**SKRAĆENO:**
- Hijerarhija nužna · podaci > arhitektura · 9/15 ≥ 78 % · overfitting glavno usko grlo.
- Dalje: mining u trening, jača regularizacija, few-shot za rijetke pasmine.

> Da sažmem u četiri točke.
>
> Prvo — **hijerarhija je nužna**: jednofazni model doseže tek 0,21, dok
> dvostupanjski pristup čisto razdvaja lak problem vrste i težak problem pasmine.
> Drugo — **podaci su važniji od arhitekture**: samo boljim podacima i rezolucijom
> mAP je narastao s 0,35 na 0,74. Treće — **pokrivenost pasmina je solidna**, 9 od 15
> modela preko 78 posto. I četvrto — **overfitting je glavno usko grlo** kod slabih
> klasa, a mining pipeline je izravan odgovor na to.
>
> Smjerovi dalje: integracija miniranih kandidata u trening, jača regularizacija
> slabih modela i few-shot pristup za najrjeđe pasmine.

---

## Slajd 13 — Hvala / pitanja  ⏱ ~10 s

**SKRAĆENO:**
- Hvala na pažnji, otvoren za pitanja.

> To je sve s moje strane. Hvala na pažnji — otvoren sam za pitanja.

---

## Pripremljeni odgovori na moguća pitanja

- **Zašto YOLO, a ne neki drugi detektor?** Jednofazni je, brz, dobro balansira
  točnost i brzinu — pogodan za interaktivnu inferenciju i za rudarenje preko
  velikih količina slika.
- **Zašto EfficientNetV2-S, a ne veći model?** Dobra točnost uz mali broj
  parametara; budući da treniram zaseban model po vrsti, kompaktnost je bitna.
- **Zašto je ovca tako loša?** Malo raznolikih primjera + vizualno vrlo slične
  pasmine → klasičan overfitting. Rješenje je više podataka, ne veći model.
- **Je li direktni YOLO bezvrijedan?** Nije — služi kao mjerni okvir (baseline)
  koji kvantificira koliko hijerarhija zapravo pomaže.
- **Kako osiguravaš da ne curiš podatke?** Originalne oznake se nikad ne mijenjaju
  automatski; mining kandidati idu u zasebnu reviziju prije bilo kakve integracije.
