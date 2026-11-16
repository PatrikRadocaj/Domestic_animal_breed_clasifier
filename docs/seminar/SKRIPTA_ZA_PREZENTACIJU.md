# SKRIPTA ZA PREZENTACIJU
## Višerazinski sustav za prepoznavanje životinjskih vrsta i pasmina

---

> **Kako koristiti ovu skriptu:**
> Tekst pod svakIM slajdom je ono što govoriš. Dijelovi u zagradama `[...]` su napomene za tebe, ne govoriš ih naglas.
> Ukupno trajanje: ~12–15 minuta.

---

## SLIDE 1 — NASLOVNI SLAJD

*[Dok se pojavljuje slajd, kratka pauza, kontakt očima s publikom]*

> "Dobar dan. Ovaj projekt bavi se automatskim prepoznavanjem životinjskih vrsta i pasmina na fotografijama.
>
> Ono što na prvi pogled izgleda kao jednostavan zadatak — 'je li to pas ili mačka' — zapravo je višerazinski problem koji zahtijeva dva potpuno različita modela koji rade sekvencijalno. Zašto? To ću objasniti odmah na sljedećem slajdu.
>
> Tehnologije koje koristimo su YOLO11 za detekciju i EfficientNetV2-S za klasifikaciju, sve implementirano u Pythonu s PyTorchom."

---

## SLIDE 2 — ZAŠTO VIŠERAZINSKI PRISTUP?

*[Ovo je najvažniji slajd — ovdje objasniš temeljnu motivaciju]*

> "Zašto ne možemo samo trenirati jedan model koji direktno prepoznaje pasminu?
>
> Postoje tri ključna razloga.
>
> **Prvi** — nije dovoljno samo detektirati vrstu. Treba razlikovati pasmine unutar iste vrste. Labrador i zlatni retriver razlikuju se gotovo samo bojom dlake. Border collie i australski ovčar imaju gotovo identičnu siluetu. To su izuzetno suptilne vizualne razlike.
>
> **Drugi razlog** — neuravnoteženost podataka je dramatična. U našem skupu, mačka ima 4.542 primjera, a patka samo 5. To je omjer 900:1. Svaki model koji uči na takvim podacima jednostavno nauči ignorirati rijetke klase.
>
> **Treći razlog** — podaci dolaze iz različitih izvora s nekonzistentnim oznakama i formatima. Sam pipeline za čišćenje podataka zahtijeva značajan posao.
>
> **A što se dogodi ako ipak pokušamo jedan model?**
>
> Upravo to smo isprobali kao kontrolni eksperiment — trenirali smo direktni YOLO breed model koji pokušava prepoznati 100+ pasmina odjednom, bez ikakve hijerarhije. Nakon 25 epoha treniranja, postiže mAP@50 = 0.228 s recallom ispod 0.22. Model nešto uči — ali je daleko od praktično upotrebljivog. Preciznost pada s daljnjim treningom jer se klasifikacijske glave ne mogu inicijalizirati za 100+ vizualno sličnih klasa.
>
> To je direktan empirijski dokaz da hijerarhija nije samo arhitekturalna preferencija — nego nužnost."

---

## SLIDE 3 — NEURAVNOTEŽENOST KLASA

*[Pokaži na grafikon]*

> "Ovaj grafikon prikazuje distribuciju klasa u skupu podataka za detekciju vrsta.
>
> Ukupno imamo 27.433 slika, 18 klasa, s podjelom 82% trening, 12% validacija, 6% test.
>
> Vidite dramatičnu razliku — mačka dominira s tisućama primjera, dok patka, guska i neke druge vrste imaju svega nekoliko ili nula primjera.
>
> Ovo je bio jedan od ključnih izazova. Standardni model treniran na ovakvim podacima nauči klasificirati gotovo sve kao mačku ili psa jer time postiže visoku prosječnu točnost, a pritom nikad ne vidi gusku.
>
> Rješenje su bile tri tehnike: sintetički kompoziti, PHR slike koje smo ručno dodali, i balansiran sampling koji osigurava da svaka klasa bude jednako zastupljena u svakom mini-batchu."

---

## SLIDE 4 — SINTETIČKI KOMPOZITI

*[Ovo je tehnički zanimljiv dio — pokaži entuzijazam]*

> "Sintetički kompoziti su naše rješenje za nedostatak podataka za rijetke klase.
>
> Ideja je jednostavna: uzmeš izrezak životinje iz postojeće slike — recimo patku — i postaviš je na realnu pozadinsku teksturu. Dodaš malo šuma: pozicijski jitter ±6%, skaliranje između 0.88 i 1.20 puta, i automatski generiraš bounding box za novu poziciju.
>
> Ovo se radi **on-the-fly** za vrijeme treniranja — nije potrebno unaprijed generirati tisuće statičnih datoteka. Klasa `HybridSpeciesYoloDataset` u svaki batch miješa: 40% sintetičke slike, 40% realne slike i 20% augmentiranih.
>
> Na desnoj strani vidite primjer — realni izresci životinja postavljeni na pozadinsku teksturu s automatski generiranim bounding boxovima.
>
> Ovo je praktično rješenje koje funkcionira — ali nije magija. Sintetičke slike su uvijek nešto 'sterilnije' od realnih jer životinja ne interagira s pozadinom na prirodan način. Zato imamo i 40% realnih slika u mixu."

---

## SLIDE 5 — ARHITEKTURA I KONFIGURACIJA

*[Srednji dio prezentacije — pregled tehničkih detalja]*

> "Arhitektura sustava su dvije sekvencijalne faze.
>
> **Faza 1: YOLO11s species detektor**
> Treniran na 640×640 pikselima, 100 epoha, batch 16, s cosine learning rate schedulerom i hybrid datasetom koji smo maloprijed opisali. Završen finalni run ima 40 epoha s mAP@50 = 0.781.
>
> **Faza 2: EfficientNetV2-S breed klasifikatori**
> Ovdje je ključna odluka — **jedan model po vrsti**, a ne jedan model za sve. Dakle, imamo odvojeni model za pse, odvojeni za mačke, odvojeni za goveda, i tako dalje. Svaki od tih modela treniran je na 384×384 pikselima, 60 epoha, s AdamW optimizatorom i label smoothingom.
>
> Zašto jedan model po vrsti? Jer model koji klasificira samo pasmine pasa vidi puno manje klasa, ima puno bolje balansiran skup podataka za tu vrstu, i može se specijalizirati. Modularnost znači da možemo dodati novu vrstu bez rušenja svega ostaloga.
>
> **Kontrolni eksperiment** — direktni YOLO breed model — vidite na dnu desno označen crveno. 25 epoha, mAP@50 = 0.228. Model nešto uči ali daleko ispod onoga što postiže dvostupanjski pristup."

---

## SLIDE 6 — SPECIES DETEKTOR — REZULTATI

*[Pohvali se rezultatima — ovo je stvarni napredak]*

> "Evo konkretnih brojki za species detektor.
>
> Finalni run postiže mAP@50 = **0.781**, Precision 0.788, Recall 0.755, i mAP@50-95 = 0.477 što je stroga mjera koja zahtijeva precizno preklapanje.
>
> Na desnoj strani vidite tablicu iteracija — ovo je priča o tome kako kvaliteta podataka mijenja sve.
>
> Počeli smo s ranim eksperimentom na 0.346. Dodavanjem guske i srodnih rijetkih klasa skočili smo na 0.617. Poboljšanjem box thresholda na 0.666. Ručno dodanim PHR slikama na 0.679. I konačno — finalnim runcom sa 40 epoha i novim balansiranim skupom — **0.781**.
>
> To je poboljšanje od **126%** u odnosu na početak, isključivo kroz bolju pripremu podataka i duže treniranje. Arhitektura modela — YOLO11s — je ista od prve do zadnje iteracije. Ovo je praktičan dokaz da podaci znače više od arhitekture."

---

## SLIDE 7 — BREED KLASIFIKATORI — REZULTATI

*[Ovo je složen slajd — drži se ključnih poruka]*

> "Za breed klasifikatore imamo 16 istrenirano modela — za svaku vrstu s dovoljno podataka.
>
> Rezultati variraju dramatično ovisno o broju i raznolikosti podataka.
>
> **Dobri rezultati** — mazga postiže 98.4% top-1 točnosti. Paunovi 94.6%, deve 94.3%, kunići 91.9%, kokoši 90.2%. Ove vrste imaju relativno malo pasmina s jasnim vizualnim razlikama.
>
> **Problematični rezultati** — pas postiže 47.2% i ovca 35.3%. Zašto? Ovca je posebno zanimljiv slučaj — pasmine se razlikuju gotovo isključivo bojom runa. To je gotovo nemoguće pouzdano klasificirati bez visokokvalitetnih, raznolikih slika.
>
> Ono što je bitno — svih 16 modela je završeno i funkcionira. Oni s nižom točnošću nisu 'pokvareni', nego odražavaju inherentnu težinu zadatka i nedostatak raznolikih trening primjera."

---

## SLIDE 8 — KONVERGENCIJA I OVERFITTING

*[Malo tehnički, ali pokaži da razumiješ problem]*

> "Ovaj grafikon prikazuje zašto pas i ovca zaostaju — i što s tim.
>
> Vidite klasičan pattern overfittinga: train točnost za psa ide prema 99%, ali val točnost stagnira na 49%. Isti pattern za ovcu, samo još izrazitiji.
>
> Uzrok nije arhitektura modela — EfficientNetV2-S je odličan model. Uzrok je nedostatak **raznolikih** primjera. Imamo mnogo slika zlatnog retrivera, ali malo slike border collija u različitim uvjetima osvjetljenja, pozama i pozadinama.
>
> Rješenje je mining pipeline — sustav koji automatski pretražuje neanotirane skupove podataka i pronalazi životinje koje dataset nije označio. Pronašli smo 324.633 kandidata. Ovi kandidati direktno adresiraju nedostatak raznolikosti za problematične pasmine.
>
> Integracija mining kandidata u trening skupove je sljedeći korak u razvoju projekta."

---

## SLIDE 9 — ALATI I INFRASTRUKTURA

*[Naglasi da projekt nije samo model — nego kompletan sustav]*

> "Projekt nije samo skup modela — nego funkcionalan sustav koji se može koristiti i demonstrirati.
>
> **Mining i review alati** — lijevo vidite panel s primjerima iz mininga. Detektor pronalazi životinje koje su propuštene ili pogrešno označene u originalnim skupovima. Gornji lijevi primjer pokazuje govedo koje je pogrešno označeno kao konj. Srednji prikazuje neanotirani konj koji je model pronašao. Donji prikazuje mačku pogrešno označenu kao zec.
>
> Desno je box review app — interaktivni alat koji omogućuje ručni pregled kandidata iz mininga. Možeš potvrditi, odbaciti ili korigirati svaki bounding box bez mijenjanja originalnih oznaka.
>
> Osim toga, projekt ima:
> - **17 CLI naredbi** koje pokrivaju cijeli pipeline — od preuzimanja podataka do inferencije
> - **Web aplikaciju** za upload slike i inferenciju u browseru
> - **AMD ROCm GPU** runtime — izoliran Python 3.12 environment koji ne dira globalni PATH
> - **GitHub CI i pytest** — 8 test datoteka, automatski pipeline za provjeru integriteta
>
> Ovo znači da projekt može koristiti netko tko nije gledao ni jednu liniju koda — samo pokrenuti CLI ili web app."

---

## SLIDE 10 — ZAKLJUČAK

*[Završi jasno i samopouzdano]*

> "Pet ključnih zaključaka.
>
> **Prvo** — hijerarhija je nužna, ne izborna. Direktni YOLO breed model s 25 epoha postiže mAP@50 = 0.228 s recallom ispod 22%. Dvostupanjski pristup daje smislene rezultate za svih 16 vrsta. Eksperiment jasno potvrđuje odluku.
>
> **Drugo** — podaci su važniji od arhitekture. Isti YOLO model s poboljšanim podacima: 0.346 → 0.781. Više od 126% poboljšanja bez ikakve promjene arhitekture.
>
> **Treće** — 9 od 15 breed modela postiže ≥ 78% top-1 točnosti. Donkey 98.4%, pig 96.7%, peafowl 94.6%. Moduli s dovoljno raznolikih podataka konvergiraju stabilno.
>
> **Četvrto** — dog i sheep zaostaju zbog overfittinga, ne zbog arhitekture. Train 99% vs val 49% i 35%. Mining pipeline s 324 tisuće kandidata direktan je odgovor.
>
> **Peto** — sustav je operativan, ne samo istreniran. CLI, web app, mining alati, review alati, CI — sve postoji i funkcionira.
>
> Hvala na pažnji — rado odgovaram na pitanja."

---

## MOGUĆA PITANJA I ODGOVORI

---

**P: Zašto YOLO11 a ne neki drugi detektor?**

> "YOLO11 je trenutno jedan od najbržih i najtočnijih open-source detektora za real-time primjenu. Ključno je da ima odlično podržan Python API, dobar export za deployment i integrirano rješenje za breed detection u jednom alatnom okviru. Za species detekciju nam treba brzina jer je to prvi stupanj pipeline-a."

---

**P: Zašto EfficientNetV2-S za breed klasifikator?**

> "EfficientNetV2-S je odabran zbog omjera točnosti i veličine modela. Predtrenirani je na ImageNet-21k što znači da već 'razumije' vizualne značajke životinja. S 384×384 ulaznom rezolucijom dobiva dovoljno detalja za razlikovanje finih vizualnih razlika između pasmina. Alternativa bi bio ViT — koji je provabno točniji za fine-grained klasifikaciju — ali zahtijeva puno više VRAM-a i duže inferiranje."

---

**P: Koliko traje inferencija jedne slike?**

> "Na GPU-u — specifično AMD ROCm setup koji koristimo — YOLO detekcija traje otprilike 20-50ms po slici. Breed klasifikacija još 10-20ms. Dakle, ukupno ispod 100ms za cijeli pipeline na jednoj slici, što je dovoljno za real-time primjenu."

---

**P: Kako rješavaš problem kad životinja nije u skupu podataka?**

> "Pipeline vraća samo species predikciju. Ako YOLO detektira vrstu za koju nema breed modela — ili ako je pouzdanost detekcije ispod praga od 0.25 — vraćamo samo vrstu bez pasmine. Web app i JSON izlaz eksplicitno označavaju je li breed klasifikacija dostupna ili ne."

---

**P: Što je mining pipeline točno?**

> "Mining pipeline pokreće naš species detektor na neanotiranim skupovima podataka — slikama koje imaju samo oznake vrsta bez pasmina, ili čak bez ikakvih oznaka. Detektor pronalazi životinje, sprema bounding boxove i confidence score za svakog kandidata. Rezultat je 324.633 kandidata koji su vizualno pregledani kroz box review app. Ovi kandidati su potencijalni trening primjeri za breed modele, posebno za pasmine koje su trenutno slabo zastupljene."

---

**P: Zašto sintetički kompoziti umjesto data augmentacije?**

> "Augmentacija — horizontalni flip, brightness, rotation — ne rješava problem kada imaš 5 slika patke ukupno. Augmentacija samo variira tih 5 slika, ali model i dalje vidi samo iste 5 originalne primjere. Sintetički kompoziti stvaraju vizualno nove scene — patka na novoj pozadini, u novoj poziciji, u novom mjerilu. To je više kao novi primjer nego augmentirani stari primjer."

---

**P: Koliko je projekt big u smislu koda?**

> "Projekt ima 17 CLI entry pointova definiranih u pyproject.toml, pokriva cijeli pipeline od preuzimanja podataka, čišćenja, treniranja, mininga, web aplikacije do inferencije. Koristi Hydra i OmegaConf za konfiguraciju eksperimenata i wandb za tracking metrika. Dakle, nije to samo par skripta — to je modularan, konfigurabilni sustav."

---

## NAPOMENE ZA NASTUP

- **Tempo**: govori sporije nego misliš da trebaš. Publika treba vremena za procesiranje.
- **Slide 2**: najvažniji slide. Ako ništa drugo, ovaj objasni jako dobro.
- **Slide 6**: pohvali se rezultatima — 0.781 je dobar rezultat. Nemoj biti skroman.
- **Slide 7**: ne brini za loše rezultate (ovca, pas) — to su zanimljivi slučajevi, ne greška.
- **Kontrolni eksperiment**: uvijek naglasi da je to **namjerna** provjera koja **potvrđuje** naš pristup, ne neuspjeh.
- **Ukupno trajanje**: oko 12-15 minuta + pitanja. Ne žuri.
