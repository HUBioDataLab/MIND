# RNA Virtual Atom Fast Experiment - Agent Handoff

## 1. Gorev ve karar sorusu

Bu gorevin amaci RNA'nin 23-slot temsilini gercekten kullanan gecici bir `.pt`
chunk uretmek ve C1' uzerine yigilan virtual atomlarin protein CB problemine benzer
bir anchor-atom recall cokusu yaratip yaratmadigini kontrollu bir fast A/B egitimiyle
olcmektir.

Ana karar sorusu:

> RNA virtual atomlari C1' koordinatinda tutuldugunda C1' recall saglikli kaliyor mu,
> yoksa model bu koordinati cogunlukla V olarak ogrenip C1' atomunu kaybediyor mu?

Bu deney tamamlanmadan RNA icin BoltzGen/pseudo-atom yaklasimina gecme karari
verilmemelidir. Deney yalnizca bir training run degil; metric vocabulary duzeltmesi,
gecici cache uretimi, cache butunluk testleri, esit kosullu A/B fast run ve log
yorumlamasindan olusur.

## 2. Baslamadan once okunacaklar

Proje ve aktif durum icin:

- `AGENTS.md`: proje mimarisi, veri yollari, komutlar ve repo kurallari.
- `PROJECT_STATUS_JULY_2026.md`: Temmuz 2026 durum ozeti ve RNA cache kanitlari.
- `Docs/CB_DUSUKLUGU_ACIKLAMA.md`: protein CB recall cokusu ve virtual atom nedeni.
- `Docs/features/EPOCH_BALANCED_MULTIMODAL_BATCHING.md`: yeni multimodal sampler.
- `Docs/NEXT_GOALS.md`: genel sonraki hedefler (varsa guncel kodla karsilastir).
- `/home/yusuf/.codex/skills/validate-feature/SKILL.md`: MIND'e ozel test akisi.

Ilgili kod:

- `data_loading/adapters/rna_adapter.py`: 23-slot standardizasyon ve C1' anchor.
- `data_loading/data_types.py`: `UniversalMolecule/Block/Atom` siniflari.
- `data_loading/cache_to_pyg.py`: streaming PKL -> PyG `.pt` donusumu.
- `data_loading/create_chunk_metadata.py`: sampler metadata uretimi.
- `core/metrics/rna_atom_types.py`: RNA per-position metric vocabulary.
- `core/metrics/rna_per_atom_metrics_callback.py`: RNA atom/nucleotide metrikleri.
- `core/pretraining_config_fast.yaml`: uc epochluk fast training sablonu.
- `core/train_pretrain.py`: chunk kesfi, callback ve training giris noktasi.

Calisma agaci halihazirda dirty olabilir. `git status --short` ve ilgili
`git diff` ciktilarini once oku. Kullaniciya veya onceki agente ait degisiklikleri
resetleme, checkout etme ya da ezme.

## 3. Simdiye kadarki kesin veri bulgulari

Mevcut bes RNA `.pt` chunk daha once dogrudan tarandi:

| Chunk | Molekul | Atom | `V*` pos_code | `z == 119` |
|---|---:|---:|---:|---:|
| 0 | 1,957 | 2,455,144 | 0 | 0 |
| 1 | 1,957 | 3,765,136 | 0 | 0 |
| 2 | 1,957 | 4,098,440 | 0 | 0 |
| 3 | 1,957 | 3,830,017 | 0 | 0 |
| 4 | 1,955 | 4,035,613 | 0 | 0 |
| Toplam | 9,783 | 18,184,350 | 0 | 0 |

Bes source PKL de tarandi: 874,604 block icinde `V*` veya
`is_virtual=True` atom yoktu. Kontrol olarak protein chunk 0'da hem `V*` hem
`z == 119` sayisi tam 3,460,089 bulundu; dolayisiyla tarama yontemi V atomunu
tespit edebiliyor.

`data_loading/cache_to_pyg.py` atomun `pos_code` alanini degistirmeden `.pt`ye
kopyalar. Bu nedenle mevcut RNA `.pt`lerde V olmamasi converter'in V'yi silmesi
degil, kaynak PKL'nin zaten V icermemesi demektir.

Eski RNA loglari da bu bulguyla uyumludur:

- Nisan ve Haziran runlarinda tek RNA grafin atom sayisi sikca 23'e tam bolunmez.
- Son metrik runlarinda `Base == BaseR` ve RNA nucleotide overall accuracy
  `Real-only` accuracy ile aynidir. Gercek V slotlari olsaydi bu iki toplam ayrisirdi.
- Logdaki eski `V` satiri gercek V kaniti degildir; vocabulary disindaki gercek
  N6/O2/O4/N4 atomlari yanlislikla V sinifina aktarilmistir.

Referans loglar:

- `logs/June2026/rna_nt_metricsfast_20260617_122200.log`
- `logs/June2026/rna_nt_metrics_2epoch_12k_20260617_122646.log`
- `logs/June2026/multidomain_100ep_pdb_unimol_rna_20260609_065900.log`

Eski fast runin son RNA ozeti: C1' recall 0.9371, backbone 0.8617, BaseR
0.5307, nucleotide all-or-nothing 0.3742. Eski iki-epoch buyuk runin son ozeti:
C1' recall 0.8381, backbone 0.9256, BaseR 0.8213, all-or-nothing 0.6799.
Bunlar baglamsal baseline'dir; vocabulary ve aktif kod farki nedeniyle yeni A/B
runlarinin yerine gecmez.

## 4. Once duzeltilecek metric vocabulary problemi

### Sorun

RNA base'i her nucleotide'da en fazla 11 slot kaplar, fakat A/U/G/C boyunca
birlesen benzersiz atom adlari 11 degil 15'tir:

```text
N9, C8, N7, C6, C5, C4, N3, C2, N1, N2, O6, N6, O2, O4, N4
```

**Onemli:** `MAX_SIDECHAIN_ATOMS=11` degistirilmeyecek. Canonical A/U/G/C
arasinda tek nucleotide'da en cok 11 gercek base agir atomu (G) bulunur. Buradaki
15 yalniz farkli nucleotide turleri uzerindeki position-name union'i ve metric
etiketi sayisidir; modelin element vocabulary'si veya fiziksel slot sayisi degildir.

Mevcut `RNA_SIDECHAIN_ATOMS` yalnizca ilk 11 etiketi icerir ve N6, O2, O4,
N4'u atlar. Buna karsilik ayni dosyadaki purine/pyrimidine referans listelerinde
bu atomlar zaten bulunur. Yani "bir residue'daki 11 base slotu" ile "tum base'ler
arasindaki unique position vocabulary" birbirine karistirilmistir.

Callback'in mevcut `_pos_code_to_atom_class()` fonksiyonu vocabulary disindaki
her seyi `V` dondurur. Sonuc olarak gercek N6/O2/O4/N4 atomlari V satirinin
support ve recall hesabina karisir. Bu, gercek virtual atomlarin davranisini
olcmeyi imkansizlastirir.

Bu yalnizca metric etiketleme problemidir. MIND MLM head atom adini degil
elementi (C/N/O/P/V vb.) tahmin eder. Position vocabulary'yi 15 base adina
tamamlamak model output boyutunu veya training hedeflerini degistirmez; metrikte
hangi fiziksel pozisyondaki elementin dogru tahmin edildigini dogru gruplar.

### Gerekli duzeltme

1. `RNA_SIDECHAIN_ATOMS` standart A/U/G/C'nin 15 benzersiz base atom adini
   icermeli. `ALL_RNA_ATOMS` boylece 12 backbone + 15 base + 1 V = 28 etiket
   olmali.
2. Yalnizca bos/gercekten virtual `pos_code` (`V`, `V1`, ...) V sinifina gitmeli.
3. Vocabulary disindaki modifiye gercek atomlar (`N5`, `N8`, `C7`, `OP3` vb.)
   sessizce V olmamali. Per-position tablo icin `None`/ayri OOV-real etiketiyle
   atlanmali ve OOV sayaci raporlanmali; element-level overall ve uygun real-base
   metriklerine katilmaya devam edebilir.
4. Callback'teki genis `except Exception: pass` nedeniyle metric hatalari gorunmez
   olabilir. Bu deney sirasinda en azindan ilk hatayi acikca logla veya strict test
   modunda exception'i yeniden firlat.

Minimum odakli testler:

```python
assert _pos_code_to_atom_class('N6') == 'N6'
assert _pos_code_to_atom_class('O2') == 'O2'
assert _pos_code_to_atom_class('O4') == 'O4'
assert _pos_code_to_atom_class('N4') == 'N4'
assert _pos_code_to_atom_class('V7') == 'V'
assert _pos_code_to_atom_class('OP3') != 'V'
assert len(ALL_RNA_ATOMS) == 28
assert len(ALL_RNA_ATOMS) == len(set(ALL_RNA_ATOMS))
```

Ek olarak sentetik bir callback batch/testi kur: N6/O2/O4/N4 supportlari kendi
satirlarina, yalniz V1/V2 supportlari V satirina gitmeli. Bu test gecmeden cache
uretimine veya training'e gecme.

## 5. Kodun git kokeni

Asagidaki komutlarla yeniden dogrula:

```bash
git blame -L 35,70 -- core/metrics/rna_per_atom_metrics_callback.py
git blame -L 33,63 -- core/metrics/rna_atom_types.py
git show b9fc39c:core/per_atom_metrics_callback.py | nl -ba | sed -n '135,145p'
git log --follow --format='%h %ad %an <%ae> %s' --date=short -- \
  core/metrics/rna_per_atom_metrics_callback.py
```

Bulgular:

- Bugunku dosyadaki tam `_pos_code_to_atom_class()` helper'i uncommitted'dir;
  `git blame` bunu `00000000 Not Committed Yet` gosterir. Bu helper, Haziran RNA
  nucleotide/block metrikleri bu calisma oturumunda genisletilirken Codex tarafindan
  eklendi. Dolayisiyla bu tam satira commitlenmis bir kisi adi atamak yanlistir.
- Ancak "vocabulary disindaysa V yap" davranisi ilk RNA callback'inde zaten vardi.
  Arya Zeynep Mete'nin `b9fc39c` (`Add per-atom metrics for RNA`, 2026-04-21)
  commitinde su mantik bulunur:

```python
ALL_RNA_ATOMS.index(c) if c in ALL_RNA_ATOMS else ALL_RNA_ATOMS.index('V')
```

- Eksik `RNA_SIDECHAIN_ATOMS` listesi ve `ALL_RNA_ATOMS` birlesimi de ayni
  `b9fc39c` commitinden gelir.
- Yusuf'un `0ed1f1d` commitindeki `_pos_code_to_element()` bu spesifik hatanin
  kaynagi degildir; gercek atomlarda ilk karakterden elementi, yalniz `V*` icin V'yi
  dondurur.

Muhtemel dusunce suydi: Metric calculator sabit bir class index bekliyordu;
bilinmeyen `pos_code` icin `ALL_RNA_ATOMS.index()` crash etmesin diye mevcut V
sinifi catch-all olarak kullanildi. Ayrica "11 sidechain slot" vocabulary'nin de
11 atom adi olmasi gerektigi varsayildi. Bu savunmaci fail-open yaklasim kodu
calisir tuttu, fakat kimyasal olarak farkli iki durumu - gercek ama listede eksik
atom ile sentetik virtual atomu - ayni etikete birlestirdi. Mevcut helper bu eski
varsayimi daha okunur bir fonksiyona tasiyarak hatayi surdurdu. Dogru savunma
unknown-real'i gorunur kilmak, V olarak gizlemek degildir.

## 6. Gecici virtual RNA cache uretimi

### Izolasyon ve kaynak

Uretim verisine dokunma:

```text
/home/yusuf/data/rna/processed_graphs_chunk_0/raw/universal_rna_chunk_0.pkl
```

Gecici kok:

```text
/tmp/mind_rna_virtual_fast/
```

Beklenen layout:

```text
/tmp/mind_rna_virtual_fast/
  processed_graphs_chunk_0/
    raw/
    processed/
```

Config'teki `chunk_dir` tam chunk'i degil
`/tmp/mind_rna_virtual_fast/processed_graphs` degerini almali. Loader bu basename
ile kardes `processed_graphs_chunk_0` klasorunu glob ile bulur.

### Donusturucu script

Repo'da su an eski universal PKL'yi yeniden standardize eden hazir CLI yoktur.
Agent, tercihen gecici veya odakli bir script yazmali. Script:

1. `cache_to_pyg.load_molecules_iteratively()` ile input PKL'yi molecule molecule
   okumali; tum chunk'i tek listede tutmamali.
2. `RNAAdapter(use_23atom_uniform=True)` olusturmali.
3. Yalniz adapter'in `STANDARD_RNA_BASES` kumesinde tanidigi RNA blocklari icin
   eski `block.atoms` listesini
   `adapter._standardize_residue_to_23_atoms(block.atoms, block.symbol, block_idx)`
   ile degistirmeli. Bilinmeyen/ion blocku 23 slota zorlamamali; sayip raporlamali.
4. Her cikis atomunda `block_idx`, `atom_idx_in_block` ve `entity_idx` tutarliligini
   acikca duzeltmeli/kontrol etmeli. RNA entity degeri mevcut gercek atomdan alinmali;
   kodda su an virtual icin sabit `0` kullanilir.
5. Molecule'lari `pickle.dump(..., protocol=pickle.HIGHEST_PROTOCOL)` ile streaming
   output PKL'ye yazmali.
6. Input molecule sayisi, output molecule sayisi, block sayisi, nucleotide bazli
   real/V slot sayisi ve C1'/C4'/first-atom anchor kullanim sayisini raporlamali.

Kaynak PKL eski atom adlarini ve koordinatlarini tasidigi icin raw CIF olmadan bu
donusum yapilabilir. Input objelerini yerinde degistirmek kaynak dosyayi degistirmez,
ancak ayni proses icinde once/sonra karsilastirma gerekiyorsa deep copy kullan.

Adapter'in "exactly 23" docstring'ine korlemesine guvenme: fonksiyon base tarafini
her zaman 11 slota tamamlar, fakat eksik backbone atomlarini doldurmaz. Bu nedenle:

- Her block icin base-slot sayisi tam 11 olmali.
- Yalniz 12 backbone atomu tam olan block toplam 23 atom olmali.
- Eksik backbone/modifiye residue'ler ayrica sayilip raporlanmali; otomatik fail
  nedeni olmamali.

Standart, eksiksiz base'lerde beklenen V sayisi:

| Nucleotide | Real base atomu | V slotu |
|---|---:|---:|
| A | 10 | 1 |
| G | 11 | 0 |
| C | 8 | 3 |
| U | 8 | 3 |

Eksik veya modifiye residue bu sayilari degistirebilir. Standart A/U/G/C raporu ile
modified raporunu ayri tut.

Mevcut source PKL auditinde 874,604 standard-etiketli blockun 2,954'u 11'den
fazla non-backbone atom tasidi. Canonical kimya icin 11 yeterli oldugundan bunlar
ekstra/alisilmadik atom adlari olarak ayrica raporlanmali; standardizer'in canonical
atomlari koruyup yalniz ekstralari kestigi assert edilmelidir.

### PyG donusumu

Kaynak RNA chunk ile ayni edge parametrelerini kullan:

```bash
conda activate mind_esa3d

python data_loading/cache_to_pyg.py \
  --input-pkl /tmp/mind_rna_virtual_fast/universal_rna_virtual_chunk_0.pkl \
  --output-dir /tmp/mind_rna_virtual_fast/processed_graphs_chunk_0 \
  --dataset-type rna \
  --cutoff 12.0 \
  --max-neighbors 48 \
  --force

python data_loading/create_chunk_metadata.py \
  --chunk-dir /tmp/mind_rna_virtual_fast/processed_graphs_chunk_0/processed \
  --overwrite
```

`cache_to_pyg.py` input PKL'yi output root altindaki `raw/` klasorune kendi
kopyalar. Urettigi dosya adi yaklasik olarak
`optimized_universal_rna_virtual_chunk_0_cutoff_12.0_neighbors_48.pt` olur.

## 7. Training oncesi zorunlu butunluk testleri

Tek bir audit script hem source negative control hem virtual positive control
uretmelidir. Training'i ancak asagidakilerin tumu gecerse baslat.

### Source negative control

- Source `.pt`: 1,957 graph, 2,455,144 atom.
- `pos_code == V veya V*` sayisi 0.
- `z == 119` sayisi 0.

### Virtual PKL positive control

- Input ve output molecule sayisi ayni: 1,957.
- En az bir `V*` vardir ve her V atomunda `is_virtual is True`,
  `virtual_type == 'V'`, `element == 'V'`.
- Her block'ta base tarafinda real + V toplam 11 slottur.
- C1' bulunan blocklarda her V koordinati C1' koordinatiyla tam esit veya
  `allclose(atol=1e-7)` olmalidir.
- C1' yoksa C4' fallback; o da yoksa first-atom fallback kullanimi sayilmalidir.
- N6/O2/O4/N4 gercek atom olarak kalmali; `is_virtual=False`, elementleri N/O ve
  koordinatlari korunmus olmali.
- Standard A/C/G/U ile modified residue sayilari ayri raporlanmalidir.

### Virtual PT positive control

- Graph sayisi 1,957 olmali.
- `V*` support sifirdan buyuk olmali.
- `z == 119` sayisi `V*` sayisina tam esit olmali. Standart RNA'da baska elementin
  119'a dusmesi vocabulary/element mapping sorunudur.
- `len(data.pos) == len(data.z) == len(data.pos_code)` ve son `slices['pos']`
  degeri toplam atom sayisina esit olmali.
- Her graph icin `block_idx`, `block_symbols` ve atom slice'lari tutarli olmali.
- `pos`, `edge_attr` ve distance alanlarinda NaN/Inf olmamali.
- Metadata `num_samples` ve `total_atoms` degerleri `.pt` audit sonucu ile ayni
  olmali.
- N6/O2/O4/N4 kendi pos_code'lariyla bulunmali ve V sayimina girmemeli.

Ek bir protein control scan'i yararlidir: bilinen protein chunk 0 taramasinda
V support sifirdan buyuk cikmiyorsa audit script bozuktur.

Audit sonucunu makine-okunur JSON ve kisa terminal ozeti olarak sakla. Run loguna
source/virtual PT path, SHA256 veya en azindan size+mtime, graph/atom/V sayilarini
yaz; yanlis chunk ile training baslatma riski boyle azalir.

## 8. Fast A/B training tasarimi

Eski log ile yeni virtual runi tek basina karsilastirmak yerine iki yeni run yap:

- **A - nonvirtual control:** metric vocabulary duzeltilmis aktif kod + mevcut RNA
  chunk 0.
- **B - C1 virtual treatment:** ayni commit/worktree, ayni seed ve ayni config;
  yalniz RNA `chunk_dir` temp virtual chunk'a bakar.

Iki YAML'yi `core/pretraining_config_fast.yaml` sablonundan `/tmp` altinda uret.
YAML'yi `yaml.safe_load/safe_dump` ile degistir; production config'i edit etme.

Her iki config icin sabit tutulacaklar:

- `seed: 42`
- `max_epochs: 3`
- `max_atoms_per_batch: 12000`
- PDB/UniMol/RNA `chunk_range: [0, 0]`
- model, optimizer, MLM ve sampler ayarlari
- ayni GPU modeli; mumkunse runlari sirali calistir

Degisecekler yalniz `wandb_run_name`, `out_path` ve B'deki RNA `chunk_dir`dir.
Configleri baslatmadan once diff et ve RNA disinda semantik fark olmadigini dogrula.

Ornek calistirma (GPU'yu once `nvidia-smi`/`make status` ile sec):

```bash
mkdir -p logs/July2026

CUDA_VISIBLE_DEVICES=<GPU> python -m core.train_pretrain \
  --config-yaml-path /tmp/mind_rna_virtual_fast/nonvirtual_control.yaml \
  > logs/July2026/rna_nonvirtual_vocabfixed_fast.log 2>&1

CUDA_VISIBLE_DEVICES=<GPU> python -m core.train_pretrain \
  --config-yaml-path /tmp/mind_rna_virtual_fast/c1_virtual.yaml \
  > logs/July2026/rna_c1_virtual_fast.log 2>&1
```

Arka planda baslatilacaksa timestamp ve `nohup` kullan; shell wrapper ile Python
child'in iki process olarak gorunmesinin normal oldugunu unutma. Iki training'i ayni
GPU'da eszamanli baslatma. Tamamlandiktan sonra exit/completion satirini ve traceback
olmadigini kontrol et.

Mevcut fast run yaklasik 10 dakika surmustur. Tek chunk standardizasyon + PyG
donusumu makine yukune gore kabaca 3-10 dakika, iki fast run toplam yaklasik
20-30 dakika beklenir. Audit ve metric duzeltmesiyle birlikte 30-60 dakikalik bir
calisma makuldur.

## 9. Logda zorunlu kanitlar

Her iki run icin:

- Config path ve `seed=42` gorunmeli.
- PDB, UniMol ve RNA chunk 0 yuklenmeli.
- RNA control run source yolu; treatment run temp yolu gostermeli.
- Her epoch loss finite olmali, traceback/OOM olmamali.
- Son validation/test RNA per-atom, overall, backbone/base, per-nucleotide
  all-or-nothing ve atom-level block tablolari basilmali.

Treatment run icin ayrica:

- V satirinin support'u gercek `V*` masked atomlardan gelmeli ve sifirdan buyuk
  olmali.
- N6, O2, O4, N4 ayri satirlarda support almalidir.
- `Base total > BaseR total` olmalidir.
- Nucleotide `total > real_total` veya Accuracy ile Real Accuracy'nin paydalari
  ayrismalidir.
- Bu ayrismalar yoksa virtual data/callback entegrasyonu gecersizdir; model sonucunu
  yorumlama.

Callback su an exception'lari sessizce yutabiliyor. Tablo eksikse "support yok"
diye varsayma; callback hata raporlama yolunu acip yeniden calistir.

## 10. Sonucu nasil yorumlayacaksin

Esas karsilastirma A ve B'nin son epochunda yapilir. Su metrikleri yan yana tabloya
koy:

- C1' recall ve support
- V recall ve gercek V support
- overall backbone recall/accuracy
- overall base ve BaseR accuracy
- nucleotide all-or-nothing Accuracy/Real Accuracy
- A, U, G, C bazli metrikler
- N6/O2/O4/N4 recall ve support
- train/val loss ve RNA domain loss

Per-position tabloda precision yapisal olarak yaklasik 1 olabilir; callback
element-dogruysa ilgili true position indexini kullanir. Bu nedenle ana kolon recall'dur.

### Basarili C1 stacking

Asagidakiler birlikte gorulurse C1 anchor yaklasimi calisiyor kabul edilebilir:

- B'de gercek V support vardir ve model V'yi anlamli duzeyde ogrenir (tercihen
  V recall >= 0.80).
- C1' recall A'ya gore en fazla yaklasik 0.05-0.10 mutlak duser ve tercihen 0.85
  uzerinde kalir.
- Backbone genel metrigi 0.05'ten fazla cokmez.
- BaseR ve nucleotide real-only metriklerinde yaygin/ciddi bozulma yoktur.
- N6/O2/O4/N4 artik V support'una karismaz.

Bu durumda RNA-style backbone stacking makul bir temsil secimidir. Sonraki deney
protein virtual atomlarini CB yerine CA/backbone anchor'a koyan esit kosullu A/B
olmalidir; dogrudan BoltzGen'e gecmek gerekmez.

### Anchor collapse / basarisiz

- C1' recall 0.50 altina iner veya A'ya gore 0.20'den fazla mutlak duser,
- V recall yuksek kalir,
- diger backbone atomlari C1' kadar bozulmazsa,

protein CB'dekiyle ayni tur anchor-confusion kaniti vardir. C1 stacking cozum
degildir; BoltzGen/pseudo-atom geometrisi degerlendirilmelidir.

### Gecersiz veya belirsiz deney

- V support 0: yanlis cache/config kullanildi, deney gecersiz.
- N6/O2/O4/N4 hala V'ye gidiyor: metric fix gecersiz.
- B'de `Base == BaseR`: virtuallar callback'e ulasmiyor, deney gecersiz.
- Tum RNA ve diger domain metrikleri birlikte cokuyor: anchor etkisinden once
  pipeline, edge veya training farkini ara.
- A ve B de underfit/dengesizse: epoch sayisini artir; anchor karari verme.
- 0.03 altindaki farklari tek seed ile guclu biyolojik sonuc gibi yorumlama.

C/U her tam residue'da 3, A'da 1, G'de 0 V beklendiginden anchor zarari varsa
C1' etkisi C/U'da daha guclu, G'de en dusuk olabilir. Mevcut aggregate C1' satiri
bunu gizler. Agent mumkunse C1' recall'u nucleotide bazinda da loglamali veya final
audit/metric snapshot'inda hesaplamalidir.

## 11. Tamamlanma kriterleri ve teslim

Gorev ancak su maddeler tamamlaninca biter:

- Metric vocabulary ve unknown-real davranisi duzeltildi, odakli testleri gecti.
- Source negative ve virtual positive cache auditleri gecti.
- Temp virtual PKL/PT ve metadata uretildi; production data degismedi.
- Esit kosullu A ve B fast runlari basariyla tamamlandi.
- Son epoch metrikleri yan yana raporlandi.
- Karar `C1 stacking basarili`, `anchor collapse` veya `belirsiz` olarak kanitlarla
  yazildi.
- Kullanilan komutlar, configler, log pathleri ve audit JSON pathi raporda verildi.
- `validate-feature` skillindeki ilgili smoke/log kontrolleri uygulandi.

Final raporda yalniz "training bitti" deme. Hangi `.pt`nin kullanildigini, gercek
V sayisini, N6/O2/O4/N4'un ayri sayildigini ve C1' A/B farkini rakamlarla goster.
