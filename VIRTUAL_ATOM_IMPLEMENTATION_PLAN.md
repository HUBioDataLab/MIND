# MIND Virtual Atom Tamamlama Planı

## Amaç

Protein ve RNA'da sabit slot temsilini korurken maskelenen block'un cevabını
geometriden sızdırmayan, gerçek atomlarla virtual atomları aynı koordinata
yığmayan ve distance görevlerini yapay mesafelerle kirletmeyen ortak bir runtime
template sistemi kurmak.

Bu plan iki temel kurala bağlıdır:

1. Mevcut production `.pt` dosyaları ilk deneylerde değiştirilmeyecek.
2. Bir aşamanın kabul kriterleri geçmeden sonraki aşamaya ve production cache
   üretimine geçilmeyecek.

## Mevcut Durum ve Veri Kaynakları

- Protein production cache: `/home/yusuf/data/proteins/processed_14atom/`
- Protein ham yapılar: `/home/yusuf/data/proteins/raw_structures_hq_40k/`
- RNA production cache: `/home/yusuf/data/rna/processed_graphs/`
- RNA ham CIF kökü: `/media/ubuntu/8TB/arya/rna_data/`
- RNA ham CIF sayısı (2026-08-02 audit): `12,933`
- Geçici C1'-stacked RNA cache:
  `/tmp/mind_rna_virtual_fast/processed_graphs_chunk_0/`
- Ana baseline log:
  `logs/June2026/multidomain_100ep_pdb_unimol_rna_20260609_065900.log`

Production RNA cache'inde virtual atom yoktur. Geçici RNA cache'inde 23-slot
deneyi için C1' üzerine yığılmış V atomları vardır. Protein production
cache'inde ise virtual atomlar CB/CA üzerindedir.

## Hedef Davranış

Maskelenen protein residue'sinde:

- `N/CA/C/O` backbone koordinatları local frame'i tanımlar.
- On sidechain slotunun `z` tipi gizlenir.
- On slotun input koordinatları, residue türünden bağımsız aynı `S1..S10`
  local-frame template'ine taşınır.
- Hedefler değişmez: gerçek element veya `V` tahmin edilir.

Maskelenen RNA nucleotide'inde:

- Audit sonucunda seçilecek ve accepted block'larda zorunlu olan üç backbone
  anchor'ı local frame'i tanımlar. İlk aday `C1'-C2'-C3'`tür; kapsam ve
  dejenerasyon auditinden geçmeden kesinleştirilmez.
- On bir base slotunun `z` tipi gizlenir.
- On bir slot aynı `S1..S11` template'ine taşınır.
- Hedefler gerçek element veya `V` olarak korunur.

Her iki domainde:

- Element id kontratı `11 common + 1 rare + 1 virtual`dır. Common elementler
  `H/C/N/O/F/Na/P/S/Cl/Br/I`, rare gerçek element `z=119`, virtual padding
  `z=120`, MLM input maskesi `z=0` kullanır.
- Virtual kararı yalnız `element='V'` metninden verilmez; bu sembol kimyasal
  vanadyum da olabilir. Yalnız `is_virtual=true` veya `pos_code=V/V<number>`
  olan adapter padding'i `120` olur; gerçek vanadyum rare `119`a gider.
- Legacy production cache'lerdeki `V*=119` değerleri `.pt` yeniden yazılmadan
  `pos_code` yardımıyla loader'da `120`ye taşınır; virtual olmayan rare
  `119` olarak kalır.
- Legacy RNA cache'lerde `BR` adıyla yanlış `z=5/119` taşıyan bromlar da
  loader'da `35`e alınır.
- Template yalnızca MLM için seçilen block'a uygulanır.
- `clean_pos` gerçek PDB/CIF koordinatlarını saklar.
- `input_pos` modelin gördüğü template koordinatlarını taşır.
- Template node'una temas eden pair/edge'ler ilk sürümde short/long distance
  loss'tan çıkarılır.
- Template koordinatlarından spatial edge'ler runtime'da yeniden kurulur.
- Unmasked protein/RNA block'ları gerçek atom geometrisini korur.

## Global Kabul ve Durdurma Kuralları

Bir aşama `GO` olmadan sonraki aşamaya geçilmez. `STOP` oluşursa yeni kod
genişletilmez; aynı küçük testte neden bulunur.

| Alan | GO | Uyarı | STOP |
|---|---:|---:|---:|
| Template NaN/Inf | `0` | - | `>0` |
| Masked-template sıfır mesafeli edge | `0` | - | `>0` |
| Protein geometry-only probe balanced accuracy | `<=0.10` | `0.10-0.15` | `>0.15` |
| RNA geometry-only probe balanced accuracy | `<=0.35` | `0.35-0.40` | `>0.40` |
| Protein N/CA/C/O recall | her biri `>=0.90` | baseline'dan `0.03-0.05` düşüş | `<0.90` veya `>0.05` düşüş |
| Protein V recall | `>=0.80` | `0.70-0.80` | `<0.70` |
| Protein CB iyileşmesi | baseline'a göre `>=+0.10` mutlak | `+0.05-0.10` | `<+0.05` |
| RNA C1' recall | `>=0.90` | `0.87-0.90` | `<0.87` |
| RNA backbone recall | `>=0.83` | `0.80-0.83` | `<0.80` |
| RNA V recall | `>=0.80` | `0.70-0.80` | `<0.70` |
| Train total loss | epoch 1→5 azalır | yavaş/oynak azalma | düz veya artan |
| Eligible distance loss | kontrolün en fazla `%10` üstü | `%10-20` | `>%20` veya NaN |
| Ortalama step-time | kontrolün en fazla `%20` üstü | `%20-35` | `>%35` |

Kısa runlarda support düşükse tek bir recall değeriyle karar verilmez.
Yukarıdaki metrikler ancak kontrol ve treatment aynı seed, split, sampler,
epoch, atom bütçesi ve GPU ile koştuğunda gate olarak kullanılır.

---

## Aşama 0 - Baseline ve Artifact Kontratı

### Amaç

Davranışı değiştirmeden mevcut protein/RNA cache'lerinin gerçek durumunu,
baseline metriklerini ve deney artifact yollarını dondurmak.

### Yapılacaklar

- [x] Dirty worktree ve ilgili dosyaların diff'i kaydedilecek; kullanıcı
  değişiklikleri korunacak.
- [x] Deney artifact kökü `/tmp/mind_virtual_template_v1/` olarak belirlenecek.
- [x] Protein chunk 0 ve RNA production chunk 0 için graph/atom/block/slot/V
  sayımları JSON olarak kaydedilecek.
- [x] Geçici C1'-stacked RNA `.pt` yeniden audit edilecek; dosya boyutu, mtime,
  graph/atom/V sayısı kaydedilecek.
- [x] Mevcut cache'lerde sıfır mesafeli edge sayısı, degree dağılımı ve
  distance-loss'a uygun edge/pair sayısı domain bazında raporlanacak.
- [x] Flag-off baseline config'i kopyalanacak; production YAML bu aşamada
  değiştirilmeyecek.
- [x] Baseline step-time en az 100 training step üzerinden ölçülecek.

### Beklenen

- Protein block'ları canonical durumda tam 14 atom olmalı.
- RNA production cache'inde `V*` support `0` olmalı.
- Legacy geçici RNA cache'inde `V* > 0` ve diskte `V*=119` pozitif kontrolü
  görülmeli; loader sonrasında bütün `V*` değerleri `120` olmalı.
- Virtual olmayan rare atomlar `119`, `Br` ise common sınıf `35` olmalı.
- Protein ve C1'-stacked RNA'da anchor stacking kaynaklı sıfır mesafeli
  edge'ler yeniden görülmeli; bu treatment öncesi pozitif kontroldür.

### STOP

- Hangi `.pt` dosyasının kullanıldığı kesin değilse.
- `len(pos)`, `len(z)`, `len(block_idx)` veya `len(pos_code)` uyuşmuyorsa.
- Protein accepted block'ları 14 slot değilse veya RNA accepted block'ları
  23 slot değilse.
- Eksik/tekrarlı backbone nedeniyle RNA target quarantine oranı `%2.5`i
  aşıyorsa.
- Kontrol runı deterministik mask/seed ile tekrar üretilemiyorsa.

### Gate artifact'i

`/tmp/mind_virtual_template_v1/stage0_baseline.json`

### Aşama 0 Sonucu (2026-08-02): GO

Ölçüm ve baseline maddelerinin tamamı bitti. İlk audit'te iki engel
bulundu; ikisi de production `.pt` dosyaları değiştirilmeden çözüldü:

- Protein: `556,486 / 556,486` block tam `14` slot.
- Production RNA: `V=0`, sıfır-mesafeli edge `0`.
- Geçici C1'-stacked RNA: `113,766 / 116,175` block tam `23` slot.
  Eksik/tekrarlı canonical backbone taşıyan `2,409` block (`%2.074`)
  graph'tan silinmeyecek; context olarak kalacak fakat RNA block-MLM/template
  target'ı olarak seçilmeyecek. Accepted block'ların tamamı tam 23 slottur
  ve quarantine oranı `%2.5` gate'inin altındadır.
- Ham CIF'te `_atom_site.type_symbol=B`, atom adı `BR` olan brom kayıtları
  adapter'da `BR` olarak normalize edildi. Gerçek `(B, B)` bor kayıtları
  `B` olarak korunur ve 11 common element dışında olduğu için rare
  `z=119`a gider.
- `11+1` vocabulary tekrar doğrulandı: `Br` on bir common elementten biridir;
  `B` değildir. Yanlışlıkla eklenen `B:5` whitelist genişlemesi geri
  alındı.
- Rare ve virtual ayrıldı: gerçek rare `119`, virtual `120`dır. Eski
  production/geçici cache'lerdeki `V*=119`, `.pt` değiştirilmeden loader'da
  `120`ye remap edilir.
- Gerçek legacy RNA graph smoke testinde diskteki cache'in mtime'ı aynı
  kaldı; RAM'de `145 / 145` virtual atom `120` olarak görüldü.
- Production RNA chunk 0'ın `1,957` graph'ı loader'dan geçirildi; 14
  graph'taki `27 / 27` `BR` atomu RAM'de `35` oldu ve `.pt` mtime'ı değişmedi.
- Production cache'leri kullanan mixed protein+RNA+UniMol eğitim smoke testi
  `180` adım çalıştı. `120` target için out-of-range/NaN oluşmadı; total
  loss Step 0 `8.3697`den Step 100 `3.0930`a, MLM loss `4.7918`den `1.0226`ya
  indi. Feature smoke amacıyla validation öncesinde kontrollü durduruldu.
- Eski/yeni geçici `.pt` karşılaştırmasında yalnızca `27` `BR`
  atomunun `z` değeri `119 -> 35` değişti; koordinat, edge, block ve slice
  tensorları aynı kaldı. Bu artifact virtual-id ayrımından önceki disk
  karşılaştırması olarak tarihsel kayıttır.
- Aynı seed ile MLM maskesi yeniden üretildi; yedi Stage 0 testi geçti.
- GPU 2 üzerinde `100` adım tamamlandı. İlk 10 warmup sonrası uçtan uca
  ortalama süre `0.3083 sn/adım`.

Gate artifact'i düzeltilmiş cache audit'i, exact `.pt` diff'i ve block
quarantine raporlarına bağlandı. Stage 1'e geçilebilir.

---

## Aşama 1 - Template Matematiği, Veri Olmadan Birim Testleri

### Amaç

Model, `.pt` ve adapter değiştirmeden ortak local-frame/template modülünü
doğrulamak.

### Yapılacaklar

- [x] Protein `N-CA-C` local frame fonksiyonu yazılacak.
- [x] RNA anchor adayları `/media/ubuntu/8TB/arya/rna_data/` üzerinde yalnız
  okuma auditiyle karşılaştırılacak.
- [x] P-only coarse-grained RNA molecule'leri atomik veri hattından ayıran
  filtre gerçek chunk 0 üzerinde production dosyalarına dokunmadan
  doğrulanacak.
- [x] Accepted RNA block politikası ve kesin local-frame anchor'ı audit
  sonucuna göre seçilecek.
- [x] Protein için 10, RNA için 11 adet tip-bağımsız local template noktası
  tek bir ortak modülde tanımlanacak.
- [x] Aynı seed/input için bitwise veya tanımlı tolerance içinde aynı output
  testi eklenecek.
- [x] Translation ve rotation equivariance testi eklenecek.
- [x] Collinear/eksik anchor durumunda silent fallback yerine açık quarantine
  nedeni üretilecek.
- [x] Template noktaları arası ve anchor-template minimum mesafesi test
  edilecek.

### Aşama 1.1 Sonucu (2026-08-02): GO

- Protein frame origin'i `CA`, `+x` ekseni `CA->C`, `+y` ekseni `CA->N`in
  `+x`e dik bileşeni ve `+z=x cross y` olarak sabitlendi.
- Aynı input iki çağrıda bitwise aynı origin/basis üretti.
- Orthonormal maksimum mutlak hata `3.33e-16`, determinant
  `0.9999999999999999`, local/global round-trip maksimum hata `4.44e-16`.
- Rotation equivariance basis maksimum hatası `6.66e-16`; translation origin
  hatası `0`.
- Eksik, çakışan, collinear ve NaN/Inf anchor durumları silent fallback
  yerine ayrı `LocalFrameError.reason` üretti.
- Beş ham protein dosyasındaki `859 / 859` residue frame'i kuruldu;
  quarantine nedeni sayısı `0`.
- Bu alt adım model, cache, edge, coordinate veya masking davranışını
  değiştirmedi.

Gate artifact'i:
`/tmp/mind_virtual_template_v1/stage1_protein_frame.json`

### Aşama 1.2 Sonucu (2026-08-02): STOP - RNA quarantine gate

Bu alt adım yalnız okuma auditi yaptı; adapter, model, masking, edge ve
production `.pt` dosyaları değiştirilmedi.

- Ham CIF parser yolu post-fix smoke testinde `2` dosya / `150` block'u
  sorunsuz okudu. Tam karşılaştırma, mevcut RNA cache'lerinin gerçek girdisi
  olan beş `universal_rna_chunk_*.pkl` kaynak akışının tamamında yapıldı:
  `9,783` molecule ve `874,604` nucleotide block.
- Exact RNA backbone sözleşmesini `846,528` block geçti (`%96.7899`),
  `28,076` block quarantine oldu (`%3.2101`). Bu oran Stage 1 için önceden
  koyulan `%2.5` STOP sınırını aştığı için anchor seçimi yapılmadı ve ortak
  template noktalarına geçilmedi.
- Quarantine oranı kaynaklara göre homojen değil: chunk 0 `%2.0736`, chunk 1
  `%6.7956`, chunk 2 `%0.8196`, chunk 3 `%4.9702`, chunk 4 `%1.0358`.
- En yaygın ihlal `22,398` block'ta (`quarantine'in %79.78'i`) fosfatın `P`
  atomu dışında bütün canonical sugar/phosphate pozisyonlarının eksik
  olmasıdır. İkinci yaygın ihlal `3,386` block'ta yalnız `P/OP1/OP2`
  eksikliğidir; üçüncüsü `861` block'ta yalnız `O2'` eksikliğidir. İlk üç
  pattern quarantine'in `%94.90`ını açıklar.
- Audit'in ilk sürümünde position map, production kabul kontrolündeki
  `O1P -> OP1`, `O2P -> OP2` alias normalizasyonunu uygulamıyordu. Audit
  üretim sözleşmesiyle eşitlendi; böylece kabul edilmiş block'lardaki sahte
  `69` fosfat-anchor eksiği kayboldu.
- Kabul edilen `846,528` block'un tamamında `C1'-C2'-C3'`,
  `O4'-C1'-C2'`, `C2'-C3'-C4'` ve diğer bütün karşılaştırılan adaylar
  eksiksiz, sonlu ve non-collinear çerçeve verdi (`%100`).
- Sayısal koşullulukta `C2'-C3'-C4'` en güçlü adaydır: minimum dik yükseklik
  `1.1435 Å`, minimum açı sinüsü `0.8834`. `C1'-C2'-C3'` de güçlüdür:
  `0.8934 Å` ve `0.8690`. Bu karşılaştırma yalnız aday listesini daraltır;
  quarantine gate çözülmeden kesin anchor kararı değildir.

Sonraki alt adım, özellikle chunk 1 ve 3'te yoğunlaşan üç ana ihlal
pattern'ini molecule/CIF ve residue kaynağına kadar izleyip parser ile target
quarantine politikasının doğru olup olmadığını belirlemektir.

Gate artifact'i:
`/tmp/mind_virtual_template_v1/stage1_rna_anchor_candidates.json`

### Aşama 1.3 Sonucu (2026-08-02): Kök neden bulundu, filtre implementasyonu bekliyor

Stage 1.2'deki `%3.2101` değeri tek tip bir parser/anchor hatası değildir.
Beş source-PKL akışındaki `874,604` block molecule ve ham CIF kökenine kadar
izlendi:

- `22,398` block gerçekten yalnız bir `P` atomundan oluşur. Bunlar parser'ın
  atom kaybetmesi değildir; örneğin `3cw1_V.cif` kaynak dosyası da residue
  başına yalnız `P` koordinatı içerir. Bu coarse-grained block'lar `55`
  molecule'de toplanır; `53` molecule tamamen P-only, iki düşük çözünürlüklü
  molecule ise atomik ve P-only bölgeleri aynı graph'ta karıştırır.
- `3,235` block'ta yalnız `P/OP1/OP2` eksiktir ve block index `0`dır. Bu,
  zincirin doğal 5-prime terminal residue'siyle uyumludur; parser bug'ı olarak
  değerlendirilmez. Başka indexlerde aynı pattern yalnız `151` block'tur.
- `861` block'ta yalnız `O2'` eksiktir. Dokuz molecule'ün bütün block'ları bu
  durumdadır. Ham örneklerde `T` içeren fakat metadata'da polyribonucleotide
  yazan DNA-benzeri/yanlış etiketli zincirler de vardır. Conservative exact
  backbone target politikası bunları otomatik quarantine eder.
- Diğer eksik backbone pattern'leri `1,428`, duplicate backbone ise yalnız
  `3` block'tur.

Önerilen yapı-seviyesi veri kalite politikası: içinde tek bir P-only nucleotide
bulunan molecule'ü atom-seviyeli RNA pretraining girdisinden tamamen dışla.
Eksik fosfat/O2' veya başka atom taşıyan kalan molecule'leri graph context'i
olarak koru; fakat yalnız exact 12-backbone block'larını 23-slot template/MLM
target yap. Bu, eksik gerçek koordinat uydurmaz ve mevcut uniform target
sözleşmesini gevşetmez.

Bu politikanın yalnız okuma simülasyonu:

- Dışlanan: `55 / 9,783` molecule, `22,407 / 874,604` block.
- Kalan: `852,197` block.
- Exact-backbone quarantine: `5,678 / 852,197 = %0.6663` (`%2.5` gate altı).
- `C1'-C2'-C3'` frame-ineligible: `676 / 852,197 = %0.0793`.
- Production PKL/CIF/PT dosyası değiştirilmedi; bu henüz aktif filtre değil.

Sonraki kontrollü alt adımda bu yapı-seviyesi predicate flag-off varsayılanla
uygulanacak ve source stream üzerinde `55` reject / `9,728` retain beklentisi
birim ve smoke testleriyle doğrulanacak. Test geçmeden `.pt` üretimi, kesin RNA
anchor seçimi veya template entegrasyonu yapılmayacak.

Gate artifact'i:
`/tmp/mind_virtual_template_v1/stage1_rna_quarantine_provenance.json`

### Aşama 1.4 Sonucu (2026-08-02): GO - P-only production filtresi dry-run

Onaylanan yapı-seviyesi filtre, ham CIF veya source PKL'yi silmeden
`UniversalMolecule -> PyG .pt` dönüşüm yoluna flag olarak eklendi:

- Predicate virtual padding'i yok sayar. Hem tarihsel `[P]` hem gelecekteki
  `[P, V1, ..., V11]` biçimi P-only kabul edilir; `P+OP1`, gerçek vanadyum
  veya normal 5-prime terminal residue reject edilmez.
- Flag kapalıyken önceki conversion davranışı korunur. Flag açıkken bir
  molecule'de tek P-only block bulunması molecule'ün tamamını reject eder.
- Filtreli cache adına `_rna_no_p_only` imzası eklenir. Böylece eski filtresiz
  `.pt`, filtreli cache sanılarak sessizce tekrar kullanılamaz.
- Her filtreli conversion, rejected molecule ID, block sayısı ve P-only block
  sayısını içeren JSON sidecar manifest yazar. Source PKL değiştirilmez.
- Flag hem `cache_to_pyg.py` CLI'ına hem chunk orchestration CLI/config geçişine
  bağlandı; varsayılan halen `false`dır.

Sekiz Stage 1 assertion geçti. `/tmp` içindeki iki moleküllü gerçek conversion
smoke testinde P-only molecule reject, atomik molecule retain edildi; filtreli
`.pt` ve tek kayıtlı reject manifest üretildi. Beş gerçek source-PKL dry-run
sonucu önceki araştırmayla birebir eşleşti:

- Flag-off sözleşmesi: `9,783 / 9,783` molecule retain.
- Flag-on: `55` reject, `9,728` retain.
- Dışlanan block: `22,407`; kalan block: `852,197`.
- Exact-backbone quarantine: `5,678 / 852,197 = %0.6663`.
- Sugar-frame failure: `676 / 852,197 = %0.0793`.
- Stage 0 atom-id/masking assertion'ları da regresyon vermedi.

Production `.pt` dosyaları bu alt adımda yeniden üretilmedi ve eğitim girdisi
henüz değişmedi. Sonraki alt adımda yalnız chunk 0, `/tmp` altında filtreli
PyG cache'e dönüştürülecek: `1,957` girdiden `11` reject ve `1,946` retained
graph beklenir. Production chunk 0 mtime/checksum'ı değişirse veya sayılar
saparsa durulur. Tüm production cache yenilemesi, virtual geometri değişikliği
tamamlandıktan sonra tek sefer yapılır.

Gate artifact'i:
`/tmp/mind_virtual_template_v1/stage1_rna_p_only_filter_dry_run.json`

### Aşama 1.5 Sonucu (2026-08-02): GO - Gerçek RNA chunk 0 filtreli dönüşüm

Sentetik smoke testten sonra filtre, production ile aynı `12.0 Angstrom`
cutoff ve `48` maksimum komşu ayarlarıyla gerçek
`universal_rna_chunk_0.pkl` üzerinde çalıştırıldı. Çıktı yalnızca
`/tmp/mind_virtual_template_v1/processed_graphs_chunk_0/` altına yazıldı.

- Girdi `1,957`, P-only nedeniyle reject `11`, retained graph `1,946` oldu;
  önceden hesaplanan beklenti birebir tuttu.
- Cache adı `_rna_no_p_only` imzası taşıyor. Reject manifest tam `11`
  molecule kimliğini, block ve P-only block sayılarıyla kaydetti.
- Filtreli `.pt` gerçekten yüklenerek bütün `1,946` graph tarandı:
  benzersiz molecule kimliği `1,946`, reject-ID overlap `0`, kalan P-only
  block `0`, shape/NaN/Inf/geçersiz-edge hatası `0`.
- Source PKL'nin SHA-256 değeri dönüşümden önce ve sonra
  `0b7fdbb888d105c2cbdbbae92afc783f28ae77f31f81a52a8d7f3e06c37e0ffa`
  olarak kaldı; size, inode ve mtime da değişmedi.
- Production chunk 0 `.pt` SHA-256 değeri önce ve sonra
  `584314a7969070ca2aac38f0bade32c34bcf381f91b12a9ba3912d0afce458a2`
  olarak kaldı; size, inode ve mtime da değişmedi.
- Filtreli geçici `.pt` SHA-256 değeri
  `3115ba2ca2ddaf1b1d001f98c7a7d12850baf075788a81a925edd5ef793ff154`
  oldu.

Multiprocessing'in tensor aktarımı için açtığı yerel socket ilk sandbox
denemesinde izin hatası verdi; bu filtre/veri hatası değildi. Aynı komut
gerekli çalıştırma izniyle tamamlandı ve yukarıdaki içerik kontrollerinden
geçti.

Tüm production RNA cache'leri bu noktada yenilenmeyecek. P-only filtresi,
kesin RNA frame/template geometrisi ve virtual slotlar tamamlandıktan sonra
tek production rebuild'de birlikte uygulanacak.

Gate artifact'leri:

- `/tmp/mind_virtual_template_v1/processed_graphs_chunk_0/processed/optimized_universal_rna_chunk_0_rna_no_p_only_cutoff_12.0_neighbors_48.pt`
- `/tmp/mind_virtual_template_v1/processed_graphs_chunk_0/processed/optimized_universal_rna_chunk_0_rna_no_p_only_cutoff_12.0_neighbors_48_rejects.json`

### Aşama 1.6 Sonucu (2026-08-02): STOP - Geometri kazananı bulundu, L-RNA politikası gerekli

Beş source PKL, P-only molecule'ler bellekte dışlandıktan sonra salt-okunur
tarandı. `C2'-C3'-C4'`, `C1'-C2'-C3'` ve `O4'-C1'-C2'` adayları accepted
exact-backbone block'ların tamamında normal ve ters atom listesiyle
kuruldu.

- Girdi `9,783` molecule / `874,604` block; P-only reject `55` molecule /
  `22,407` block; kalan `9,728` molecule / `852,197` block oldu.
- Exact-backbone accepted `846,519`, quarantine `5,678` ve quarantine oranı
  `%0.6663` oldu.
- Üç adayın accepted coverage'ı `%100`; atom listesini ters çevirince
  origin ve basis maksimum farkı `0`; sağ-elli olmayan frame `0`;
  determinant aralığı yaklaşık `1 +/- 9e-16` oldu.
- Worst-case geometri kazananı provisional `C2'-C3'-C4'`tür: minimum dik
  yükseklik `1.1435 Angstrom`, minimum `sin(angle)=0.8834`. `C1'-C2'-C3'`
  için bunlar `0.8934 / 0.8690`, `O4'-C1'-C2'` için
  `1.1125 / 0.5387`dir. Dolayısıyla ilk aday iki worst-case ölçüyü de
  tek başına kazanır.
- Buna karşın provisional frame'de stereokimya referansı `O2'`,
  `846,360` block'ta pozitif, `159` block'ta negatiftir; near-plane `0`dır.
  Bu atom sırası kaynaklı frame flip değildir: negatif block'lar da ters
  atom listesinde bitwise aynı frame'i verir.
- Negatif block'ların `91`i yalnız üç tamamen ters-kiralli molecule'de
  toplanır: `4r8i_B=33/33`, `4wb3_D=29/29`, `4wb3_E=29/29`.
  Resmi yapı tanımları `4R8I`i mirror-image RNA aptamer, `4WB3`ü
  mirror-image L-RNA/L-DNA aptamer olarak tanımlar. Bu nedenle bunlar parser
  veya frame hatası değil, gerçek yapay L-nükleik asitlerdir.
- Kalan `68` negatif block, `37` başka molecule'de lokal aykırılık olarak
  dağılır; bu molecule'lerin tamamını silmek gereksiz veri kaybı olur.
- Source PKL size/mtime/inode fingerprint'leri audit öncesi ve sonrası aynı
  kaldı. Production CIF/PKL/PT, adapter, cache ve training davranışı
  değiştirilmedi.

Planın mirror/yön belirsizliği STOP kuralı nedeniyle provisional geometri
kazananı henüz kesin production anchor sözleşmesi yapılmadı. Sonraki
salt-okunur politika simülasyonu şunu sınayacak: exact block'larının
yarıdan fazlası noncanonical `O2'` yönü taşıyan bu üç L-RNA molecule'ü
tamamen dışla; kalan molecule'lerdeki `68` lokal aykırı block'u graph
context olarak koru fakat template/MLM target yapma. Beklenen simülasyon
sonucu `9,725` molecule, `852,098` block, `846,360` accepted target ve
`5,738 / 852,098 = %0.6734` quarantine'dir. Ardından üç aday yeniden
karşılaştırılmadan kesin anchor seçilmeyecek.

Gate artifact'i:
`/tmp/mind_virtual_template_v1/stage1_rna_anchor_selection.json`

### Aşama 1.7 Sonucu (2026-08-04): GO - D/L politikası ve RNA anchor kesinleşti

Aşama 1.6'da bulunan gerçek mirror L-RNA'lar için önerilen politika beş
source PKL'nin tamamında salt-okunur simüle edildi:

1. Herhangi bir P-only block içeren molecule tamamen dışlanır.
2. Exact-backbone block'larının `%50`den fazlası provisional
   `C2'-C3'-C4'` frame'inde nonpositive `O2'` signed-z taşıyorsa molecule
   mirror/L-RNA kabul edilip canonical D-RNA domain'inden tamamen dışlanır.
3. Molecule mirror değilse tekil noncanonical-yönlü block graph context olarak
   korunur fakat template/MLM target eligibility'den quarantine edilir.

Simülasyon sonucu beklentiyle birebir eşleşti:

- Girdi `9,783` molecule / `874,604` block.
- P-only reject `55` molecule / `22,407` block.
- Mirror reject yalnız `3` molecule / `99` block:
  `4r8i_B`, `4wb3_D`, `4wb3_E`. Bunlarda `91` exact target vardı.
- Kalan `9,725` molecule / `852,098` block.
- Backbone-contract quarantine `5,670`, lokal chirality quarantine `68`;
  toplam `5,738 / 852,098 = %0.6734`.
- Accepted canonical D-RNA target `846,360`.

Üç anchor adayı temiz target kümesinde sıfırdan tekrar hesaplandı.
Kesin seçim `C2'-C3'-C4'`tür; roller
`plane_anchor=C2'`, `origin=C3'`, `positive_x_anchor=C4'` olarak sabittir.
Bu adayda:

- accepted coverage `%100`;
- noncanonical `O2'` yönü `0 / 846,360`;
- atom-listesi sırası farkı `0`;
- sağ-elli olmayan frame `0`;
- minimum dik yükseklik `1.1435 Angstrom`;
- minimum `sin(angle)=0.8834`.

`O2'` frame anchor'ı değildir. Seçilen üç anchor'ın tanımladığı
`+z` eksenine göre RNA şeker el-yönünü doğrulayan bağımsız
stereokimya kalite kontrolü/reference atomudur.

Bütün stop condition'ları false, source PKL fingerprint'leri audit öncesi
ve sonrası aynıdır. Bu adım production filtre/config/cache, adapter, template
noktaları veya training davranışını değiştirmedi. Sonraki adımda kesin
RNA frame sözleşmesi `core/local_frame.py` içinde Torch implementasyonuna
alınacak; template noktaları ayrı gate olarak daha sonra eklenecek.

Gate artifact'i:
`/tmp/mind_virtual_template_v1/stage1_rna_chirality_policy_simulation.json`

### Aşama 1.8 Sonucu (2026-08-04): GO - Kesin RNA Torch local frame

Audit ile seçilen RNA frame, `core/local_frame.py` içinde
`rna_c2_c3_c4_frame(pos_codes, positions)` Torch fonksiyonu olarak uygulandı.
Runtime PyG temsilindeki paralel `pos_code` ve `[N,3]` coordinate tensorunu
doğrudan kabul eder; anchor'ı atom geliş sırasıyla değil position code ile
bulur. Kesin sözleşme:

- plane anchor `C2'`;
- origin `C3'`;
- `+x`, `C3' -> C4'`;
- `+y`, `C3' -> C2'` vektörünün `+x`e dik bileşeni;
- `+z = x cross y`.

`O2'` fonksiyonun anchor'ı değildir; D/L target eligibility için bağımsız
chirality kalite kontrolü olarak kalır. Fonksiyon missing, duplicate,
invalid-table-shape, coincident, collinear ve nonfinite durumlarda silent
fallback yerine ayrı `LocalFrameError.reason` üretir.

Doğrulama sonuçları:

- Protein regresyonları dahil bütün Stage 1 doğrudan testleri `15 / 15` geçti.
- Determinism, sağ-elli determinant, local/global round-trip,
  translation/rotation equivariance ve atom-sırası bağımsızlığı geçti.
- Beş source PKL'nin her birinden `100`, toplam `500` gerçek canonical block
  NumPy selection auditi ile karşılaştırıldı.
- Float64 origin maksimum farkı `0`, basis maksimum farkı `2.22e-16`.
- Production-benzeri float32 basis maksimum farkı `1.38e-7`
  (`1e-5` tolerance altı).
- Ters atom listesinde origin/basis farkı tam `0`.
- Maksimum ortonormallik hatası ve determinantın `1`den farkı `6.66e-16`.
- Source PKL fingerprint'leri smoke sonrası aynı kaldı.

Bu adım template/V noktaları, adapter, masking, edge, production filter/cache
ve training davranışını değiştirmedi.

Gate artifact'i:
`/tmp/mind_virtual_template_v1/stage1_rna_torch_frame.json`

### Aşama 1.9 Sonucu (2026-08-04): GO - Tip-bağımsız 10/11 nokta şablonları

`core/virtual_atom_templates.py` yalnız sabit local koordinat matematiği
taşıyan ortak modül olarak eklendi. Protein fonksiyonu her residue için aynı
`[10,3]`, RNA fonksiyonu her nucleotide için aynı `[11,3]` tensoru döndürür.
API residue/nucleotide türü, gerçek atom sayısı, target veya random seed kabul
etmez; yalnız tensor `dtype/device` seçilebilir. Her çağrı ayrı tensor sahibi
olduğu için bir caller'ın mutasyonu sonraki block'ların şablonunu değiştiremez.

İlk RNA adayı küçük örnekte güvenli görünmesine rağmen beş source PKL'nin tam
auditinde haklı olarak STOP verdi: `4wzd_2K` molecule, block `60` içindeki
gerçek `OP2` ile template slot `6` arası `0.6717 Angstrom`du ve `0.75`
güvenlik sınırının altındaydı. Bu parser/frame hatası değil, nadir phosphate
konformasyonuna çarpan ilk geometrik adaydı. RNA bulutu tür bilgisi eklenmeden
aynı merkezde local-z etrafında `180` derece döndürüldü ve yarıçapı
`1.3 -> 1.0 Angstrom` yapıldı. Tam accepted RNA backbone matrisi üzerinde
aday karşılaştırmasından sonra seçilen sürümün minimum backbone mesafesi
`1.3263 Angstrom` oldu.

Nihai doğrulama:

- Stage 1 doğrudan testleri `19 / 19` geçti.
- Protein/RNA shape sırasıyla tam `[10,3]` ve `[11,3]`; finite ve unique.
- Tekrarlı float32/float64 çağrılar bitwise aynı; type/count input'u yok.
- Nokta-nokta minimumu protein `1.1232`, RNA `0.9309 Angstrom`.
- Anchor-template minimumu protein `1.9637`, RNA `2.9066 Angstrom`.
- Protein chunk 0: `4,000` molecule, `556,486 / 556,486` accepted block;
  frame failure, `<0.75 Angstrom` collision ve bağlantısız block `0`.
- Beş RNA source PKL: `9,783` molecule / `874,604` block; onaylanan P-only,
  mirror ve lokal chirality politikalarından sonra `846,360` accepted target.
  Frame failure, `<0.75 Angstrom` collision ve backbone'a hiç 5-Angstrom
  spatial bağlantısı olmayan block `0`.
- RNA'daki `846,360 x 11 = 9,309,960` nokta örneğinin yalnız `5`inde o noktanın
  backbone'a doğrudan 5-Angstrom edge'i yoktur; bu bir disconnected block
  değildir. Şablonun bütün noktaları kendi aralarında en fazla `2 Angstrom`
  olduğu ve her block'ta başka template-backbone edge bulunduğu için graph
  bağlı kalır. Bu sayaç bilgi amaçlı korunmuştur.
- Rijit translation/rotation equivariance `atol=1e-12` ile geçti.
- Bütün source PKL size/mtime/inode fingerprint'leri audit öncesi ve sonrası
  aynı kaldı.
- Adapter, masking, edge, distance loss, production CIF/PKL/PT ve training
  davranışı bu adımda değiştirilmedi.

Gate artifact'i:
`/tmp/mind_virtual_template_v1/stage1_template_audit.json`

### Beklenen

- Template koordinatlarında NaN/Inf `0`.
- Aynı local template bütün AA/NT türlerinde aynı normalize geometrisini
  vermeli.
- Template noktaları birbirinden ve anchor'lardan en az `0.75 Å` ayrı olmalı.
- Rijit döndürülmüş/taşınmış girdinin template'i de aynı dönüşümü
  izlemeli (`atol <= 1e-5`).

### STOP

- Herhangi bir residue/nucleotide türü farklı template geometrisi alıyorsa.
- Anchor sırası atom dosyasının geliş sırasına bağlıysa; atomlar
  `pos_code` ile bulunmalı.
- RNA accepted block'ları için seçilen anchor coverage `%100` değilse.
- Quarantine oranı `%2.5`i aşarsa; modified residue ve parser politikası
  yeniden incelenir.

### Gate artifact'i

`/tmp/mind_virtual_template_v1/stage1_template_audit.json`

---

## Aşama 2 - Leakage Probe

### Amaç

Template geometrisinin tek başına AA/NT cevabını taşımadığını training
pipeline'ina girmeden kanıtlamak.

### Yapılacaklar

- [x] Bütün protein türleri için yalnız normalize template koordinatından
  20-class basit probe dataset'i kurulacak.
- [x] A/C/G/U için yalnız normalize template koordinatından 4-class probe
  dataset'i kurulacak.
- [x] Slot sırası, degree, template edge uzaklıkları ve anchor-relative
  koordinatlar probe'a ayrı ayrı verilerek sızıntı aranacak.
- [x] Train/validation ayrılığı molecule/structure bazlı yapılacak; aynı
  yapı iki tarafa düşmeyecek.

### Beklenen

- Protein balanced accuracy chance `%5`e yakın ve `<=%10`.
- RNA balanced accuracy chance `%25`e yakın ve `<=%35`.
- Sabit slot/internal-edge bilgisinin eklenmesi accuracy'yi chance üstüne
  çıkarmamalı.
- Gerçek anchor/backbone içeren özellikler mutlak chance ile değil, aynı gerçek
  geometri baseline'ına göre kontrol edilmeli; template eklenmesi balanced
  accuracy'yi en fazla `0.03` artırmalı.

### STOP

- Protein probe `>%15` veya RNA probe `>%40` ise.
- AA/NT türüne göre template noktası, edge sayısı veya degree değişiyorsa.
- Gerçek-context kontrollü template artışı `>0.05` ise.

### Gate artifact'i

`/tmp/mind_virtual_template_v1/stage2_leakage_probe.json`

### Aşama 2 Sonucu (2026-08-04): GO - Sabit template kimlik sızdırmıyor

`data_loading/audit_virtual_atom_leakage.py` salt-okunur Stage 2 probe'u
olarak eklendi. Molecule kimliği `seed=20260804` ile SHA-256 üzerinden önce
train/validation'a atanır; block örnekleme bundan sonra yapılır. Böylece aynı
molecule'ün residue/nucleotide'leri iki split'e düşemez. Protein için sınıf
başına train/validation `2,000/500` (`40,000/10,000` toplam), RNA için
`5,000/1,250` (`20,000/5,000` toplam) dengeli örnek kullanıldı. Her iki
domainde train/validation molecule overlap `0`dır.

İlk probe tanımı gerçek üç frame anchor'ını da mutlak template gate'ine
koyduğu için protein tarafında `0.2286` ile STOP verdi. Kök neden sabit
template değildi:

- Protein sabit template XYZ, slot sırası ve internal graph özelliklerinin
  her biri tam `0.05` chance verdi.
- RNA'da aynı üç sabit özellik tam `0.25` chance verdi.
- Fixed-template feature hash'i protein bütün 20, RNA bütün 4 sınıfta aynıydı.
- Gerçek protein frame anchor'ları tek başına `0.2159-0.2298`, bütün görünür
  protein backbone'u `0.2546-0.2647` balanced accuracy taşıdı.
- Gerçek RNA anchor'ları tek başına `0.2864-0.2938`, bütün görünür RNA
  backbone'u `0.3476-0.3646` taşıdı.

Bu, target-dependent virtual geometri sızıntısı değil; gerçek backbone
konformasyonunun block kimliğiyle doğal korelasyonudur. Probe kontrollü
karşılaştırma olarak düzeltildi: yalnız sabit template/slot/internal graph
özellikleri mutlak chance gate'ine, gerçek anchor/backbone içeren özellikler
ise aynı gerçek-geometri baseline'ına göre artış gate'ine girer. Veri, seed,
split ve örnekleme değiştirilmeden nihai koşu tekrarlandı.

Nihai sonuçlar:

- Protein fixed-template maksimum balanced accuracy `0.0500`
  (`GO <= 0.10`, `STOP > 0.15`).
- RNA fixed-template maksimum balanced accuracy `0.2500`
  (`GO <= 0.35`, `STOP > 0.40`).
- Protein anchor-relative kontrollü fark: lineer `-0.0008`, shallow-tree
  `+0.0054`.
- Protein template-graph / visible-backbone farkı: lineer `+0.0169`,
  shallow-tree `-0.0144`.
- RNA anchor-relative kontrollü fark: lineer `-0.0014`, shallow-tree
  `+0.0120`.
- RNA template-graph / visible-backbone farkı: lineer `+0.0012`, shallow-tree
  `-0.0122`.
- Maksimum pozitif kontrollü farklar `%3` GO sınırının altında: protein
  `0.0169`, RNA `0.0120`.
- Protein `556,486 / 556,486`; RNA kabul politikası sonrası
  `846,360 / 846,360` canonical block probe koleksiyonuna uygundu.
- Stage 1+2 doğrudan testleri `24 / 24`, `git diff --check` geçti.
- Bütün source PKL size/mtime/inode fingerprint'leri önce/sonra aynı kaldı.
- Adapter, masking, edge, distance loss, production CIF/PKL/PT, config ve
  training davranışı değiştirilmedi. Bu nedenle fast training bu matematik/
  read-only probe adımında çalıştırılmadı.

Gate artifact'i:
`/tmp/mind_virtual_template_v1/stage2_leakage_probe.json`

---

## Aşama 3 - Runtime Entegrasyon, Flag-Off No-Op

### Amaç

Mevcut `.pt` dosyalarını değiştirmeden template'i batch-time masking yoluna
eklemek ve kapalıyken mevcut davranışı bit-identical tutmak.

### Yapılacaklar

- [x] Config'e default `false` bir `masked_slot_template.enabled` flag'i
  eklenecek.
- [x] Domain bazlı `protein` ve `rna` enable flag'leri eklenecek.
- [x] Protein block mask seçimi tek kez yapılacak; template ve MLM aynı mask'i
  kullanacak.
- [x] RNA için proteinle aynı semantikte deterministik validation destekli
  nucleotide block-MLM yolu eklenecek.
- [x] Runtime slot indeksleri `block_idx + pos_code` ile türetilecek; bu aşamada
  cache'e yeni tensor yazılmayacak.
- [x] Orijinal koordinatlar `clean_pos`, model girdisi `input_pos/batch.pos`
  olarak ayrılacak.
- [x] Yalnız masked block'un side/base slotları template'e taşınacak.
- [x] Template uygulanan graph'larda hybrid edge'ler runtime'da yeniden
  kurulacak; diğer domain graph'larına dokunulmayacak.
- [x] Template node'una temas eden short-range edge'ler loss mask'inden
  çıkarılacak.
- [x] Template node'u içeren long-range pair'ler sampler'dan çıkarılacak.
- [x] Her batch için `templated_blocks`, `templated_nodes`, `excluded_short_edges`,
  `excluded_long_pairs`, `edge_rebuild_ms` loglanacak.

### Beklenen

- Flag `false`: aynı seed ile `mlm_mask`, `pos`, `edge_index`, tüm loss'lar ve
  ilk forward logits baseline ile aynı olmalı.
- Flag `true`: yalnız seçilmiş protein/RNA block'larının side/base
  koordinatları ve ilgili edge'leri değişmeli.
- Gerçek hedefler `original_types` içinde değişmeden kalmalı.
- Distance loss'a giren hiçbir pair template node'una temas etmemeli.

### STOP

- Flag kapalıyken herhangi bir davranış değişirse.
- Mask seçimi template ve MLM arasında farklıysa.
- Non-protein/non-RNA graph edge'leri değişirse.
- Template koordinatları distance targetı olarak kullanılırsa.
- Validation aynı seed/batch için farklı mask veya edge üretirse.

### Stage 3.1 sonucu (2026-08-04): GO

- `masked_slot_template.enabled/protein/rna` config kontratı eklendi; eksik
  config ile üç alanı açıkça `false` vermek aynı normalize edilmiş sonucu
  üretiyor.
- Stage 3.1'de henüz bağlanmamış bir treatment koşusunun yanlışlıkla
  başlatılmaması için `enabled=true` fail-closed olarak reddediliyor.
- Sentetik karma protein+RNA batch'inde training ve deterministic validation
  yolları için `z`, `original_types`, `masked_types`, `mlm_mask`, `pos`,
  `edge_index`, `edge_attr`, block/batch üyeliği ve RNG state bitwise aynı
  kaldı.
- Aynı sentetik kontrolde MLM, short-range distance ve stochastic long-range
  distance loss'ları bitwise aynı kaldı; template/clean-pos/loss-exclusion
  alanları flag kapalıyken oluşmadı.
- Gerçek mini batch kontrolü protein `A0A6D0ENQ7` (`MET`, `LYS`; 28 node) ve
  RNA `4rum_A` (`G`; 23 node) ile 774 directed edge üzerinde hem training hem
  validation için geçti.
- Stage 0-3 doğrudan regresyon testleri `37 / 37`; `py_compile` ve
  `git diff --check` geçti. Kaynak protein/RNA PKL size/mtime/inode
  fingerprint'leri değişmedi; CIF/PKL/PT yeniden yazılmadı ve GPU training
  başlatılmadı.

### Stage 3.2 sonucu (2026-08-04): GO

- Protein residue ve RNA nucleotide seçimi `BlockMaskSelection` adlı tek bir
  sonuçta hem seçilmiş block id'lerini hem node maskesini taşıyor; aynı block
  için ikinci bir random draw yapılmıyor.
- Mevcut protein block-MLM aynı RNG sırası ve sonucu koruyarak ortak seçiciye
  geçirildi. RNA için `rna_block_mlm` ve `rna_block_mlm_ratio` eklendi;
  production fast/multidomain config'lerinde flag açıkça `false` bırakıldı.
- Kontrollü sentetik üç-domain batch'inde protein maskeleri block başına
  `0/14`, RNA maskeleri `0/23`; UniMol maskesi eski atom-level davranışta
  kaldı. Training aynı seed ile, validation aynı batch/epoch ile tekrarlandı.
- Gerçek mini batch'te protein `A0A6D0ENQ7` (`MET`, `LYS`) içinden `14` node,
  RNA `4rum_A/G` içinden tam `23` node birlikte maskelendi. Koordinatlar ve
  edge tensorleri değişmedi.
- Flag-off gerçek/sentetik training ve validation sonuçları eski yol ile
  bitwise aynı; kaynak PKL fingerprint'leri değişmedi. Stage 0-3 testleri
  `41 / 41`, `py_compile` ve `git diff --check` geçti.
- Bu ortamda CUDA görünmediği (`cuda_available=false`) için fast GPU training
  çalıştırılmadı. Bu seçim-only aşamasında production RNA flag'i kapalıdır;
  öğrenilebilirlik training'i koordinat entegrasyonu sonrası kontrollü A/B
  aşamasında yapılacaktır.

### Stage 3.3 sonucu (2026-08-04): GO

- `map_masked_template_slots` seçilmiş ve tamamen MLM-maskeli block'ları
  `batch + block_idx + pos_code` ile graph-local eşliyor; batch'i değiştirmeden
  protein için tam `10`, RNA için tam `11` template node indeksi döndürüyor.
- Backbone yalnız MLM hedefinde kalıyor; template node maskesine protein için
  `N/CA/C/O`, RNA için 12 phosphate/sugar backbone atomu girmiyor.
- Eksik/tekrarlı backbone, yanlış slot sayısı, yanlış `V<n>` numarası,
  virtual padding'den sonra gelen gerçek atom, desteklenmeyen/tekrarlı gerçek
  pozisyon, parçalı MLM block'u ve hizasız `pos_code` metadata'sı açık reason
  code ile reddediliyor.
- Gerçek RNA `4rum_A/G` eski cache node sırasının bugünkü adapter priority
  sırasından farklı olduğu görüldü. Sağlam block'u atmak veya `.pt`yi yeniden
  yazmak yerine gerçek atomlar `pos_code` üzerinden yalnız döndürülen runtime
  görünümünde canonical sıralanıyor; batch node sırası değişmiyor.
- Gerçek mini batch'te `A0A6D0ENQ7` içinden seçilen `MET` block'u tam `10`,
  `4rum_A/G` tam `11` slotla kabul edildi; quarantine `0`, koordinat/edge
  mutation `0`, source fingerprint değişimi `0`.
- Daha geniş 2.000-block RNA taraması büyük pickle nesneleri nedeniyle bu
  kabukta güvenilir tamamlanmış rapor üretmedi ve GO kanıtı sayılmadı. Bu
  kapsam Stage 4'te planlanmış en az 100 gerçek graph audit'inde ölçülecek.
- Stage 0-3 testleri `44 / 44`; `py_compile` ve `git diff --check` geçti. CUDA
  görünmediği için seçim/eşleme-only bu adımda GPU training çalıştırılmadı.

### Stage 3.4 sonucu (2026-08-04): GO

- `build_masked_coordinate_corruption` input batch'i mutate etmeden gerçek
  koordinatların klonu `clean_pos`, yalnız güvenli slotları değiştirilmiş
  `input_pos` ve `template_node_mask` döndürüyor. Henüz `batch.pos` veya model
  forward yoluna bağlanmadı.
- Protein `N/CA/C` ve RNA `C2'/C3'/C4'` Stage 1 frame'leri yeniden kullanıldı.
  Collinear/nonfinite frame ve RNA `O2' <= 0` orientation kontrolü uydurma
  fallback yerine açık coordinate quarantine üretiyor.
- Sentetik batch'te seçilmiş protein `10` + RNA `11` olmak üzere yalnız `21`
  node değişti; protein `4` ve RNA `12` backbone node'u ile seçilmemiş block
  bitwise aynı kaldı. ALA ve GLY aynı local 10-nokta template'ini `1e-6 Å`
  toleransla aldı.
- Gerçek `A0A6D0ENQ7/MET + 4rum_A/G` mini batch'inde template node `21`,
  değişmeyen node `30`, mapping/frame quarantine `0`, batch mutation `0`.
  Ortalama yer değiştirme protein `2.1302 Å`, RNA `2.3201 Å`; kaynak PKL
  fingerprint'leri değişmedi.
- Stage 0-3 testleri `47 / 47`; `py_compile` ve `git diff --check` geçti. Saf
  sonuç henüz training davranışı olmadığı ve CUDA görünmediği için fast GPU
  training çalıştırılmadı.

### Stage 3.5-3.7 sonucu (2026-08-04): GO

- Saf koordinat sonucu gerçek training/validation/test sırasına bağlandı:
  önce tek block-MLM seçimi yapılıyor, sonra yalnız aynı seçimin protein
  sidechain `10` ve RNA base `11` slotu template'e taşınıyor.
- Eski edge'lerden template node'una değmeyenler sıra ve `edge_attr` ile
  bitwise korunuyor. Template node'una değen eski edge'ler kaldırılıyor ve
  `input_pos` üzerinde aynı graph içinde `5 Å`, en fazla `48` komşulu
  bidirectional spatial edge'ler kuruluyor. UniMol-only batch güvenli ve tam
  bir no-op.
- Yeni yapay template edge'lerinin hiçbiri short-range distance loss'a,
  template node'u içeren hiçbir pair long-range sampler'a alınmıyor. MLM
  hedefi olan `original_types` değişmiyor.
- Sentetik karma protein/RNA validation batch'inde tam `21` template node
  uygulandı; değişmeyen koordinatlar ve MLM hedefleri bitwise aynı kaldı.
  Validation maskesi, koordinatları ve edge'leri tekrarlı koşuda deterministik
  çıktı. Template node embedding'leri kasıtlı `NaN` yapıldığında dahi iki
  distance loss sonlu kaldı; bu, loss dispatcher'ın bu node'ları gerçekten
  dışladığını doğruladı.
- Flag-off training/validation maskeleri, graph tensorleri, RNG state'i ve
  üç loss eski yol ile bitwise aynı kaldı. Bütün Stage 0-3 yapısal testleri
  `53 / 53`; `py_compile` ve `git diff --check` geçti. Production YAML
  flag'leri `false`, mevcut CIF/PKL/PT dosyaları yeniden yazılmadı.
- Tam ESA forward smoke denendi ancak mevcut ESA adjacency kodu CPU koşusunda
  CUDA tensorü oluşturmaya çalıştığı için `No CUDA GPUs are available` ile
  forward öncesinde durdu. Bu ortamda `torch.cuda.is_available() == false`;
  dolayısıyla öğrenme ve hız doğrulaması Stage 5/6 GPU A/B gate'lerine kaldı.

---

## Aşama 4 - Sentetik ve Gerçek Batch Yapısal Testleri

### Amaç

GPU training başlatmadan node, slot, edge ve loss kontratını kanıtlamak.

### Yapılacaklar

- [x] Sentetik GLY, ALA, TRP batch'i ile `4+10=14` kontratı test edilecek.
- [x] Sentetik A, C, G, U batch'i ile `12+11=23` kontratı test edilecek.
- [x] ALA'da S1 target `C`, kalan dokuz slot target `V`; GLY'de on slot `V`;
  TRP'de on slot gerçek atom olacak.
- [x] RNA'da standard A/C/G/U için beklenen real/V base slot sayıları test
  edilecek.
- [x] Gerçek protein chunk 0'dan en az 100 graph batch-time audit edilecek.
- [x] Geçici RNA virtual chunk'tan yalnız exact-23 block taşıyan uygun graph
  indeksleriyle en az 100 graph audit edilecek; source `.pt` yazılmayacak.
- [x] Eski ve yeni degree dağılımı, sıfır mesafeli edge ve eligible
  distance pair sayısı karşılaştırılacak.
- [ ] Maskelenmemiş V yığınlarının spatial mesaj kalabalığı ayrıca ölçülecek;
  masked-only template ile "maskelenmemiş V spatial edge'lerini dışla"
  varyantı ayrı bir ablation olarak karşılaştırılacak ve kanıt olmadan
  production davranışı yapılmayacak.

### Beklenen

- Her masked block tam 10/11 template slotu alır.
- Masked-template slotları arası sıfır mesafeli edge `0`.
- NaN/Inf `0`.
- `original_types`, `clean_pos`, block/graph membership değişmez.
- Distance loss support tamamen kaybolmaz; domain başına baseline support'un
  en az `%70`i kalır.

### STOP

- Slot sayısı, slot sırası veya target eşlemesi tek bir canonical block'ta
  bile yanlışsa.
- Template sonrası sıfır mesafeli edge varsa.
- Eligible distance support baseline'ın `%70`inin altına düşerse.
- Graph başına node sayısı veya batch composition beklenmedik biçimde
  değişirse.

### Gate artifact'i

`/tmp/mind_virtual_template_v1/stage4_structural_audit.json`

### Stage 4 ilk gerçek-chunk sonucu (2026-08-04): STOP

- Canonical sentetik target testleri geçti: GLY `0 real + 10 V`, ALA
  `CB + 9 V`, TRP `10 real`; RNA A/C/G/U sırasıyla `10/8/11/8` real base
  atomu ve kalan slotlarda V üretti.
- Salt-okunur mmap auditinde protein chunk 0'dan `100`, RNA virtual chunk
  0'dan `100` kabul edilmiş gerçek graph işlendi. Kaynak `.pt` fingerprint'leri
  değişmedi; node/target/block membership değişimi, nonfinite değer,
  rebuilt `edge_attr` hatası ve template node'una değen sıfır-mesafe edge `0`.
- Protein short/long eligible support oranları `0.8434 / 0.8000`; RNA
  `0.8751 / 0.8670`. Mean degree değişimi protein `76.63 -> 75.92`, RNA
  `95.94 -> 89.60`; `%70` support gate'i geçti.
- Ancak `100` sağlam protein graph bulmak için `112` graph incelendi ve
  `12` graph runtime tarafından fail-closed reddedildi. Hepsinde seçilen
  terminal residue'nün 14 slotu içinde `OXT` sidechain slotu tüketmiş ve
  beklenen bir V slotunu düşürmüştü. Bu nedenle
  `terminal_atom_in_fixed_block/wrong_slot_count/noncanonical_backbone_slot_partition/virtual_slot_index_mismatch`
  quarantine üretildi.
- Chunk-geneli salt-okunur sayımda `4,000` proteinin `3,932` tanesinde tam
  bir `OXT` bulundu. Sorun yalnız 12 moleküle özgü değildir; `%15` block mask
  terminal residue'ye denk geldiğinde görünür olur. Mevcut adapter OXT'yi artık
  sabit 14 slottan çıkarıyor fakat production cache eski semantiği taşıyor.
- Bu nedenle kabul edilmiş graph geometrisi iyi olsa da Stage 4 protein gate'i
  **STOP**: GPU A/B training'den önce eski cache'i yazmadan terminal-OXT
  block'larını template/block-MLM seçiminden dışlayan compatibility yolu veya
  yeni cache üretimi seçilip yeniden `100 + 100` audit geçmelidir.

### Stage 4 OXT compatibility tekrar sonucu (2026-08-05): GO

- Yalnız `masked_slot_template.enabled=true` ve protein template'i açıkken,
  legacy `OXT` taşıyan terminal block protein block-MLM aday havuzundan
  çıkarıldı. Mask oranı kalan canonical residue'ler üzerinde korunuyor. OXT
  yeniden etiketlenmedi, koordinatı değiştirilmedi ve `.pt` yazılmadı.
- Template kapalı yol aynı kod/RNG davranışını koruyor. Sentetik production-yol
  testinde iki canonical residue seçilirken aradaki OXT block'u hiç seçilmedi.
- Aynı gerçek-chunk audit tekrarında protein `100/100` graph ilk `100`
  incelemede kabul edildi; runtime rejection `0`. RNA exact-23 audit sonucu
  yine `100` kabul, rejection `0`.
- Protein template block/node sayısı `2,547 / 25,470`; template node'una değen
  sıfır-mesafe edge ve rebuilt edge_attr mismatch `0`. Short/long eligible
  support `0.8439 / 0.8005`, mean degree `76.81 -> 76.15`.
- RNA template block/node sayısı `1,143 / 12,573`; template node'una değen
  sıfır-mesafe edge ve rebuilt edge_attr mismatch `0`. Short/long eligible
  support `0.8751 / 0.8670`, mean degree `95.94 -> 89.60`.
- Her iki kaynak `.pt` fingerprint'i önce/sonra aynı. Gate artifact'i
  `status=GO`, failure listesi boş. Stage 4 yapısal gate tamamlandı; production
  template flag'leri hâlâ kapalıdır.

---

## Aşama 5 - Protein Küçük Öğrenilebilirlik ve Eş Koşullu A/B

### Amaç

Cache değiştirmeden protein CB/V simetrisinin gerçekten kalktığını ve
backbone/sidechain öğrenmesinin bozulmadığını göstermek.

### Yapılacaklar

- [x] 32-128 protein graph ile küçük overfit smoke yapılacak; amaç pipeline
  ve gradient akışıdır, final accuracy değildir.
- [x] Aynı seed/config/chunk/GPU ile flag-off protein kontrolü koşulacak.
- [x] Yalnız protein template flag'i açılarak treatment koşulacak.
- [x] En az 3 epoch; CB sonucu oynaksa 10 epoch eş koşullu uzatma yapılacak.
- [x] N/CA/C/O, CB, CG/CD/CE, sidechain-real, V ve per-AA real
  all-or-nothing karşılaştırılacak.
- [x] MLM, short/long distance loss ve eligible support raporlanacak.
- [x] Ortalama step-time, peak VRAM ve `edge_rebuild_ms` raporlanacak.

### Beklenen GO

- CB recall kontrolün en az `+0.10` mutlak üstüne çıkar.
- N/CA/C/O her biri `>=0.90` ve kontrolün en fazla `0.03` altındadır.
- V recall `>=0.80`.
- CG/CD/CE veya sidechain-real kontrolün `0.05`ten fazla altına düşmez.
- Total loss azalır; eligible distance loss kontrolden en fazla `%10` kötüdür.
- Step-time artışı `<=%20`.

### STOP

- CB iyileşmesi `<+0.05` ise slot kimliği ve mask yolu yeniden incelenir.
- Backbone atomlarından biri `<0.90` ise.
- V recall `<0.70` ise.
- Gerçek sidechain recall `>0.05` mutlak düşerse.
- Loss NaN/Inf olur, epochlar boyunca azalmaz veya distance support yetersizse.
- Step-time artışı `>%35` ise RNA entegrasyonundan önce edge yolu optimize
  edilir.

### Gate artifact'i

`/tmp/mind_virtual_template_v1/stage5_protein_ab.json`

### Stage 5 sonucu (2026-08-09): STOP

Bu aşamada önce yalnız maskelenen residue slotlarını template'e taşıyan yol
genişletildi: maskelenmemiş residue'lerdeki gerçek atomlar yerinde bırakıldı,
yalnız V padding node'ları residue local frame'indeki karşılık gelen canonical
slot noktasına taşındı. Her batchte bütün görünür V'leri yeniden hesaplamanın
maliyeti yüksek olduğu için kaynak cache'e dokunmadan yeni bir geçici cache
üreten `data_loading/precompute_visible_virtual_templates.py` eklendi. Runtime
bu cache ile yalnız seçilen MLM target residue edge'lerini yeniden kuruyor.
Production flag'leri kapalı kaldı.

Son gerçek-chunk audit:

- Protein `100/100`, RNA `100/100`; runtime target rejection `0`.
- Protein: `90,967` görünür V node, `25,470` target-template node; RNA:
  `11,519` görünür V node, `12,573` target-template node taşındı/denetlendi.
- Taşınan template node'una değen sıfır mesafeli edge iki domainde de `0`;
  rebuilt `edge_attr` mismatch `0`.
- Fiziksel distance support protein short/long `0.9089/0.8660`, RNA
  `0.8841/0.8777`.
- Kaynak protein ve RNA `.pt` fingerprintleri audit öncesi/sonrası aynıdır.
- 128-protein preposition cache reload kontrolünde `133,332` V node yer
  değiştirdi, gerçek atom coordinate değişimi `0`, metadata farkı `0`, taşınan
  V'ye değen sıfır edge `0` oldu. Legacy OXT taşıyan `126` terminal block
  quarantine edildi ve değişmeden kaldı.

Aynı 128 graph, seed `42`, batch `4`, 10 epoch kontrol/treatment sonucu:

| Metrik | Flag-off kontrol | Canonical görünür-V + target-template | Delta |
|---|---:|---:|---:|
| Best validation total loss | `3.476` | `3.732` | `+0.256` |
| Overall atom accuracy | `0.7761` | `0.6465` | `-0.1296` |
| Backbone recall | `0.7107` | `0.7471` | `+0.0364` |
| CB recall | `0.0446` | `0.0000` | `-0.0446` |
| Gerçek sidechain recall (SCr) | `0.5250` | `0.0030` | `-0.5220` |
| V recall | `0.9668` | `0.9715` | `+0.0047` |

Hız benchmarkı aynı geçici subset ve A6000 GPU üzerinde 5 warmup + 15 ölçüm
stepidir. Kontrol median compute `0.1281 s`, prepositioned treatment
`0.2576 s`: yaklaşık `2.01x` ve `%101` artış, dolayısıyla `%35` STOP eşiğini
aştı. Peak allocated VRAM kontrol/treatment yaklaşık `6.79/6.58 GiB`; bellek
sorunu yoktur. Prepositioned cache'i template flag kapalı çalıştırmak
`0.1289 s` verdi; ek maliyetin cache topolojisinden değil, her step seçilen
target slotlarını kurup incident edge'lerini yenilemekten geldiği izole edildi.

Sonuç yalnız optimizasyon STOP'u değildir; öğrenilebilirlik STOP'u kesindir.
Canonical coordinate, MIND encoder'a güvenilir bir residue-içi slot kimliği
sağlamadı. Encoder node girdisinde atomic-number embedding ve geometrik
özetleri kullanıyor; `pos_code`/`slot_idx` embedding'i yok ve configteki
`position_embedding_dim` kullanılmıyor. Maskelenen 10 node aynı MASK türünü
aldığında model slotları yeterince ayıramayıp V çoğunluğuna kaçtı. Veri priorı
da tanıyı doğruluyor: 128 graph içindeki canonical slot 1 hedeflerinin `%91.18`
kadarı C, yalnız `%8.82` kadarı V olduğu halde treatment CB'yi hiç bulamadı.
Bu, coordinate'in slot etiketi gibi kullanılamadığının doğrudan kanıtıdır.

Bu nedenle Stage 6 RNA uzun öğrenilebilirlik A/B'sine geçilmez. RNA kodunun
sentetik ve gerçek yapısal yolu geçmiştir; fakat proteinle ortak encoder'daki
slot-kimliği eksikliği çözülmeden RNA eğitimi aynı başarısız hipotezin pahalı
tekrarı olur. Sonraki bağımsız deney, MIND'e domain-aware residue/nucleotide
local `slot_idx` embedding'i ekleyip MASK node'larına slot kimliğini açıkça
vermektir. V/real etiketi hedef blockta input olarak verilmemelidir.

### Stage 5.1 local-slot sonucu (2026-08-09): GO

Önceki STOP'un iki ayrı nedeni kapatıldı:

1. İlk A/B'de template kolu legacy OXT blocklarını dışlarken kontrol kolu
   dışlamıyordu. Cache'ten OXT silme/dönüştürme yapılmadı. Yerleşik güvenli
   politika bağımsız `protein_block_mlm_exclude_legacy_oxt` flag'ine bağlandı;
   aynı OXT graphında kontrol ve template MLM maskelerinin birebir aynı olduğu
   test edildi.
2. Diğer grubun yaklaşımı MIND'e uyarlandı: protein için residue-local `1..14`,
   RNA için nucleotide-local `15..37` embedding eklendi. Kimlik yalnız domain,
   `block_idx` ve block içi node sırasından üretilir; `z`, `pos_code` veya
   `is_virtual` okunmaz. Bu nedenle bütün `z` değerleri maskelendiğinde de slot
   id aynı kalır ve hedef real/V bilgisi sızmaz. V hedef kaybı diğer gruptaki
   gibi `real_loss + 0.2 * virtual_loss` hesaplanır.

128 protein graph, seed `42`, batch `4`, GPU 6, 10 epoch gerçek-chunk koşusu:

| Metrik | Eski flag-off kontrol | Local slot + V ağırlığı | Delta |
|---|---:|---:|---:|
| Overall atom accuracy | `0.7761` | `0.8416` | `+0.0655` |
| Backbone recall | `0.7107` | `1.0000` | `+0.2893` |
| CB recall | `0.0446` | `1.0000` | `+0.9554` |
| Gerçek sidechain recall (SCr) | `0.5250` | `0.6877` | `+0.1627` |
| V recall | `0.9668` | `0.8330` | `-0.1338` |
| Residue real-only all-or-nothing | `0.0000` | `0.4327` | `+0.4327` |

V recall düşmesine rağmen `0.80` GO eşiğinin üstündedir; backbone, CB, gerçek
sidechain ve overall aynı anda yükseldi. Canonical-template yoluyla aynı epoch
sonu throughput eski templatesiz kontrolden düşüktür; fakat slot embedding
eklenmeden önceki canonical-template koşusu da `2.31 it/s`, yeni koşu da
`2.31 it/s` olduğundan yeni embedding/loss değişikliği ölçülebilir ek yavaşlama
getirmedi. Target edge-rebuild optimizasyonu Stage 7'de açık kalır.

Config: `/tmp/mind_virtual_template_v1/stage6_protein_slot_vweight_10ep.yaml`
Log: `/tmp/mind_virtual_template_v1/stage6_slot_vweight_10ep.log`
Checkpoint: `/tmp/mind_virtual_template_v1/stage6_slot_vweight_10ep_output/`

Protein öğrenilebilirlik kapısı GO'dur; sıradaki aktif iş aynı ortak slot
mekanizmasının RNA Stage 6 gerçek-chunk A/B doğrulamasıdır.

---

## Aşama 6 - RNA Ham Veri Kontratı ve Runtime A/B

### Amaç

RNA'nın 23-slot/anchor veri kalitesini ham CIF'lerde kesinleştirmek ve mevcut
geçici `.pt`yi değiştirmeden RNA template'inin C1' stacking zararını geri
aldığını göstermek.

### Yapılacaklar

- [ ] `/media/ubuntu/8TB/arya/rna_data/` altındaki 12,933 CIF için
  deterministik, sıralı manifest oluşturulacak.
- [ ] Standard A/C/G/U ve modified residue'ler ayrı sayılacak.
- [ ] Canonical 12-backbone atomunun eksik/duplike olma nedenleri ve local-frame
  anchor coverage raporlanacak.
- [ ] Accepted/quarantine politikası kod ve test ile kesinleştirilecek; accepted
  block için exact-23 assert zorunlu olacak.
- [ ] RNA nucleotide block-MLM sentetik ve gerçek batchte doğrulanacak.
- [ ] Mevcut nonvirtual RNA kontrol metriği bağlamsal referans olarak tutulacak.
- [ ] Ana eş koşullu karşılaştırma aynı geçici C1'-stacked RNA `.pt`
  üzerinde yapılacak:
  - B: runtime template kapalı.
  - C: runtime template açık.
- [ ] B ve C yalnız exact-23 block kontratını sağlayan graph indeksleriyle,
  aynı seed/config/split/GPU'da en az 3 epoch koşulacak.
- [ ] RNA overall, C1', backbone, BaseR, nucleotide all-or-nothing, V,
  short/long distance loss ve step-time karşılaştırılacak.

### Bilinen referanslar

| Metrik | Nonvirtual kontrol | C1' stacked başarısız deney |
|---|---:|---:|
| Overall | `0.7252` | `0.5569` |
| C1' recall | `0.9414` | `0.8043` |
| Backbone recall | `0.8665` | `0.7106` |
| BaseR | `0.5459` | `0.4613` |
| Nucleotide all-or-nothing | `0.3895` | `0.1802` |

### Beklenen GO

- Accepted RNA block'larında exact-23 ve anchor coverage `%100`.
- Quarantine oranı `<=%2.5`; nedenler JSON/CSV olarak kayıtlı.
- C1' recall `>=0.90`.
- Backbone recall `>=0.83`.
- Overall `>=0.69`, BaseR `>=0.51`, all-or-nothing `>=0.34`.
- V recall `>=0.80` ve V support `>0`.
- Masked-template sıfır mesafeli edge `0`.
- Eligible distance loss kontrolün en fazla `%10` üstünde.
- Step-time artışı `<=%20`.

### STOP

- Accepted block'ta eksik/duplike backbone veya anchor varsa.
- C1' `<0.87`, backbone `<0.80` veya V `<0.70` ise.
- Template treatment, aynı C1'-stacked cache kontrolüne göre RNA overall,
  BaseR veya all-or-nothing metriğini daha da düşürürse.
- Distance loss/support veya step-time global STOP eşiklerini aşarsa.

### Gate artifact'i

`/tmp/mind_virtual_template_v1/stage6_rna_ab.json`

---

## Aşama 7 - Hız Optimizasyonu ve Ortak Kodun Sertleştirilmesi

### Amaç

Protein ve RNA davranışı geçtikten sonra full spatial graph rebuild maliyetini
gerekirse yalnız etkilenen block edge'lerine indirmek.

### Yapılacaklar

- [ ] Full graph rebuild ile incremental masked-block rebuild aynı küçük batchte
  edge-set ve edge-distance olarak karşılaştırılacak.
- [ ] Tier 1 sequence edge, Tier 2 spatial edge ve dinamik Tier 3 random edge
  duplicate politikaları test edilecek.
- [ ] Incremental yol yalnız edge-set eşdeğerliği kanıtlanırsa default adayı
  olacak.
- [ ] 100 warmup + en az 500 measured step ile protein/RNA/multidomain throughput
  benchmark edilecek.
- [ ] Peak VRAM ve CPU data-loader bekleme süresi raporlanacak.

### Beklenen

- Optimize yol referans full-rebuild ile aynı graph semantiğini vermeli.
- Step-time artışı `<=%20`, tercihen `<=%10`.
- Peak VRAM baseline'a göre `>%15` artmamalı.

### STOP

- Incremental edge-set full rebuild'den farklıysa optimizasyon kabul edilmez;
  doğru fakat yavaş yol korunur.
- Hız için leakage veya distance exclusion kontratı gevşetilmez.

---

## Aşama 8 - Production Adapter ve Yeni Cache'ler

### Giriş kapısı

Bu aşama yalnız Stage 5 protein ve Stage 6 RNA `GO` ise başlar.

### Amaç

Runtime prototipinde doğrulanan semantiği açık veri kontratıyla protein ve RNA
adapter/cache pipeline'ına yerleştirmek. Eski cache'ler korunur.

### Yapılacaklar

- [ ] Ortak tensor alanları kesinleştirilecek: `slot_idx`, `is_virtual`,
  `block_idx`; gerekirse `edge_tier`.
- [ ] Bu alanlar label/metric ve runtime graph kontrolü için tutulacak;
  maskelenen blockta `is_virtual` modele input feature olarak verilmeyecek.
- [ ] Protein adapter accepted residue'lerde exact-14 ve canonical slot sırasını
  assert edecek.
- [ ] RNA adapter accepted nucleotide'lerde exact-23 ve canonical backbone/base
  slot sırasını assert edecek; bozuklar quarantine raporuna gidecek.
- [ ] Unmasked V node'ların statik spatial edge politikası ayrı test edilecek;
  gerçek/V etiketi masked block graph'ına sızdırılmayacak.
- [ ] Yeni cache'ler eski dizinlerin üzerine yazılmayacak:
  - protein adayı: `/home/yusuf/data/proteins/processed_14atom_template_v1/`
  - RNA adayı: `/home/yusuf/data/rna/processed_23slot_template_v1/`
- [ ] Önce her domain için yalnız bir chunk üretilecek.
- [ ] Bir-chunk adapter, PKL, PT, metadata, NaN/Inf, slot, V ve edge auditleri
  geçecek.
- [ ] Yeni bir-chunk cache ile Stage 5/6 fast A/B tekrar edilecek.
- [ ] Yalnız tekrar A/B geçerse tüm protein ve RNA chunk'ları üretilecek.
- [ ] Tüm chunk'larda graph/atom/block/V/edge toplamları ve hash/mtime manifesti
  kaydedilecek.

### Beklenen

- Protein accepted block'larının tamamı exact-14.
- RNA accepted block'larının tamamı exact-23.
- `slot_idx` her blockta kesintisiz ve benzersiz.
- `is_virtual` ile `pos_code V*` ve `z==120` kontratı domain politikasıyla
  uyumlu.
- Yeni bir-chunk fast sonuçları runtime prototipindeki kabul aralıklarında.

### STOP

- Eski production cache'e yazma/overwrite riski varsa.
- Bir chunk audit veya tekrar A/B geçmeden full reprocess başlatılırsa.
- Raw CIF/PDB'den üretim ile runtime prototipi slot/target semantiği farklıysa.

---

## Aşama 9 - Multidomain Doğrulama ve Tamamlama

### Amaç

Yeni protein + RNA kodunun UniMol ile aynı batchlerde kararlı olduğunu kanıtlamak
ve full training öncesi son kapıyı kapatmak.

### Yapılacaklar

- [ ] Yeni cache yollarıyla fast PDB + RNA + UniMol smoke koşulacak.
- [ ] Fast geçerse en az iki epoch tam-veri multidomain run koşulacak.
- [ ] Son batchlere kadar PDB + RNA + UniMol birlikte kalacak.
- [ ] Domain bazlı MLM, short/long distance loss ve eligible support izlenecek.
- [ ] Protein ve RNA kabul metrikleri Stage 5/6 eşikleriyle karşılaştırılacak.
- [ ] Throughput/VRAM baseline ile karşılaştırılacak.
- [ ] Config path, epoch, run name, atom budget ve chunk range production için
  temizlenecek.
- [ ] Tasarım dokümanları ve `CURRENT_TODOS.md` nihai sonuçla güncellenecek.

### Beklenen GO

- `val_total_loss` ve her domain loss'u azalır.
- Protein CB iyileşmesi korunur; N/CA/C/O `>=0.90`, V `>=0.80`.
- RNA C1' `>=0.90`, backbone `>=0.83`, V `>=0.80`.
- UniMol loss/metric baseline'a göre beklenmedik regresyon göstermez.
- Step-time global kabul eşiğinde kalır.
- Metabolit kapalı kalabilir; PDB + RNA + UniMol tamamlanmasını engellemez.

### STOP

- Son batchlerde domainlerden biri kaybolursa.
- Bir domain iyileşirken diğerinde kabul eşiğini aşan regresyon oluşursa.
- Loss NaN/Inf, artan validation loss veya tekrarlayan OOM varsa.
- Eski cache'e dönüş yolu ve manifest doğrulanmadan production config
  değiştirilirse.

### Tamamlanma tanımı

Protein ve RNA virtual atom kodu ancak aşağıdakilerin tamamı olduğunda
tamamlanmış sayılır:

- [x] Ortak template modülü ve leakage testleri geçiyor.
- [x] Protein/RNA block masking ve runtime graph yolları testli.
- [x] Distance loss template node'larından korunuyor.
- [ ] Protein ve RNA eş koşullu A/B gate'leri geçiyor.
- [ ] Yeni cache'ler ayrı dizinlerde tam audit edilmiş.
- [ ] İki-epoch multidomain doğrulama geçiyor.
- [ ] Eski cache/config rollback yolu dokümante.

---

## Checkbox ve Artifact Politikası

- Bu dosya kalıcı ilerleme kaydıdır; tamamlanan madde `[x]` yapılır, silinmez.
- Her `[x]` için komut, config, seed, veri subset'i ve artifact/log yolu aynı
  aşamanın altına veya gate JSON'una kaydedilir.
- `CURRENT_TODOS.md` yalnız aktif/sonraki işleri tutar; tamamlanan `[x]`
  maddeler oradan kaldırılır.
- Bir metrik gate'i geçmediyse checkbox tamamlanmış sayılmaz; sonuç STOP notu
  olarak kaydedilir.

## Current Todo

- [ ] **Stage 6 - RNA local-slot gerçek-chunk A/B:** Protein GO sonucundaki
  ortak slot embedding ve V-weighted loss'u geçici C1'-stacked RNA chunkında
  doğrula. C1', backbone, BaseR, V, distance loss ve throughput gate'lerini
  raporla; GO olmadan full-cache migration başlatma.
