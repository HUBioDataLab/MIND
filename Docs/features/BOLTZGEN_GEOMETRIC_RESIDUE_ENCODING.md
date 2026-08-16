# BoltzGen Geometrik Residue Kodlaması ve MIND Uygulama Planı

## Kısa sonuç

BoltzGen virtual atomları gerçek atomlardan uzağa dağıtılmış, öğrenilmiş
"pseudo-koordinatlar" olarak kullanmıyor. Tasarlanacak her protein residue'sünü
14 koordinatla temsil ediyor ve gerçek sidechain'den artan slotları belirli
sayılarda backbone **N** ve **O** atomlarının üstüne bindiriyor. Bu bindirilmiş
atomların sayısı amino asit kimliğinin geometrik kodudur.

Bu yöntem BoltzGen'de çalışır çünkü modelin görevi atom tiplerini `V` olarak
sınıflandırmak değil, gürültülü koordinatlardan 14 koordinatı üretmektir. MIND'de
ise koordinatlar temiz ve görünür kalırken atom elementleri MLM ile maskelenir;
bu nedenle aynı kodu doğrudan kopyalamak hedef kimliği geometriden sızdırır ve
radius graph içinde sıfır mesafeli edge'ler üretir.

Dolayısıyla iki ayrı karar vardır:

1. **Gerçek BoltzGen yaklaşımı:** MIND'e koordinat-diffusion/design görevi,
   geometrik decoder ve marker atom semantiği eklenir. Bu büyük bir model ve
   eğitim amacı değişikliğidir.
2. **MIND'in MLM problemini çözmek:** BoltzGen kodu kopyalanmaz; maskelenen
   block'un slot geometrisi residue/nucleotide tipinden bağımsız hale getirilir.
   Mevcut hedef için önerilen kısa yol budur.

Kaynak: [`papers/boltzgen.pdf`](../../papers/boltzgen.pdf), Bölüm 3.1,
özellikle Şekil 7 ve PDF sayfa 12.

## BoltzGen hangi problemi çözüyor?

Bir protein tasarım modelinde amino asit kimliği henüz bilinmiyorsa sidechain
atom sayısı da bilinmez. Örneğin GLY sidechain ağır atomu taşımazken TRP 10 ağır
sidechain atomu taşır. Diffusion tensörünün boyutunu üretim sırasında değiştirmek
yerine BoltzGen her tasarım residue'sü için baştan sabit 14 koordinat ayırır:

- Slot 1-4: `N`, `CA`, `C`, `O` backbone atomları.
- Slot 5-14: En fazla 10 sidechain atomu veya marker/virtual atom.

Gerçek sidechain'i 10 atomdan kısa olan amino asitlerde boş slotlar silinmez.
Bu slotlar amino asit türünü kodlayan yardımcı koordinatlar haline gelir ve
decode sonrasında atılır.

Bu temsilin amacı MIND'deki gibi atom sayısı sızıntısını engellemek değildir.
Tam tersine, amino asit kimliğini **bilinçli olarak geometriye kodlamaktır**.
BoltzGen bunu ayrı bir discrete residue classifier kullanmadan tamamen sürekli
koordinat uzayında üretim yapabilmek için kullanır.

## Virtual atomlar tam olarak nasıl yerleştiriliyor?

Her amino asit için önce gerçek sidechain atomları atom14 sırasındaki slotlara
yazılır. Kalan slotların bir bölümü backbone `N`, geri kalanı backbone `O`
koordinatına tam olarak kopyalanır:

```text
marker_position = position(backbone_N)  veya  position(backbone_O)
```

Makaledeki `#N` ve `#O`, sırasıyla N ve O üzerine bindirilen marker sayısıdır.
Her satırda şu eşitlik sağlanır:

```text
gerçek sidechain atom sayısı + #N + #O = 10
```

Şekil 7'deki eksiksiz kod tablosu:

| Residue | Gerçek sidechain slotları | #N | #O |
|---|---|---:|---:|
| GLY | - | 0 | 10 |
| ALA | CB | 0 | 9 |
| CYS | CB, SG | 0 | 8 |
| SER | CB, OG | 8 | 0 |
| PRO | CB, CG, CD | 0 | 7 |
| THR | CB, OG1, CG2 | 3 | 4 |
| VAL | CB, CG1, CG2 | 7 | 0 |
| ILE | CB, CG1, CG2, CD1 | 0 | 6 |
| ASN | CB, CG, OD1, ND2 | 1 | 5 |
| ASP | CB, CG, OD1, OD2 | 2 | 4 |
| LEU | CB, CG, CD1, CD2 | 4 | 2 |
| MET | CB, CG, SD, CE | 6 | 0 |
| GLN | CB, CG, CD, OE1, NE2 | 0 | 5 |
| GLU | CB, CG, CD, OE1, OE2 | 2 | 3 |
| LYS | CB, CG, CD, CE, NZ | 5 | 0 |
| HIS | CB, CG, ND1, CD2, CE1, NE2 | 0 | 4 |
| PHE | CB, CG, CD1, CD2, CE1, CE2, CZ | 0 | 3 |
| ARG | CB, CG, CD, NE, CZ, NH1, NH2 | 3 | 0 |
| TYR | CB, CG, CD1, CD2, CE1, CE2, CZ, OH | 0 | 2 |
| TRP | CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2 | 0 | 0 |

Çiftler 20 canonical amino asit için benzersizdir. Aynı sayıda gerçek sidechain
atomu olan residue'ler bile farklı `(N, O)` çiftleriyle ayrılır. Örneğin:

- PRO: `(0, 7)`
- THR: `(3, 4)`
- VAL: `(7, 0)`

Üçünde de üç gerçek sidechain atomu ve yedi marker vardır; marker dağılımı
residue kimliğini belirler.

## Decode işlemi

Diffusion başlangıçta 14 rastgele koordinattan başlar ve bunları aşamalı olarak
denoise eder. Son koordinatlardan amino asit şu şekilde çıkarılır:

1. İlk dört slot `N`, `CA`, `C`, `O` kabul edilir.
2. Slot 5-14 arasındaki koordinatlardan backbone N'ye 0.5 Å'dan yakın olanlar
   sayılır: `predicted_n_count`.
3. Backbone O'ya 0.5 Å'dan yakın olanlar sayılır: `predicted_o_count`.
4. `(predicted_n_count, predicted_o_count)` çifti tablodan amino aside çevrilir.
5. N/O üzerine bindirilmemiş kalan slotlar, bulunan amino asidin atom14
   şablonuna göre `CB`, `CG`, `OD1` gibi gerçek sidechain atomlarına atanır.
6. N/O marker'ları çıktı yapısından atılır.

Örneğin model 3 atomu N çevresine, 4 atomu O çevresine ve 3 atomu sidechain
geometrisine yerleştirirse `(3, 4)` kodundan THR seçilir. Sidechain'deki üç
koordinat sırasıyla `CB`, `OG1`, `CG2` olarak yorumlanır.

Buradaki 0.5 Å yalnızca decode toleransıdır. Eğitim hedefinde marker koordinatı
anchor koordinatına tam eşittir; diffusion çıktısının sayısal olarak tam eşit
olması beklenmez.

## Neden BoltzGen'de yığılma çalışabiliyor?

BoltzGen ve MIND aynı koordinat düzenini farklı anlamlarla kullanır:

| Konu | BoltzGen | MIND mevcut sistem |
|---|---|---|
| Ana görev | Koordinat diffusion / yapı üretimi | Element MLM + distance prediction |
| Model girdisi | Gürültülü koordinatlar | Temiz, sabit koordinatlar |
| Virtual'ın anlamı | Residue kimliğini kodlayan marker | Tahmin edilmesi beklenen jenerik `V` elementi |
| Kimlik bilgisi | Geometrik kodla bilinçli üretilir | Maskelenen block'tan saklanmak istenir |
| Decode | N/O çevresindeki marker sayısı | Her node için element sınıflandırması |
| Graph | Atom/token attention ve diffusion | Önceden cache'lenmiş radius graph |
| Marker sonucu | Decode sonrası atılır | Eğitim grafiğinde node ve edge olarak kalır |

BoltzGen koordinatlara gürültü eklediği için model temiz N/O yığınını doğrudan
input etiketi olarak görmez; denoising hedefi olarak üretmeyi öğrenir. Ayrıca
marker'ı `V` sınıfı olarak tahmin eden bir MLM loss'u yoktur. MIND'de ise temiz
koordinat ve bunlardan önceden kurulmuş edge'ler görünür kalır. Sadece `z`
element tokenı maskelenir.

Bu yüzden BoltzGen'deki yığılma bir kusur değil, tasarımın iletişim kanalıdır.
MIND'de aynı iletişim kanalı block-MLM cevabını input geometriye yazmış olur.

## RNA için BoltzGen ne söylüyor?

Makale RNA ve DNA'yı hedef/bağlanılacak molekül olarak all-atom biçiminde
destekliyor; ancak Şekil 7'deki sabit marker kodu yalnızca **tasarlanan protein
residue'leri** için tanımlanıyor. RNA nucleotide'leri için 23-slot marker tablosu,
anchor seçimi veya decode kuralı verilmemiştir.

Dolayısıyla RNA için `C1'` üzerine V koymak BoltzGen'in önerdiği bir yöntem
değildir; MIND içinde yaptığımız bir varsayımdı. A/B testi bu varsayımı reddetti:

| RNA metriği | Virtual olmayan kontrol | C1' üzerine V |
|---|---:|---:|
| Overall accuracy | 0.7252 | 0.5569 |
| C1' recall | 0.9414 | 0.8043 |
| Backbone recall | 0.8665 | 0.7106 |
| Real base accuracy | 0.5459 | 0.4613 |
| Nucleotide all-or-nothing | 0.3895 | 0.1802 |

C1' cache'inde ayrıca 868,574 sıfır mesafeli edge oluştu. Bu sonuç "RNA'da
backbone stacking çalışıyor" hipotezini kapatır.

## MIND'e birebir BoltzGen implementasyonu

Gerçek BoltzGen davranışı isteniyorsa değişiklik yalnız adapter'da yapılamaz.
Aşağıdaki parçaların birlikte uygulanması gerekir.

### 1. Ortak atom14 şablonu

Yeni bir modül protein atom14 slotlarını, `(N, O)` kodlarını ve ters decoder
tablosunu tek kaynakta tutmalıdır. Modül import sırasında şunları assert etmelidir:

- 20 residue kodu benzersiz.
- Her residue için `real_sidechain + #N + #O == 10`.
- Gerçek sidechain atom adları positional vocabulary ile tam uyumlu.
- Encode-decode round-trip bütün canonical residue'lerde başarılı.

### 2. Adapter alanları

Her atom için en az şu bilgiler korunmalıdır:

- `atom14_slot`: 0-13 slot indeksi.
- `is_marker`: Gerçek atom ile geometrik marker ayrımı.
- `marker_anchor`: Marker ise `N` veya `O`.
- `block_idx`: Hangi residue'ye ait olduğu.
- Gerçek sidechain atomlarında canonical `pos_code`.

Marker'ı yalnız `element='V'` olarak göstermek yeterli değildir; `V`, MIND'in
element MLM hedefiyle karışır. Marker semantiği ayrı bir alan olmalıdır.

### 3. Eğitim örneği seçimi

BoltzGen sadece "designed" olarak seçilen residue'leri augmented atom14
gösterimine çevirir. MIND tarafında da bir `design_mask`/`geometric_design_mask`
oluşturulmalı; folding/context residue'leri normal all-atom temsilini
korumalıdır. Her protein atomuna marker eklemek makaledeki görev karışımıyla aynı
değildir.

### 4. Dinamik koordinat gürültüsü ve graph

Temiz marker yığınlarından bir kez radius graph üretip bütün eğitim boyunca
kullanmak BoltzGen'e denk değildir. Minimum doğru uygulamada:

- Koordinat gürültüsü batch hazırlanırken örneklenmeli.
- Model gürültülü koordinatları görmeli.
- Koordinata bağlı komşuluk kullanılacaksa edge'ler gürültülü koordinatlardan
  dinamik kurulmalı veya BoltzGen benzeri atom/token attention kullanılmalı.
- Temiz hedefteki sıfır mesafeler distance-classification örneklerini domine
  etmemeli; marker edge'leri mevcut short-range distance loss'undan ayrılmalı.

### 5. Equivariant denoising head

MIND'de deneysel bir coordinate-denoising head bulunuyor; ancak scalar node
embedding'den doğrudan 3D vektör üreten mevcut MLP gerçek bir SE(3)-equivariant
BoltzGen diffusion modülü değildir ve production config'lerinde etkin değildir.
Faithful uygulama için equivariant koordinat çıktısı, noise schedule, rigid
alignment ve marker koordinat loss'u gerekir.

### 6. Geometrik decoder ve metrikler

Inference/validation aşamasında 0.5 Å kuralı uygulanmalı ve şu metrikler
raporlanmalıdır:

- 20-class residue decode accuracy.
- `(N, O)` marker-count exact accuracy.
- Marker-anchor mesafe dağılımı.
- Gerçek sidechain atom RMSD/lDDT.
- Decode sonrası geçersiz veya birden fazla koda yakın örnek oranı.

Marker'lar decode sonrasında kimyasal yapıdan çıkarılmalıdır.

### 7. RNA uzantısı ayrı bir araştırma kararıdır

RNA'ya uygulanacaksa dört nucleotide için yeni ve benzersiz geometrik kodlar
tasarlamak mümkündür, fakat bu makaleden alınmış bir uygulama olmaz. Anchor'ların
seçimi, modified nucleotide politikası ve decode eşikleri ayrı deneylerle
doğrulanmalıdır. Özellikle bu kodlar RNA atom-MLM hedefiyle aynı anda
kullanılırsa nucleotide kimliğini doğrudan sızdırır.

## MIND için önerilen uygulama

MIND'in kısa vadeli hedefi binder generation değil, çok modlu masked atom
pretraining olduğu için literal BoltzGen portu önerilmez. Daha küçük ve hedefle
uyumlu çözüm şudur:

> Bu bölüm makalenin yöntemi değil, MIND'in MLM amacı için önerilen ve henüz A/B
> testi yapılmamış bir tasarımdır.

1. Protein için 10, RNA için 11 side/base slotu korunur.
2. Maskelenen block'taki bütün slotların `z` tipi gizlenir.
3. Maskelenen block'un slot koordinatları residue/nucleotide tipinden bağımsız,
   canonical bir local-frame şablonuna taşınır. Gerçek ve virtual slot aynı input
   şablonunu kullanır; böylece geometri cevabı sızdırmaz.
4. Protein local frame'i `N-CA-C`, RNA local frame'i `C1'-C2'-C3'` gibi her
   canonical block'ta bulunması zorunlu anchor'lardan oluşturulur.
5. Şablondaki 10/11 nokta birbirinden ve gerçek anchor'lardan ayrıdır; sıfır
   mesafeli edge üretmez.
6. Bu koordinat değiştiği için masked block'a temas eden radius edge'leri batch
   sırasında yeniden oluşturulur. Cache'deki eski edge'ler kullanılamaz.
7. Target yine gerçek element veya `V` olabilir; fakat input slot geometrisi
   target sayısını veya türünü açıklamaz.

Bu yaklaşım BoltzGen'in "sabit slot + geometriyi kontrollü kullan" fikrinden
esinlenir, fakat BoltzGen'in residue kimliğini N/O sayısıyla kodlayan yöntemini
kopyalamaz. MIND'in anti-leakage amacıyla uyumlu olan fark budur.

Alternatif olarak virtual node'lar graph ve loss'tan tamamen çıkarılabilir.
Fakat bu durumda block atom sayısı tekrar residue/nucleotide kimliği hakkında
ipucu verir. Bu alternatif ancak leakage probe ile ölçülerek kabul edilmelidir.

## Uygulama sırası ve kabul kriterleri

### Aşama A - Veri kontratı

- Her protein block'u tam `4 + 10 = 14` slot olmalı.
- Her RNA block'u tam `12 + 11 = 23` slot olmalı.
- Backbone atomları canonical sırada ve her biri tam bir kez bulunmalı.
- Eksik/duplike backbone residue'leri filtrelenmeli veya karantinaya alınmalı.
- RNA ham-CIF manifest'i deterministik ve tekrar üretilebilir olmalı.

Mevcut RNA temp chunk'ında 116,175 block'un 2,409'u tam 23 atom değildir; bu
madde yeni temsil deneyinden önce çözülmelidir.

### Aşama B - Küçük yapısal testler

- Adapter başına en az 5 ham örnek.
- Bütün block'larda slot sayısı ve slot sırası assert'i.
- Virtual/marker ve gerçek atom sayılarının beklenen dağılımı.
- Koordinatlarda NaN/Inf olmaması.
- Masked-template yolunda sıfır mesafeli edge sayısının `0` olması.
- Aynı seed ile aynı şablon ve aynı cache hash'i.

### Aşama C - Leakage ve görev testleri

- Yalnız maskelenmiş block geometrisinden residue/nucleotide sınıflandıran basit
  probe, chance seviyesine yakın olmalı. Protein için chance yaklaşık `%5`, RNA
  için yaklaşık `%25`tir.
- Context eklendiğinde MLM öğrenebilmeli; yalnız slot düzeni cevabı vermemeli.
- Gerçek atom ve virtual ayrımı metrik callback'e doğru ulaşmalı.

### Aşama D - Eş koşullu fast A/B

Kontrol ve deney aynı seed, sampler, split, epoch ve atom bütçesini kullanmalı.
En az şu metrikler karşılaştırılmalı:

- RNA: overall, C1', backbone, BaseR, nucleotide all-or-nothing, V support/recall.
- Protein: N/CA/C/O, CB, sidechain-real, V ve per-AA real all-or-nothing.
- Her domain için MLM ve distance loss.
- Sıfır mesafeli edge, derece dağılımı ve batch atom dağılımı.

Yeni yaklaşım C1' stacking'deki büyük düşüşleri tekrarlamamalı. Protein CB
recall'u iyileşirken N/CA/C/O veya gerçek sidechain metrikleri bozulmamalıdır.

### Aşama E - İki epoch tam-veri doğrulaması

Fast A/B geçtikten sonra PDB + RNA + UniMol ile iki epoch çalıştırılmalı. Son
batch'lerde üç domain'in birlikte kaldığı ve per-domain loss'ların düzenli
azaldığı doğrulanmalıdır. Bundan önce production cache değiştirilmemelidir.

## Full train öncesi kalan engeller

Bu görev yalnız RNA virtual konumunu değiştirip bir fast run almak olarak
tanımlanırsa full train önündeki engeller bitmiş olmaz. Şunlar da kapanmalıdır:

1. **RNA 23-slot bütünlüğü:** 2,409 bozuk block politikası ve exact-23 assert'i.
2. **Protein virtual geometrisi:** RNA'da çalışan çözüm protein tarafında ayrıca
   A/B test edilmeli; CB problemi açık bırakılmamalı.
3. **Production cache:** Beş RNA chunk ve gerekiyorsa protein chunk'ları yeni
   kontratla yeniden üretilip metadata/audit yapılmalı.
4. **İki epoch multidomain doğrulama:** Yeni cache'lerle sampler, loss ve son
   batch kompozisyonu yeniden kontrol edilmeli.
5. **Production config temizliği:** `max_epochs`, run adı, atom bütçesi ve chunk
   aralıkları gerçek full-run değerlerine alınmalı. Mevcut multidomain config iki
   epochluk deney durumundadır.

Epoch-balanced sampler daha önce fast ve iki-epoch tam veri run'ında başarılı
oldu; bu artık açık bir engel değildir.

Yukarıdaki beş madde de "BoltzGen/MIND virtual temsil görevi" kapsamına dahil
edilip başarıyla tamamlanırsa **PDB + RNA + UniMol üç-domain full train önünde
bilinen başka kritik engel kalmaz**. Metabolit kapalı tutulabilir ve bu üç-domain
eğitimi engellemez. Metabolitin de full train'e katılması istenirse 497 chunk'ın
`bond_edge_index`/cross-modal collate davranışı ayrıca smoke test edilmelidir;
bu kısım henüz doğrulanmış sayılmamalıdır.

## Karar özeti

- BoltzGen'in virtual atomları ayrı noktalara dağıtılmıyor; N/O üzerine kodlu
  sayılarda bindiriliyor.
- Bu kod yalnız tasarlanan protein residue'leri için tanımlı; RNA tarifi yok.
- C1' stacking MIND RNA A/B testinde başarısız oldu.
- Literal BoltzGen portu adapter değişikliği değil, coordinate-diffusion ve
  geometric-decoding projesidir.
- Mevcut MIND full-pretraining hedefi için öneri, maskelenen block'lara
  tip-bağımsız ve çakışmasız local-frame slot şablonu uygulamaktır.
