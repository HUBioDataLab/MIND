# CB atomunun düşük skoru — basit anlatım ve tüm senaryolar

> Amaç: "Protein metriklerinde CB atomunun recall'ı neden çok düşük (~0.28),
> diğer bütün atomlar ~0.99 iken?" sorusunu en sade dille açıklamak; bunun bir
> model hatası mı yoksa bir temsil (representation) seçiminin yan etkisi mi
> olduğunu netleştirmek; ve CB ile ilgili kararı vermeden önce **bütün
> seçenekleri** önümüze koymak.

---

## 1. Tek cümlede

CB düşük çünkü onu **sahte (virtual) dolgu atomlarıyla tam aynı 3B noktaya**
koyduk; atom tipini maskeleyince model gerçek CB ile sahte atomları
birbirinden ayıramıyor ve "bu nokta zaten hep sahte" diyip CB'ye de "sahte"
(V) tahmini yapıyor.

---

## 2. Önce resmi çizelim (çocuk diliyle)

Bir proteini **boncuklardan** yapılmış bir kolye gibi düşün. Her amino asit
(harf) birkaç **atom** boncuğundan oluşuyor. Bizim modelimiz her amino aside
**tam 14 boncukluk sabit bir kutu** veriyor:

- **4 tanesi her zaman var**: N, CA, C, O (omurga = backbone).
- **10 tanesi yan-zincir (sidechain)** için ayrılmış slot.

Ama her amino asidin yan zinciri aynı büyüklükte değil:
- ALA (alanin) sadece **1** gerçek yan-zincir atomu taşır → adı **CB**.
- TRP (triptofan) **10** gerçek yan-zincir atomu taşır → kutusu tıka basa dolu.

Kutuyu hep 14'e tamamlamak için **boş slotları "sahte atom" (virtual, tip
'V') ile dolduruyoruz.** İşte kritik nokta: **bu sahte atomları nereye
koyduk?** → Hepsini **CB'nin tam üstüne** (aynı x,y,z koordinatına) koyduk.

Yani ALA'da o noktada: **1 gerçek CB + 9 sahte atom**, hepsi üst üste.

---

## 3. Sorun tam olarak nasıl çıkıyor?

Eğitimde modele şunu öğretiyoruz (MLM = "maskeli atom tahmini", BERT'in
atom versiyonu): bazı atomların **tipini gizliyoruz** (maskeliyoruz) ve model
o atomın ne olduğunu tahmin etmeye çalışıyor.

Önemli detay: maskeleme sadece **atom tipini** gizler, **konumu (3B yeri)
gizlemez**. Yani modelin bir atomu tanımak için elinde kalan tek ipucu çoğu
zaman **konumu**.

Şimdi CB'ye bak:
- CB'nin konumu = o noktanın koordinatı.
- Ama **9 sahte atom da tam aynı konumda.**
- Maskeleyince hepsi birbirinin aynısı görünüyor (aynı yer, aynı maske, aynı
  komşular).

Model "bu noktadaki atomların %90'ı sahte, o zaman buna da sahte (V) derim"
diyor. Sonuç: gerçek CB'ye "V" diyor → **CB recall çöküyor (~0.02–0.28).**

**Kanıt — TRP istisnası:** TRP'nin 10 yan-zincir atomu da gerçek, yani
kutusunda **hiç sahte atom yok**, CB tek başına o konumda. Bu yüzden TRP'nin
CB'si gayet iyi tahmin ediliyor. Tam da teorinin söylediği şey.

---

## 4. Neden sadece proteinde? RNA'da niye yok?

RNA adapter tasarımı da aynı "sabit kutu + sahte atomla doldurma" mantığını
öngörüyor ve sahte atomları **C1' (şeker halkasının bağlanma noktası)** üstüne
koyuyor. Ancak production RNA cache'i bu adapter değişikliğinden önce üretildiği
için aşağıdaki davranış yalnız geçici A/B cache'inde gerçekten mevcuttu:

| | Protein | RNA C1' A/B deneyi |
|---|---|---|
| Sahte atomlar nereye yığılı | **CB** | **C1'** |
| Anchor ne tür | **Yan-zincir** (GLY'de yok) | **Backbone** (şeker atomu) |
| Sonuç | CB recall çöküyor | C1' ve genel RNA metrikleri birlikte düşüyor |

RNA production cache'inde gerçekte virtual atom yoktu. Bunu fark ettikten sonra
aynı RNA chunk'ının virtual olmayan kontrolü ile C1' üzerine virtual eklenmiş
geçici kopyası üç epoch boyunca eş koşullarda karşılaştırıldı:

| Metrik | Virtual yok | C1' üzerine virtual |
|---|---:|---:|
| RNA overall accuracy | 0.7252 | 0.5569 |
| C1' recall | 0.9414 | 0.8043 |
| Backbone recall | 0.8665 | 0.7106 |
| BaseR accuracy | 0.5459 | 0.4613 |
| Nucleotide all-or-nothing | 0.3895 | 0.1802 |

C1' her nucleotide'de gerçek olsa bile aynı noktadaki çok sayıda virtual node
radius graph'a sıfır mesafeli edge olarak girdi. Deney cache'inde 868,574 böyle
edge bulundu. Dolayısıyla "backbone anchor her zaman güvenlidir" varsayımı doğru
değildir; stacking RNA'da da gerçek atom ve genel temsil kalitesini bozdu.

BoltzGen'in N/O marker kodlamasıyla bunun neden aynı yöntem olmadığı ve MIND'e
uyarlama seçenekleri
[`BOLTZGEN_GEOMETRIC_RESIDUE_ENCODING.md`](features/BOLTZGEN_GEOMETRIC_RESIDUE_ENCODING.md)
dosyasında açıklanmıştır.

---

## 5. Salad makalesi bundan bahsediyor mu? Biz salad'a uygun mu yaptık?

Salad (Jendrusch & Korbel, *Nature Machine Intelligence* 2025) her amino asidi
şöyle temsil ediyor (makale s. 1431):

> "...the backbone atoms (N, CA, C and O), **an idealized beta-carbon (CB)**
> and additional **learned pseudo-atoms**."

Yani salad'da:
- **CB her zaman var ve birinci sınıf bir atom** (idealize edilmiş, gerçek bir
  nokta). Asla dolgu değil.
- Ek "pseudo-atomlar" var ama bunlar **öğrenilmiş, ayrı noktalar** — CB'nin
  üstüne yığılmış kopyalar değil. Yan zincirleri ise ayrı bir **all-atom
  decoder** üretiyor.

**Biz salad'dan ayrıldık (kendi seçimimiz):** Salad gibi "CB'yi her zaman
gerçek tut + pseudo-atomları öğren" yerine, **sabit 14-atomluk bir kutu** kurup
boş slotları **CB'nin üstüne yığılan jenerik 'V' atomlarıyla** doldurduk.
Yani CB çöküşü **salad'ın bir sorunu değil; bizim temsil sadeleştirmemizin yan
etkisi.** Salad'ın tasarımında bu sorun ortaya çıkmazdı.

---

## 6. En baştan virtual atomları neden ve nasıl ekledik?

İki sebeple:
1. **Sabit şekil:** Transformer/attention'ın ve batch'lemenin kolay olması için
   her residünün **tam 14 atomu** olsun istedik (değişken uzunluk yerine sabit).
2. **Sızıntıyı önleme (anti-leakage):** Eğer sahte atomlara gerçek tip/sayı
   bilgisi verseydik, model "kaç gerçek yan-zincir atomu var" sayısından amino
   asidin kim olduğunu **kopya çekebilirdi**. Bunu engellemek için tüm sahte
   atomlara aynı jenerik 'V' tipini verdik ve **konum olarak da CB'yi seçtik**
   (kolay bir varsayılan — "bir yere koymak lazım, CB olsun").

İşte o "kolay varsayılan" (hepsini CB'ye koymak) CB çöküşünü doğurdu.
Sızıntıyı tip ile zaten engelliyorduk; **konumu CB yapmak gereksizdi ve zararlı
çıktı.**

---

## 7. Denediğimiz çözüm neden çalışmadı?

Denediğimiz şey: sahte atomları CB'nin üstünden alıp, CB çevresinde **2 Å'lık
küçük bir küreye dağıtmak** (her sahte atom ayrı bir noktaya). Mantık: "CB
artık tek başına kalsın, ayırt edilebilsin."

Sonuç (400 protein, A/B testi): **işe yaramadı, hatta zarar verdi.**

| Atom | Eski (çakışık) | Yeni (2 Å dağıtık) |
|---|---|---|
| CG | 0.99 | **0.63** ⬇️ |
| N | 0.63 | **0.00** ⬇️ |
| CB | 0.05 | 0.12 (≈ aynı) |
| Eğitim loss | 3.43 | **4.20** ⬆️ (daha kötü) |

**Neden:** Sahte atomlar **edge (komşuluk) grafiğine giriyor** —
filtrelenmiyorlar. Onları yapının içine 2 Å saçınca, her residüye **9 sahte
atom + 9 sahte kenar** ekledik. Bu, modelin gördüğü 3B geometriyi kirletti →
**CB biraz oynadı ama diğer her şey bozuldu.** Yani konum-kaydırma kökten
yanlış yaklaşım: sorunu çözmüyor, yeni gürültü ekliyor.

(İyi haber: bu testi 400 proteinde ~yarım saatte yaptık ve bizi saatler sürecek
boşuna bir tam-veri yeniden-üretiminden kurtardı.)

---

## 8. Peki ben bu düşük CB'yi nasıl açıklayacağım?

Şöyle de: **"CB recall'ı düşük çünkü bu bir model kalitesi sorunu değil, bir
temsil (representation) seçiminin ölçüm artefaktı."**

- Modelimiz aslında her şeyi öğreniyor — CG, CD, CE, azot, oksijen, kükürt
  atomlarının hepsi ~0.99.
- CB'yi "kaybetmesinin" sebebi, CB'yi sahte dolgu atomlarıyla aynı yere
  koymuş olmamız. Yani **biz** modele CB'yi ayırt etmeyi imkansız kıldık.
- Kimyasal olarak CB zaten **trivial**: her amino asitte (GLY hariç) var ve
  omurgaya göre neredeyse sabit bir yerde durur. Yani CB'yi "V" tahmin etmek
  modelin temsil gücünü düşürmüyor; sadece **metrik tablosu kötü görünüyor.**

Tek cümlelik mülakat cevabı: *"O düşük sayı gerçek bir başarısızlık değil;
dolgu atomlarını CB'nin üstüne koyduğumuz için ortaya çıkan, bilinçli bir
temsil sadeleştirmesinin bilinen bir yan etkisi."*

---

## 9. Gerçekten çözümü yok mu? — TÜM SENARYOLAR

CB'yi düzeltmenin tek bir "doğru" yolu yok; her birinin maliyet/risk dengesi
farklı. Kararı sen vereceksin diye hepsini koyuyorum.

### Senaryo A — Kabul et ve dokümante et (en ucuz, sıfır risk)
CB düşüklüğünü olduğu gibi bırak, raporlarda dipnotla açıkla (yukarıdaki §8).
- **Artı:** Hiç iş yok, model bozulmaz.
- **Eksi:** Metrik tablosu "çirkin" kalır; dışarıdan bakana açıklama gerektirir.
- **Ne zaman:** Foundation-model temsili önemliyse ve CB metriği bir
  "vitrin sayısı" değilse.

### Senaryo B — Metriği düzelt (model aynı, sadece raporlama)
CB'yi yan-zincir metriklerinden çıkar / ayrı göster; "Real-Acc" gibi özet
skorları CB'yi hariç tutarak da raporla, ya da CB'yi "bilinen artefakt" diye
ayrı bir satıra al.
- **Artı:** Çok küçük kod değişikliği, reprocess yok, eğitim yok. Sayılar
  birden "düzgün" görünür.
- **Eksi:** Gerçek temsili değiştirmez — sadece dürüst bir raporlama. (Bunu
  "sayıyı saklamak" gibi değil, "bilinen artefaktı işaretlemek" gibi sun.)
- **Ne zaman:** Hızlı, dürüst bir "tablo temizliği" istiyorsan.

### Senaryo C — Sahte atomları edge grafiğinden çıkar (prensipli orta yol)
Sahte atomlar CB konumunda kalsın AMA **hiç kenarı olmasın** (saf dolgu
düğümü). O zaman maskeli CB'nin **komşuları var**, sahte atomların **yok** →
model ikisini bağlantıdan ayırt edebilir; geometri de kirlenmez (Senaryo 7'deki
hatayı yapmaz).
- **Artı:** §7'deki geometri bozulmasını yaşamaz; mantıklı.
- **Eksi:** `cache_to_pyg` (edge kurma) değişikliği + protein verisini yeniden
  üretmek gerekir. **Ayrıca olası sızıntı:** kenar yapısı maskelenmediği için
  model "kaç tane kenarsız düğüm var" → yan-zincir sayısı → kısmi amino asit
  kimliği çıkarabilir. Denemeden çalışıp çalışmayacağı belli olmaz (yine küçük
  bir A/B testi gerekir, ~1 saat).
- **Ne zaman:** CB'yi gerçekten kurtarmak istiyorsan ve bir deney turu daha
  göze alıyorsan.

### Senaryo D — Salad gibi yap (en doğru, en pahalı)
Sahte-dolgu fikrini bırak: **CB'yi her zaman gerçek, birinci-sınıf bir atom
yap**; yan-zincirleri sabit slot yerine salad'daki gibi **öğrenilmiş
pseudo-atom / all-atom decoder** ile temsil et.
- **Artı:** Sorunu kökten bitirir, literatürle (salad) hizalı.
- **Eksi:** Temsil + model + veri üretimi büyük değişiklik; en uzun yol.
- **Ne zaman:** Uzun vadede "doğru" mimariye geçmek istiyorsan.

### Senaryo E — Sahte/gerçek bilgisini maskeleme dışında tut
Maskelerken "bu slot gerçek mi sahte mi" bilgisini gizleme (tipi gizle ama
gerçek-mi-sahte bayrağını ver). O zaman model maskeli CB'nin **gerçek** olduğunu
bilir → ona gerçek bir element tahmin eder.
- **Artı:** Geometriyi hiç bozmaz, CB kurtulur.
- **Eksi:** **Sayı sızıntısı geri gelir** (kaç gerçek atom var → amino asit
  ipucu). Tam da §6'da kaçındığımız şey. Sızıntının zararı ölçülmeli.
- **Ne zaman:** Sayı-sızıntısının senin downstream görevin için zararsız
  olduğunu düşünüyorsan.

---

## 10. CB'yi düzeltmezsek accuracy/recall düşük gelecek — bunu nasıl yönetiriz?

CB'yi düzeltmesek bile rapor edilen sayıları **dürüstçe ve doğru** sunmanın
yolları (Senaryo B'nin pratik hali):

1. **CB'yi ayrı işaretle:** Per-atom tablosunda CB satırına "(bilinen temsil
   artefaktı)" notu düş. Okuyan kişi onu gerçek başarısızlık sanmaz.
2. **CB-hariç özet ver:** Yan-zincir (SideChain) grup recall'ını ve "Real-Acc"
   özetini **bir de CB'yi çıkararak** raporla. Örnek: SideChain recall CB
   yüzünden 0.82 görünüyor; CB hariç ~0.99'a çıkıyor. İkisini de göster.
3. **Asıl mesajı vurgula:** "CB dışındaki 33 gerçek atomun hepsi ~0.99;
   backbone ~1.0." Tablo bir anda çok daha sağlıklı okunur.
4. **Per-AA all-or-nothing metriğinde:** CB neredeyse her residüde tek başına
   patladığı için "Real-Acc" düşük çıkıyor. CB düzelirse bu skor ~0.32'den
   ~0.85+'a fırlar (CB tek darboğaz). Raporda bunu "CB düzeltilirse beklenen
   tavan" olarak belirtebilirsin.

> Kısaca: CB'yi düzeltmesek bile, **"CB-dahil" ve "CB-hariç" iki kolon**
> göstererek hem dürüst hem de güçlü bir tablo sunabilirsin.

---

## Karar için özet tablo

| Senaryo | İş yükü | Risk | Sonuç |
|---|---|---|---|
| A — Kabul + dokümante | Yok | Yok | Sayı düşük kalır, açıklanır |
| B — Metrik/raporlama düzelt | Çok az | Yok | Tablo düzgün görünür (model aynı) |
| C — Sahte atomu edge'den çıkar | Orta (+reprocess +A/B) | Sızıntı? | CB kurtulabilir |
| D — Salad gibi (CB gerçek + decoder) | Büyük | Düşük | Kökten çözüm |
| E — Gerçek/sahte bayrağı verme | Az | Sayı sızıntısı | CB kurtulur ama sızıntı |

**Önerilen pratik yol:** Önce **B** (dürüst raporlama, hemen) + kararını
verene kadar **A**. CB'yi gerçekten kurtarmak istersen sıradaki ucuz deney
**C** (küçük A/B ile test edilebilir). Uzun vade için doğru hedef **D**.
