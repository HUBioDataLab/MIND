# MIND — Son Durum ve Sonraki Hedefler

*Güncelleme: 17 Haziran 2026*

---

## Son Yapılanlar

| Commit | Açıklama |
|--------|---------|
| `dbfbc1b` | `.pkl → .pt` dönüşüm pipeline'ı 5-7× hızlandırıldı (vektörizasyon + ProcessPoolExecutor) |
| `b7f6fca` | Per-AA all-or-nothing tablo başlığı düzeltildi (virtual atomlar metriği katılaştırıyor, gevşetmiyor) |
| `603a7fd` | OXT/NH1/NH2 gibi terminal atomları sidechain-real metriğinden dışlandı (vocab membership check) |
| `5aa0443` | Multidomain dataset yolları düzeltildi (unimol, rna, metabolit); per-atom log etiketleri netleştirildi |
| `ad755f1` | RNA metrikleri protein tablosuna sızmıyordu, düzeltildi |

**100-epoch multidomain eğitimi tamamlandı** (pdb + unimol + rna, metabolit kapalı):
`logs/June2026/multidomain_100ep_pdb_unimol_rna_20260609_065900.log`

---

## Aktif Sorunlar

### CB Recall Çöküşü (~0.017–0.28)
Virtual atomlar (9 adet, her amino asit için) CB'nin tam koordinatına konuluyor. Block-MLM sırasında tip bilgisi maskeleniyor, pozisyon kalıyor → model "bu nokta genellikle V" öğreniyor → CB de V tahmin ediliyor. TRP'de sorun yok (10 gerçek sidechain atomi, virtual yok).

**Denenen ve başarısız olan**: 2Å offset fix — edge graph geometrisini bozdu (CG/N recall çöktü), revert edildi.

### Metabolit Devre Dışı
`bond_edge_index` anahtarı PyG cross-modal collator'ı crash ediyor. 497 chunk hazır ama kullanılamıyor.

### Epoch Dengesizliği
Protein/RNA verisi epoch ortasında bitince son batch'ler sadece UniMol'den oluşuyor.

---

## Sonraki Hedefler (Öncelik Sırasıyla)

### 1. RNA Metrik Yapısını Tamamla (Arya)

RNA için protein tarafındaki metrik yapısının karşılığı çıkarılacak:

- [ ] Per-atom classification metrikleri (her atom tipi için P/R/F1)
- [ ] Overall metrikler (tüm RNA atomları üzerinden aggregate)
- [ ] Backbone / backbone-dışı grup metrikleri
  - Backbone: P, O3', C3', C4', C5', O5', C1' (sugar+fosfat)
  - Non-backbone: baz atomları (N1, C2, N3, C4, C5, C6, N6, O6, N7, C8, N9, O2, N2, O4, N4, C2, ...)
- [ ] Nucleotide-bazlı all-or-nothing metrikler (A/U/G/C için tüm atomlar doğru mu?)
- [ ] Atom-level block metrikleri (per-position, RNA versiyonu)

Referans: `core/metrics/protein_per_atom_metrics_callback.py` — aynı yapı, RNA için uyarlanacak.

### 2. CB Sorunu — RNA Backbone Yaklaşımı

**Plan:**
1. RNA'da C1' (backbone) üzerine virtual yığma nasıl çalışıyor → metriklere bak
2. Protein'de virtual atomları CB yerine CA'ya yığ (CA her zaman gerçek, CB gibi "kurban" edilmez)
3. Fast train ile yeni CA metrikleri kontrol et
4. CA recall düştü ama CB kurtuldu mu? Trade-off değerlendir

Eğer çalışıyorsa: implement et, commit et.
Eğer çalışmıyorsa: BoltzGen / salad-style learned pseudo-atom yaklaşımına geç.

**Not:** Salad implementasyonu büyük mimari değişiklik gerektirir (yeni decoder head, FAPE loss, yeni data field). Kısa vadede önce backbone stacking'i dene.

### 3. Epoch Dengesizliği Düzelt

**Mevcut davranış:** Her domain kendi epoch sınırına göre dönüyor. Protein/RNA verisi bitince sadece UniMol kalıyor.

**Hedef:** Epoch kavramını kaldır, tüm domain'lerden baştan interleave et. Her batch'te her domain'den örnek olsun.

Değişiklik yeri: `data_loading/` — multi-domain sampler mantığı.

### 4. Amino Asit Metrikleri (Mehmet Ali'den)

Yeni metrikler Mehmet Ali'den alınacak, entegre edilecek.

### 5. Genel Metrik Özeti

Tüm atom tipleri dahil edilen aggregate bir metrik çıktısı isteniyor — şu an per-atom tablolar ayrı ayrı print ediliyor. Bir "overall" satırı veya özet tablo eklenecek.

---

## Konfigürasyon Referansı

| Config | Dosya | Kullanım |
|--------|-------|---------|
| Fast sanity check | `core/pretraining_config_fast.yaml` | `make train CONFIG=f LOG=test_name` |
| Multidomain tam eğitim | `core/pretraining_config_multidomain.yaml` | `make train CONFIG=m LOG=run_name` |
| GPU otomatik seçim | — | `make train GPU=auto CONFIG=m` |

Loglar: `logs/<MonthYear>/<name>_<YYYYMMDD_HHMMSS>.log` (multidomain) veya `<name>fast_<YYYYMMDD_HHMMSS>.log` (fast).
