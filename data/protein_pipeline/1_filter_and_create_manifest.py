# data/protein_pipeline/1_filter_and_create_manifest.py

"""
python data/protein_pipeline/1_filter_and_create_manifest.py \
    --mode manifest \
    --metadata-file ../data/proteins/afdb_clusters/representatives_metadata.tsv.gz \
    --target-count 40000 \
    --output ../data/proteins/afdb_clusters/manifest_hq_40k.csv \
    --existing-structures-dir ../data/proteins/raw_structures_hq_40k \
    --plddt 70 \
    --max-len 512
"""

"""
    python data/protein_pipeline/1_filter_and_create_manifest.py \
    --mode analyze \
    --metadata-file ../data/proteins/afdb_clusters/representatives_metadata.tsv.gz
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import os

# AlphaFold DB URL şablonları
PDB_URL_TEMPLATE = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"
CIF_URL_TEMPLATE = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.cif"

def get_existing_ids(structures_dir: Path) -> set:
    """Verilen klasördeki mevcut PDB/CIF dosyalarından UniProt ID'lerini çıkarır."""
    if not structures_dir.is_dir():
        return set()
    
    existing_ids = set()
    print(f"Mevcut yapılar taranıyor: {structures_dir}")
    # .pdb ve .cif uzantılı tüm dosyaları tara
    for f in structures_dir.glob('*.pdb'):
        existing_ids.add(f.stem)
    for f in structures_dir.glob('*.cif'):
        existing_ids.add(f.stem)
        
    print(f"-> {len(existing_ids):,} adet yapı zaten mevcut.")
    return existing_ids

def analyze_metadata(metadata_path: Path):
    """Metadata dosyasını okur ve pLDDT, uzunluk ve birleşik filtrelerin dağılımını analiz eder.""" # GÜNCELLENDİ
    print(f"Metadata dosyası okunuyor ve analiz ediliyor: {metadata_path}")
    try:
        df = pd.read_csv(metadata_path, sep='\t', header=None, compression='gzip')
        df.columns = ['repId', 'isDark', 'nMem', 'repLen', 'avgLen', 'repPlddt', 'avgPlddt', 'LCAtaxid']
    except Exception as e:
        print(f"HATA: Dosya okunamadı. \n{e}")
        return

    total_representatives = len(df)
    print(f"\nToplam Temsilci Protein Sayısı: {total_representatives:,}")

    # --- pLDDT Skor Dağılım Analizi ---
    print(f"\n--- pLDDT Skor Dağılım Analizi ---")
    plddt_thresholds = [90, 80, 70, 60, 50]
    for threshold in plddt_thresholds:
        count = np.sum(df['repPlddt'] >= threshold)
        percentage = (count / total_representatives) * 100
        print(f"pLDDT >= {threshold}: {count:10,} adet protein ({percentage:.2f}%)")
    print("-------------------------------------\n")

    # --- Protein Uzunluk (repLen) Dağılım Analizi ---
    print(f"--- Protein Uzunluk (repLen) Dağılım Analizi ---")
    len_thresholds = [100, 200, 300, 350, 400, 500, 750, 1024]
    for threshold in len_thresholds:
        count = np.sum(df['repLen'] <= threshold)
        percentage = (count / total_representatives) * 100
        print(f"Uzunluk <= {threshold:<4}: {count:10,} adet protein (Toplamın %{percentage:.2f}'si)")
    print("-------------------------------------\n")
    
    # --- YENİ EKLENDİ: Birleşik Filtre Analizi (pLDDT ve Uzunluk) ---
    print(f"--- Birleşik Filtre Analizi (pLDDT > 70) ---")
    plddt_filter = 70
    # Önerdiğiniz ve ek olarak birkaç farklı uzunluk eşiği
    combined_len_thresholds = [256, 300, 350, 384, 512, 768, 1024]
    
    # Sadece pLDDT > 70 olanların sayısı (karşılaştırma için)
    base_count = np.sum(df['repPlddt'] >= plddt_filter)
    
    for threshold in combined_len_thresholds:
        # İki koşulu birden sağlayan proteinlerin sayısını buluyoruz
        count = np.sum((df['repPlddt'] >= plddt_filter) & (df['repLen'] <= threshold))
        
        # Yüzdeyi toplam veri setine ve pLDDT>70 olanlara göre hesaplayalım
        percentage_total = (count / total_representatives) * 100
        percentage_base = (count / base_count) * 100 if base_count > 0 else 0
        
        print(f"pLDDT >= {plddt_filter} VE Uzunluk <= {threshold:<4}: {count:10,} adet protein (Toplamın %{percentage_total:.2f}'si, pLDDT>70 olanların %{percentage_base:.2f}'si)")
    print("-------------------------------------\n")

def create_manifest(metadata_path: Path, target_count: int, output_path: Path, existing_structures_dir: Path, plddt_threshold: int = 70, max_len: int = 350):
    """
    Mevcut dosyaları kontrol ederek bir indirme manifestosu oluşturur.
    """
    # Önce mevcut dosyaların ID'lerini bir sete al
    existing_ids = get_existing_ids(existing_structures_dir)
    
    print(f"Metadata dosyası okunuyor: {metadata_path}")
    try:
        df = pd.read_csv(metadata_path, sep='\t', header=None, compression='gzip')
        df.columns = ['repId', 'isDark', 'nMem', 'repLen', 'avgLen', 'repPlddt', 'avgPlddt', 'LCAtaxid']
    except Exception as e:
        print(f"HATA: Dosya okunamadı. \n{e}")
        return

    # Filter by pLDDT and sequence length
    print(f"Filtreleme yapılıyor: pLDDT >= {plddt_threshold} VE Sekans Uzunluğu <= {max_len}...")
    hq_df = df[(df['repPlddt'] >= plddt_threshold) & (df['repLen'] <= max_len)].sort_values(by='repPlddt', ascending=False)
    
    # Mevcut olan ID'leri aday listesinden çıkar
    if existing_ids:
        print(f"Mevcut {len(existing_ids):,} ID, aday listesinden çıkarılıyor...")
        hq_df = hq_df[~hq_df['repId'].isin(existing_ids)]

    total_hq_candidates = len(hq_df)
    print(f"İndirilecek {total_hq_candidates:,} adet yeni, kriterlere uygun aday bulundu.")

    if total_hq_candidates < target_count:
        print(f"UYARI: Hedeflenen {target_count:,} yeni protein sayısına ulaşılamadı. Bulunan {total_hq_candidates:,} adet ile devam ediliyor.")
        final_df = hq_df
    else:
        final_df = hq_df.head(target_count)

    print(f"Manifest için {len(final_df):,} adet yeni protein seçildi. URL'ler oluşturuluyor...")
    
    final_df['pdb_url'] = final_df['repId'].apply(lambda x: PDB_URL_TEMPLATE.format(x))
    final_df['cif_url'] = final_df['repId'].apply(lambda x: CIF_URL_TEMPLATE.format(x))
    
    manifest_cols = ['repId', 'repPlddt', 'repLen', 'pdb_url', 'cif_url']
    final_df[manifest_cols].to_csv(output_path, index=False)
    
    print(f"\nManifest dosyası başarıyla oluşturuldu: {output_path}")
    print(f"-> Bu manifest, indirilecek {len(final_df):,} YENİ protein içermektedir.")

def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description="Metadata'yı analiz eder ve mevcut dosyaları hariç tutarak indirme manifestosu oluşturur."
    )
    parser.add_argument("--metadata-file", type=Path, required=True, help="...")
    parser.add_argument("--mode", type=str, choices=['analyze', 'manifest'], required=True, help="...")
    parser.add_argument("--target-count", type=int, help="'manifest' modu için hedef protein sayısı.")
    parser.add_argument("--output", type=Path, help="'manifest' modu için çıktı CSV dosya yolu.")
    parser.add_argument("--plddt", type=int, default=70, help="Minimum pLDDT eşiği.")
    parser.add_argument("--max-len", type=int, default=512, help="Maksimum sekans (amino asit) uzunluğu.")
    parser.add_argument(
        "--existing-structures-dir",
        type=Path,
        help="'manifest' modu için: mevcut PDB/CIF'lerin bulunduğu klasör."
    )
    
    args = parser.parse_args()

    if args.mode == 'analyze':
        analyze_metadata(args.metadata_file)
    elif args.mode == 'manifest':
        if not all([args.target_count, args.output, args.existing_structures_dir]):
            parser.error("'manifest' modu için --target-count, --output ve --existing-structures-dir zorunludur.")
        create_manifest(args.metadata_file, args.target_count, args.output, args.existing_structures_dir, args.plddt, args.max_len)

if __name__ == "__main__":
    main()