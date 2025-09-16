#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# source /opt/anaconda3/bin/activate
""" 
python data_loading/proteins/prepare_dataset.py \
    --manifest-file ../data/proteins/afdb_clusters/manifest_hq_40k.csv \
    --structures-outdir ../data/proteins/raw_structures_hq_40k \
    --workers 16
"""

import argparse
from pathlib import Path
import time
from typing import Tuple, Optional

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------- #
# İndirme yardımcıları 
# ---------------------------- #

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
}

def is_html_error(payload: str) -> bool:
    """İndirilen metin HTML hata sayfası gibi mi? Çok kaba bir kontrol."""
    lower = payload.strip().lower()
    return lower.startswith("<!doctype html") or lower.startswith("<html")

def download_text(url: str, timeout: int = 60, max_retries: int = 3) -> Optional[str]:
    """Basit GET indirme (metin). Hata durumunda birkaç kez tekrar dener."""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            if r.status_code == 200:
                text = r.text
                if text and not is_html_error(text):
                    return text
            # 200 değilse veya hata varsa kısa bekleyip tekrar dene
        except requests.exceptions.RequestException:
            pass # Hata durumunda döngü devam edecek ve tekrar deneyecek
        time.sleep(1.25 * attempt)
    return None

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')

def download_one_structure(uniprot_id: str,
                           pdb_url: str,
                           cif_url: str,
                           out_dir: Path) -> Tuple[str, str]:
    """Tek bir protein için PDB indir; olmazsa CIF indir."""
    pdb_path = out_dir / f"{uniprot_id}.pdb"
    cif_path = out_dir / f"{uniprot_id}.cif"

    # Dosya zaten varsa indirmeyi atla
    if pdb_path.exists() or cif_path.exists():
        return uniprot_id, "SKIPPED_EXIST"

    # Önce PDB
    text = download_text(pdb_url)
    if text:
        save_text(pdb_path, text)
        return uniprot_id, "PDB_OK"

    # Fallback: CIF
    text = download_text(cif_url)
    if text:
        save_text(cif_path, text)
        return uniprot_id, "CIF_OK"

    return uniprot_id, "FAILED"

def parallel_download_from_manifest(manifest_csv: Path,
                                    out_dir: Path,
                                    max_workers: int = 8) -> dict:
    """Manifest CSV'den paralel indirme yap."""
    try:
        df = pd.read_csv(manifest_csv)
    except Exception as e:
        print(f"HATA: Manifest dosyası okunamadı: {manifest_csv}\n{e}")
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(df):,} adet yapı indirilecek. Paralel çalışan sayısı: {max_workers}")

    stats = {"PDB_OK": 0, "CIF_OK": 0, "FAILED": 0, "SKIPPED_EXIST": 0, "TOTAL": len(df)}
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(download_one_structure, row.repId, row.pdb_url, row.cif_url, out_dir): row.repId
            for row in df.itertuples(index=False)
        }
        
        # İlerleme çubuğu için tqdm kullanımı
        from tqdm import tqdm
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Proteinler indiriliyor")
        for fut in pbar:
            uid, status = fut.result()
            stats[status] = stats.get(status, 0) + 1
            # İlerleme çubuğunda anlık durumu göster
            pbar.set_postfix(
                PDB=stats['PDB_OK'],
                CIF=stats['CIF_OK'],
                FAIL=stats['FAILED'],
                SKIP=stats['SKIPPED_EXIST']
            )
            
    return stats

# ---------------------------- #
# Argparse & main
# ---------------------------- #

def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description="Manifest dosyasından PDB/CIF yapılarını paralel olarak indirir."
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        required=True,
        help="İndirilecek proteinleri listeleyen manifest CSV dosyasının yolu."
    )
    parser.add_argument(
        "--structures-outdir",
        type=Path,
        required=True,
        help="İndirilen PDB/CIF dosyalarının kaydedileceği klasör."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Paralel indirme için kullanılacak iş parçacığı sayısı."
    )
    args = parser.parse_args()

    print("--- Paralel İndirme Script'i Başlatıldı ---")
    start_time = time.time()
    
    stats = parallel_download_from_manifest(
        manifest_csv=args.manifest_file,
        out_dir=args.structures_outdir,
        max_workers=args.workers,
    )
    
    end_time = time.time()
    print("\n--- İndirme İşlemi Tamamlandı ---")
    print(f"Toplam Süre: {end_time - start_time:.2f} saniye")
    print("Sonuç Özeti:")
    import json
    print(json.dumps(stats, indent=2))
    print(f"Yapılar şu klasöre kaydedildi: {args.structures_outdir}")

if __name__ == "__main__":
    main()