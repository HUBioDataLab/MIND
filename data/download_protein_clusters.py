# data/download_protein_clusters.py
# source /opt/anaconda3/bin/activate

import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import os

# İndirilecek dosyanın URL'si ve varsayılan adı
METADATA_URL = "https://afdb-cluster.steineggerlab.workers.dev/2-repId_isDark_nMem_repLen_avgLen_repPlddt_avgPlddt_LCAtaxId.tsv.gz"
DEFAULT_FILENAME = "representatives_metadata.tsv.gz"

def download_file(url: str, out_path: Path):
    """
    Verilen URL'den bir dosyayı stream ederek ve ilerleme çubuğu göstererek indirir.
    Dosya zaten varsa ve boyutu doğruysa indirmeyi atlar.
    """
    print(f"İndirme hedefi: {out_path}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # HTTP 4xx/5xx hataları için exception fırlatır
        
        total_size = int(response.headers.get('content-length', 0))

        # Dosya zaten varsa ve boyutu eşleşiyorsa indirmeyi atla
        if out_path.exists() and out_path.stat().st_size == total_size:
            print(f"Dosya '{out_path.name}' zaten mevcut ve boyutu doğru. İndirme atlandı.")
            return

        print(f"İndiriliyor: {url}")
        with open(out_path, 'wb') as f, tqdm(
            desc=out_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024*8):
                size = f.write(chunk)
                pbar.update(size)
        
        print(f"İndirme tamamlandı: {out_path}")

    except requests.exceptions.RequestException as e:
        print(f"HATA: Dosya indirilemedi. İnternet bağlantınızı veya URL'yi kontrol edin.\n{e}")
        # İndirme başarısız olursa yarım kalan dosyayı sil
        if out_path.exists():
            out_path.unlink()
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")
        if out_path.exists():
            out_path.unlink()


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description="Foldseek'ten önceden hesaplanmış küme temsilcisi metadata'sını indirir."
    )
    # Proje ana dizininden çalıştırıldığını varsayarak varsayılan yolu belirliyoruz.
    default_path = Path(__file__).resolve().parents[3] / "data" / "proteins" / "afdb_clusters"
    parser.add_argument(
        "--outdir",
        type=Path,
        default=default_path,
        help=f"İndirilen dosyanın kaydedileceği klasör. Varsayılan: {default_path}"
    )
    args = parser.parse_args()

    # Çıktı klasörünü oluştur
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Tam dosya yolunu oluştur
    output_file_path = args.outdir / DEFAULT_FILENAME
    
    # İndirme işlemini başlat
    download_file(METADATA_URL, output_file_path)


if __name__ == "__main__":
    main()