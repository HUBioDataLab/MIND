import os
import json
import shutil
from tqdm import tqdm

# --- AYARLAR ---
SOURCE_JSON = "rna3db-jsons/filter.json"  # İndirdiğin filter.json yolu
SOURCE_DIR = "raw_structures"             # Mevcut .cif dosyalarının olduğu yer
TARGET_DIR = "filtered_structures"        # Yeni dosyaların gideceği yer
OUTPUT_JSON = "filtered_raw_structures.json" # Yeni oluşturulacak özet dosya
MIN_LENGTH = 20                           # Minimum nükleotid sayısı
MAX_LENGTH = 5000                         # Maksimum nükleotid sayısı (YENİ)

# 1. Kaynak JSON'ı Yükle
print(f"Loading metadata from {SOURCE_JSON}...")
with open(SOURCE_JSON, 'r') as f:
    full_data = json.load(f)

# 2. Hedef klasörü oluştur
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)
    print(f"Created directory: {TARGET_DIR}")

filtered_data = {}
files_to_copy = []

# 3. Filtreleme Mantığı
print(f"Filtering structures ({MIN_LENGTH} <= Length <= {MAX_LENGTH})...")

# Mevcut dosyaları kontrol et (sadece indirmiş olduklarını listeye al)
available_files = set(os.listdir(SOURCE_DIR))

for db_id, info in full_data.items():
    filename = f"{db_id}.cif"
    
    # Kriter 1: Dosya diskte var mı?
    if filename not in available_files:
        continue
        
    # Kriter 2: Uzunluk kontrolü (Min ve Max aralığı)
    length = info.get('length', 0)
    
    if MIN_LENGTH <= length <= MAX_LENGTH:
        filtered_data[db_id] = info
        files_to_copy.append(filename)

print(f"Found {len(files_to_copy)} valid structures.")

# 4. Dosyaları Kopyala
print(f"Copying files to {TARGET_DIR}...")
for filename in tqdm(files_to_copy):
    src = os.path.join(SOURCE_DIR, filename)
    dst = os.path.join(TARGET_DIR, filename)
    shutil.copy2(src, dst)

# 5. Yeni JSON dosyasını kaydet
print(f"Saving filtered metadata to {OUTPUT_JSON}...")
with open(OUTPUT_JSON, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print("\n--- Process Complete ---")
print(f"Filtered files count: {len(filtered_data)}")
print(f"New data folder: {TARGET_DIR}")
print(f"New metadata file: {OUTPUT_JSON}")