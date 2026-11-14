import os
import json
import shutil
from tqdm import tqdm

# --- SETTINGS ---
SOURCE_JSON = "rna3db-jsons/filter.json"  # path of the downloaded filter.json
SOURCE_DIR = "raw_structures"             # directory holding the existing .cif files
TARGET_DIR = "filtered_structures"        # directory for the filtered copies
OUTPUT_JSON = "filtered_raw_structures.json" # summary file to be created
MIN_LENGTH = 20                           # minimum number of nucleotides
MAX_LENGTH = 5000                         # maximum number of nucleotides

# 1. Load the source JSON
print(f"Loading metadata from {SOURCE_JSON}...")
with open(SOURCE_JSON, 'r') as f:
    full_data = json.load(f)

# 2. Create the target directory
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)
    print(f"Created directory: {TARGET_DIR}")

filtered_data = {}
files_to_copy = []

# 3. Filtering logic
print(f"Filtering structures ({MIN_LENGTH} <= Length <= {MAX_LENGTH})...")

# Check the existing files (keep only the ones that were downloaded)
available_files = set(os.listdir(SOURCE_DIR))

for db_id, info in full_data.items():
    filename = f"{db_id}.cif"
    
    # Criterion 1: does the file exist on disk?
    if filename not in available_files:
        continue
        
    # Criterion 2: length check (min and max range)
    length = info.get('length', 0)
    
    if MIN_LENGTH <= length <= MAX_LENGTH:
        filtered_data[db_id] = info
        files_to_copy.append(filename)

print(f"Found {len(files_to_copy)} valid structures.")

# 4. Copy the files
print(f"Copying files to {TARGET_DIR}...")
for filename in tqdm(files_to_copy):
    src = os.path.join(SOURCE_DIR, filename)
    dst = os.path.join(TARGET_DIR, filename)
    shutil.copy2(src, dst)

# 5. Save the new JSON file
print(f"Saving filtered metadata to {OUTPUT_JSON}...")
with open(OUTPUT_JSON, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print("\n--- Process Complete ---")
print(f"Filtered files count: {len(filtered_data)}")
print(f"New data folder: {TARGET_DIR}")
print(f"New metadata file: {OUTPUT_JSON}")