import os
import pandas as pd

# Configuration
raw_dir = "data/rna/raw_structures"
output_csv = "data/rna/manifests/manifest_rna.csv"

# Scan files
data = []
for f in os.listdir(raw_dir):
    if f.endswith(".cif"):
        data.append({
            "name": f.replace(".cif", ""),
            "path": os.path.join(raw_dir, f),
            "length": 0 # Optional: You can calculate sequence length here if needed for filtering
        })

# Save
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"Manifest created with {len(df)} entries.")