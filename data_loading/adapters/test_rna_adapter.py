from data_loading.adapters.rna_adapter import RNAAdapter

# Initialize adapter
adapter = RNAAdapter(include_modified=True, include_ions=False)

# Process dataset with manifest
adapter.process_dataset(
    data_path="data/rna/raw_structures",
    cache_path="data/rna/processed/rna_cache.pkl",
    manifest_file="data/rna/manifests/manifest_rna.csv",
    max_samples=None  # Process all
)

# python3 -m data_loading.adapters.test_rna_adapter
# test command
