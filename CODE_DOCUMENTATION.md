# Code Documentation: Protein Pipeline

## Overview
This document provides brief descriptions of each code file in the protein data processing pipeline.

---

## `data/protein_pipeline/1_filter_and_create_manifest.py`

### Purpose
Filters AlphaFold Database metadata and creates a download manifest CSV file for protein structure files (PDB/CIF).

### Key Functionality

#### Two Operation Modes:

1. **Analyze Mode** (`--mode analyze`)
   - Analyzes the metadata distribution without creating any files
   - Provides statistics on:
     - pLDDT score distribution at various thresholds (50, 60, 70, 80, 90)
     - Sequence length distribution
     - Combined filter statistics (pLDDT + length constraints)

2. **Manifest Mode** (`--mode manifest`)
   - Creates a filtered CSV manifest file for downloading protein structures
   - Filters proteins by:
     - Minimum pLDDT score (quality threshold)
     - Maximum sequence length (amino acid count)
   - Excludes proteins that already exist in the target directory
   - Sorts by pLDDT score (highest quality first)
   - **Fetches download URLs from AlphaFold DB API** using parallel workers for efficiency

### Key Functions

- `load_metadata()`: Loads and parses the compressed TSV metadata file from AlphaFold DB
- `get_existing_structure_ids()`: Scans a directory for existing structure files to avoid re-downloading
- `get_download_urls_from_api()`: Fetches PDB and CIF download URLs from AlphaFold DB API
- `analyze_metadata()`: Performs statistical analysis on the metadata
- `create_manifest()`: Creates the filtered manifest CSV with API-fetched download URLs

### Input/Output

**Input:**
- AlphaFold DB metadata file (compressed TSV: `representatives_metadata.tsv.gz`)
- Optional: Directory of existing structure files (to exclude from manifest)

**Output:**
- Manifest CSV file containing:
  - UniProt ID (`repId`)
  - pLDDT score (`repPlddt`)
  - Sequence length (`repLen`)
  - PDB download URL (`pdb_url`) - fetched from API
  - CIF download URL (`cif_url`) - fetched from API

### Usage Example

```bash
# Analyze metadata
python data/protein_pipeline/1_filter_and_create_manifest.py \
    --mode analyze \
    --metadata-file ../data/proteins/afdb_clusters/representatives_metadata.tsv.gz

# Create manifest for 40,000 proteins
python data/protein_pipeline/1_filter_and_create_manifest.py \
    --mode manifest \
    --metadata-file ../data/proteins/afdb_clusters/representatives_metadata.tsv.gz \
    --target-count 40000 \
    --output ../data/proteins/afdb_clusters/manifest_hq_40k.csv \
    --existing-structures-dir ../data/proteins/raw_structures_hq_40k \
    --plddt 70 \
    --max-len 512
```

### Notes
- Uses AlphaFold DB API (`https://alphafold.ebi.ac.uk/api/prediction/{UniProtID}`) to fetch current download URLs
- Parallel processing with configurable workers (default: 100) for efficient API calls
- The manifest prioritizes higher quality structures by sorting on pLDDT score
- URLs are fetched dynamically, ensuring compatibility with AlphaFold DB URL structure changes

---

## `data/protein_pipeline/2_download_pdbs_from_manifest.py`

### Purpose
Downloads protein structure files (PDB/CIF) from URLs listed in a manifest CSV file using parallel workers.

### Key Functionality
- Reads a manifest CSV file created by `1_filter_and_create_manifest.py`
- Downloads structures in parallel using multiple worker threads (configurable, default: 8)
- Attempts PDB format first, falls back to CIF if PDB is unavailable
- Skips files that already exist locally
- Provides progress tracking with real-time statistics
- Includes retry logic for failed downloads

### Key Functions
- `download_text()`: Downloads a file from URL with retry logic (3 attempts with exponential backoff)
- `is_html_error()`: Validates downloaded content is not an HTML error page
- `download_one_structure()`: Downloads a single protein structure (tries PDB, then CIF)
- `parallel_download_from_manifest()`: Orchestrates parallel downloads from manifest

### Input/Output

**Input:**
- Manifest CSV file (created by `1_filter_and_create_manifest.py`)

**Output:**
- PDB or CIF files saved to the specified output directory
- Download statistics (success/failure counts):
  - `PDB_OK`: Successfully downloaded PDB files
  - `CIF_OK`: Successfully downloaded CIF files (fallback)
  - `FAILED`: Failed downloads
  - `SKIPPED_EXIST`: Files that already existed locally

### Usage Example

```bash
python data/protein_pipeline/2_download_pdbs_from_manifest.py \
    --manifest-file ../data/proteins/afdb_clusters/manifest_hq_40k.csv \
    --structures-outdir ../data/proteins/raw_structures_hq_40k \
    --workers 16
```

### Notes
- Uses parallel workers for efficient downloading (recommended: 16+ workers for large datasets)
- Progress bar shows real-time download statistics
- Automatically handles HTTP errors and retries failed downloads
- Validates downloaded content to ensure it's not an error page

---

## Pipeline Flow

```
1. Download metadata
   └─> data/download_protein_clusters.py
       └─> representatives_metadata.tsv.gz

2. Filter and create manifest
   └─> data/protein_pipeline/1_filter_and_create_manifest.py
       └─> manifest_hq_40k.csv (with API-fetched URLs)

3. Download structures
   └─> data/protein_pipeline/2_download_pdbs_from_manifest.py
       └─> raw_structures_hq_40k/ (PDB/CIF files)

4. Process to universal format
   └─> data_loading/process_chunked_dataset.py
       └─> processed_graphs_40k_chunk_*/ (PyTorch Geometric format)

5. Train model
   └─> core/train_pretrain.py
```

---

## Recent Improvements

### API Integration
- Both scripts now use the AlphaFold DB API to ensure URL compatibility
- `1_filter_and_create_manifest.py` fetches URLs directly from the API
- Parallel processing significantly speeds up URL fetching (100 workers)

### Code Quality
- Clean, maintainable code with English documentation
- Proper error handling and retry logic
- Type hints throughout
- Progress tracking for long-running operations

### Performance
- Parallel API calls for URL fetching (configurable workers)
- Parallel downloads with configurable worker count
- Efficient file existence checking to avoid re-downloads

