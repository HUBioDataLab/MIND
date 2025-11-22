# Code Documentation: MIND Data Processing Pipeline

## Overview
This document provides brief descriptions of each code file in the data processing pipeline, organized by pipeline stage.

---

## Protein Pipeline

This section covers scripts for downloading and preparing raw protein structure data.

## `data/download_protein_clusters.py`

### Purpose
Downloads AlphaFold Database cluster representative metadata file from Foldseek. This is the first step in the protein data pipeline.

### Key Functionality
- Downloads compressed TSV metadata file (`representatives_metadata.tsv.gz`) from AlphaFold DB

### Usage Example
```bash
python data/download_protein_clusters.py --outdir ../data/proteins/afdb_clusters
```

### Notes
- Default output directory: `data/proteins/afdb_clusters/`
- Downloads metadata file needed for filtering and manifest creation
- Used by: `data/protein_pipeline/1_filter_and_create_manifest.py`

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

## Data Loading Pipeline

This section covers scripts for processing raw data into formats suitable for training.

---

## `data_loading/process_chunked_dataset.py`

### Purpose
Orchestrates the chunked dataset processing pipeline for large-scale datasets. Splits datasets into manageable chunks and processes each chunk through two stages: universal cache creation and PyTorch Geometric conversion.

### Key Functionality
- Splits manifest file into configurable number of chunks
- For each chunk, orchestrates two-stage processing:
  1. Creates universal `.pkl` cache files (via `cache_universal_datasets.py`)
  2. Converts `.pkl` to PyTorch Geometric `.pt` format (via `cache_to_pyg.py`)
- Supports processing specific chunk ranges (e.g., `--chunk-range 0-3`)
- Handles errors gracefully with user prompts for continuation
- Automatically detects and processes chunks for training via `LazyUniversalDataset`

### Key Functions
- `load_config()`: Loads configuration from YAML file
- `parse_chunk_range()`: Parses chunk range strings into list of indices
- `apply_config_defaults()`: Applies configuration defaults to arguments
- `validate_arguments()`: Validates all required arguments and paths
- `process_chunk()`: Processes a single chunk (pkl creation → pt conversion)
- `run_command()`: Executes subprocess commands with error handling
- `print_summary()`: Displays final pipeline summary with statistics

### Input/Output

**Input:**
- Raw data directory (PDB/CIF files)
- Manifest CSV file (created by `1_filter_and_create_manifest.py`)
- Optional: YAML config file for default parameters

**Output:**
- Universal `.pkl` cache files in cache directory
- PyTorch Geometric `.pt` files in chunked output directories:
  - `{output_base}_chunk_0/processed/*.pt`
  - `{output_base}_chunk_1/processed/*.pt`
  - etc.

### Usage Example

```bash
# With config file (recommended)
python data_loading/process_chunked_dataset.py \
    --config-yaml-path core/pretraining_config_protein.yaml \
    --data-path ../data/proteins/raw_structures_hq_40k \
    --manifest-file ../data/proteins/afdb_clusters/manifest_hq_40k.csv \
    --num-chunks 50

# Process specific chunks only
python data_loading/process_chunked_dataset.py \
    --dataset pdb \
    --data-path ../data/proteins/raw_structures_hq_40k \
    --manifest-file ../data/proteins/afdb_clusters/manifest_hq_40k.csv \
    --num-chunks 50 \
    --chunk-range 0-2 \
    --cache-dir ../data/proteins/cache_chunked \
    --output-base ../data/proteins/processed_graphs_40k \
    --cutoff 5.0 \
    --max-neighbors 64
```

### Notes
- Called by: Users preparing datasets for training
- Calls: `cache_universal_datasets.py` and `cache_to_pyg.py` as subprocesses
- Used by: `core/train_pretrain.py` (automatically detects chunked datasets via `LazyUniversalDataset`)
- Supports all dataset types: QM9, LBA, PDB (protein)
- Chunked processing enables handling datasets that don't fit in RAM
- Output directories follow pattern: `{output_base}_chunk_{index}/`

---

## `data_loading/cache_universal_datasets.py`

### Purpose
Creates universal `.pkl` cache files from raw dataset files using the adapter system. Supports all dataset types (QM9, LBA, COCONUT, PDB, RNA) and handles chunking for large-scale datasets.

### Key Functionality
- Converts raw data files to universal representation format using dataset-specific adapters
- Supports chunking via manifest file splitting for parallel processing
- Memory-efficient processing using generator pattern (handles 500GB+ datasets)
- Creates `.pkl` cache files that can be converted to PyTorch Geometric format

### Key Functions
- `get_adapter()`: Returns adapter instance and default data path for dataset type
- `cache_dataset()`: Main function that orchestrates caching with optional chunking
- `_create_chunk_manifest()`: Creates temporary manifest file for a specific chunk
- `_generate_cache_filename()`: Generates cache filename based on parameters
- `list_cached_datasets()`: Lists all cached datasets in a directory

### Input/Output

**Input:**
- Raw data directory (PDB/CIF files, QM9 data, etc.)
- Optional: Manifest CSV file (for chunking protein datasets)
- Optional: Chunking parameters (`--num-chunks`, `--chunk-index`)

**Output:**
- Universal `.pkl` cache files:
  - `universal_{dataset}_{max_samples}.pkl` (with max_samples)
  - `universal_{dataset}_chunk_{index}.pkl` (with chunking)
  - `universal_{dataset}_all.pkl` (all samples)

### Notes
- **RAM Efficiency**: Uses generator pattern via `BaseAdapter._data_generator()` to process one molecule at a time. Each molecule is immediately written to disk via `pickle.dump()`, preventing memory accumulation. Can handle 500GB+ datasets safely.
- **Chunking**: For large datasets, split manifest file into chunks. Each chunk creates a temporary manifest file (disk-based, not memory), then processes only that chunk's molecules.
- **Adapter System**: Uses adapter pattern to support multiple dataset types. Each adapter (`protein_adapter.py`, `qm9_adapter.py`, etc.) implements `BaseAdapter` interface with `load_raw_data()` and `create_blocks()` methods. Adapters are instantiated via `get_adapter()` function which returns the appropriate adapter class based on dataset name.
- **Called by**: `process_chunked_dataset.py` (orchestrates chunked processing)
- **Calls**: Dataset-specific adapters (`protein_adapter.py`, `qm9_adapter.py`, `lba_adapter.py`, `coconut_adapter.py`, `rna_adapter.py`)
- **Used by**: `cache_to_pyg.py` (reads `.pkl` files to convert to PyG format)
- Supports all dataset types: QM9, LBA, COCONUT, PDB, RNA

---

## `data_loading/adapters/protein_adapter.py`

### Purpose
Converts raw PDB/CIF protein structure files to universal molecular representation format. Supports both PDB and CIF formats, with optional filtering via manifest files and configurable heteroatom inclusion.

### Key Functionality
- Parses PDB and CIF files using BioPython
- Converts each amino acid residue to a UniversalBlock
- Supports manifest-based file filtering (AlphaFold naming convention)
- Configurable heteroatom inclusion (ligands, ions, etc.)
- Handles alternate atom locations and filters hydrogen atoms

### Key Functions
- `load_raw_data()`: Loads protein structure files, optionally filtered by manifest CSV
- `create_blocks()`: Parses PDB/CIF file and converts residues to UniversalBlock objects
- `convert_to_universal()`: Wraps blocks into UniversalMolecule with metadata

### Input/Output

**Input:**
- Directory containing .pdb/.cif files
- Optional: Manifest CSV file with 'repId' column for filtering

**Output:**
- UniversalMolecule objects with:
  - Blocks representing amino acid residues (and optionally heteroatoms)
  - Atoms with positions, element symbols, and entity indices

### Notes
- **Called by**: `cache_universal_datasets.py` (via `get_adapter()` function)
- **Implements**: `BaseAdapter` interface (abstract methods: `load_raw_data()`, `create_blocks()`)
- **File Naming**: Supports AlphaFold naming convention (`AF-{id}-F1-model_v4.pdb`) and direct filename matching
- **Entity Indexing**: Protein residues have `entity_idx=0`, heteroatoms have `entity_idx=1`
- **Filtering**: Skips hydrogen atoms and alternate locations (keeps only primary or 'A' location)
- **Error Handling**: Skips files that cannot be parsed and continues processing

---

## `data_loading/cache_to_pyg.py`

### Purpose
Converts universal `.pkl` cache files to PyTorch Geometric `.pt` format for efficient training. Supports all dataset types (QM9, LBA, COCONUT, PDB, RNA) and handles large datasets with memory-efficient processing.

### Key Functionality
- Converts universal molecular representations to PyTorch Geometric Data objects
- Constructs edges using radius graph algorithm with configurable cutoff distance
- Caches processed tensors to disk for instant loading during training
- Memory-efficient processing using generator pattern (handles terabyte-scale datasets)
- Supports dataset-specific classes with optimized defaults (QM9, LBA, COCONUT, PDB, RNA)

### Key Functions
- `load_molecules_iteratively()`: Generator function to load molecules one by one from pickle stream
- `OptimizedUniversalDataset.process()`: Main processing method that converts molecules to PyG format
- `OptimizedUniversalDataset._universal_to_pyg()`: Converts single UniversalMolecule to PyG Data object
- `OptimizedUniversalDataset._element_to_atomic_number()`: Converts element symbols to atomic numbers (dictionary lookup)
- `OptimizedUniversalDataset._calculate_edge_features()`: Calculates edge distances between connected atoms

### Input/Output

**Input:**
- Universal `.pkl` cache file (created by `cache_universal_datasets.py`)
- Optional: Processing parameters (cutoff distance, max neighbors, max samples, max atoms)

**Output:**
- PyTorch Geometric `.pt` file in `processed/` subdirectory:
  - `optimized_{cache_name}_{config_sig}.pt` (includes processing parameters in filename)
- File contains collated PyG Data objects ready for training

### Notes
- **Memory Efficiency**: Uses generator pattern (`load_molecules_iteratively`) to process molecules one at a time. For datasets >50K molecules, use external chunking via `process_chunked_dataset.py` to split into smaller chunks (recommended: 10K-20K molecules per chunk).
- **Edge Construction**: Uses `torch_cluster.radius_graph` to construct edges based on distance cutoff. Each atom connects to neighbors within cutoff distance (default: 5.0 Å), up to max_neighbors limit (default: 32).
- **Caching**: Processed `.pt` files are cached based on processing parameters. Changing cutoff or max_neighbors creates a new cache file automatically.
- **Called by**: `process_chunked_dataset.py` (orchestrates chunked processing)
- **Uses**: `data_types.py` (UniversalMolecule, UniversalBlock, UniversalAtom)
- **Used by**: Training scripts via `LazyUniversalDataset` (loads chunks on-demand)
- Supports all dataset types: QM9, LBA, COCONUT, PDB, RNA
- No external dependencies: RDKit removed, uses dictionary lookup for element-to-atomic-number conversion

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

4. Process to universal format (chunked)
   └─> data_loading/process_chunked_dataset.py
       ├─> Calls cache_universal_datasets.py
       │   └─> universal_{dataset}_chunk_{index}.pkl (universal cache)
       └─> Calls cache_to_pyg.py
           └─> processed_graphs_40k_chunk_*/processed/*.pt (PyTorch Geometric)

5. Train model
   └─> core/train_pretrain.py
       └─> Uses LazyUniversalDataset (loads chunks on-demand)
```
