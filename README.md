# Molecular Intelligence for Novel Discovery (MIND)

## About The Project

MIND is a universal foundation model for molecular representation learning, designed to handle diverse biological molecules like DNA, RNA, proteins, and small molecules within a single, unified framework. The model is built upon a Graph Transformer backbone and is trained using a self-supervised strategy to comprehend the universal principles of 3D molecular structure.

## Project Structure and Key Files

This project is organized into several key directories. Here's a summary of the most important files and their roles in the pipeline.

```
MIND/
├── core/
│   ├── pretraining_model.py
│   ├── train_pretrain.py
│   └── pretraining_config_protein.yaml
├── data/
│   ├── protein_pipeline/
│   │   ├── 1_filter_and_create_manifest.py
│   │   └── 2_download_pdbs_from_manifest.py
│   └── download_protein_clusters.py
├── data_loading/
│   ├── adapters/
│   │   ├── base_adapter.py
│   │   └── protein_adapter.py
│   ├── cache_universal_datasets.py
│   ├── cache_to_pyg.py
│   └── data_types.py
└── CODE_DOCUMENTATION.md
```

### `core/` - The Heart of the Model

This directory contains the main logic for the model architecture and the training loop.

-   **`pretraining_model.py`**: Defines the `PretrainingESAModel` architecture, including the `UniversalMolecularEncoder`, the `ESA` backbone, and all pre-training loss functions (`long_range_distance_loss`, `mlm_loss`, etc.).
-   **`train_pretrain.py`**: The main script used to start a training run. It handles loading the configuration, creating the dataset and data loaders, initializing the model and trainer, and starting the `PyTorch Lightning` training loop.
-   **`pretraining_config_protein.yaml`**: The YAML configuration file where all hyperparameters for a protein pre-training run are defined, such as learning rate, batch size, model dimensions, and pre-training tasks.

### `data/` - Raw Data Acquisition

This directory contains scripts for downloading and preparing the initial raw data.

-   **`download_protein_clusters.py`**: Downloads the raw metadata file (`representatives_metadata.tsv.gz`) from the AlphaFold DB clusters. This is the very first step.
-   **`protein_pipeline/1_filter_and_create_manifest.py`**: Reads the large metadata file and creates a smaller, filtered `manifest.csv` file based on user-defined criteria like pLDDT score and protein length. Uses the AlphaFold DB API to fetch current download URLs with parallel processing for efficiency.
-   **`protein_pipeline/2_download_pdbs_from_manifest.py`**: Reads the `manifest.csv` file and downloads the corresponding protein structure files (`.pdb`/`.cif`).

### `data_loading/` - Data Processing and Conversion

This directory is the bridge between raw data and the format required by the model. It handles the memory-intensive data conversion pipeline.

-   **`data_types.py`**: Defines the "Universal Representation" for all molecules (`UniversalMolecule`, `UniversalBlock`, `UniversalAtom`). This is the standardized format used in the intermediate steps.
-   **`adapters/`**: Contains adapter classes (`ProteinAdapter`, etc.) that are responsible for converting different raw data formats (like `.pdb` files) into the Universal Representation.
    -   **`base_adapter.py`**: The abstract base class for all adapters. It contains the main logic for processing a dataset and saving it to an iterable `.pkl` cache file.
-   **`cache_universal_datasets.py`**: An orchestrator script that uses an adapter (e.g., `ProteinAdapter`) to convert a folder of raw files into a memory-efficient, iterable `.pkl` cache file.
-   **`cache_to_pyg.py`**: The final and most critical processing script. It reads the `.pkl` cache file, converts the data into `PyTorch Geometric` graph objects, builds the graph structure (e.g., `radius_graph`), and saves the final, optimized dataset as a `.pt` file, ready for training. We have extensively modified this file to handle large datasets without running out of RAM.

This guide explains how to set up the environment and run the complete data processing and pre-training pipeline for the protein dataset.

## Getting Started

Follow these steps to set up your local environment for development and training.

### Prerequisites

-   A Linux-based system with NVIDIA GPUs and CUDA installed.
-   **Conda** package manager.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd MIND
    ```

2.  **Create and activate the Conda environment:**
    The project dependencies are specified in the `env.yml` file.
    ```bash
    # Create the conda environment from the provided file
    conda env create -f GET/env.yml

    # Activate the environment
    conda activate mind_esa3d
    ```

## Training Pipeline: Step-by-Step Guide

This guide will walk you through the entire process, from downloading raw data to starting the pre-training job. The pipeline is designed to be memory-efficient and scalable to millions of proteins by processing data in chunks.

### Step 1: Download Initial Metadata

First, we need to download the metadata file from the AlphaFold DB clusters. This file contains information about all protein structures, which we will filter in the next step.

```bash
python data/download_protein_clusters.py
```
This will download `representatives_metadata.tsv.gz` into the `data/proteins/afdb_clusters/` directory.

### Step 2: Create a Filtered Manifest

Next, we create a "manifest" file. This is a CSV file that lists the specific proteins we want to use for training, based on quality (pLDDT score) and size (amino acid length). The script uses the AlphaFold DB API to fetch current download URLs, ensuring compatibility with the latest database structure.

```bash
# This example creates a manifest for 40,000 proteins with a pLDDT score > 70 and fewer than 512 amino acids.
python data/protein_pipeline/1_filter_and_create_manifest.py \
    --mode manifest \
    --metadata-file data/proteins/afdb_clusters/representatives_metadata.tsv.gz \
    --target-count 40000 \
    --output data/proteins/afdb_clusters/manifest_hq_40k_len512.csv \
    --existing-structures-dir data/proteins/raw_structures_hq_40k \
    --plddt 70 \
    --max-len 512
```

### Step 3: Download Protein Structures

Using the manifest file created in the previous step, this script downloads the actual 3D structure files (`.pdb` or `.cif`) for the selected proteins.

```bash
python data/protein_pipeline/2_download_pdbs_from_manifest.py \
    --manifest-file data/proteins/afdb_clusters/manifest_hq_40k_len512.csv \
    --structures-outdir data/proteins/raw_structures_hq_40k \
    --workers 16
```

### Step 4: Process Data into Scalable Chunks

This is the main data processing step. A new automated script, `process_chunked_dataset.py`, handles the entire memory-intensive conversion. It splits the manifest into smaller parts ("chunks") and processes each one independently to avoid running out of RAM.

This script performs two stages for each chunk:
1.  Converts raw `.pdb` files to the universal `.pkl` format.
2.  Converts the `.pkl` file to the final, optimized PyTorch Geometric `.pt` format.

```bash
# This command processes the 40k manifest into 2o chunks (20k proteins each).
# For 1M proteins, you can increase --num-chunks to 200.
python data_loading/process_chunked_dataset.py \
    --config-yaml-path core/pretraining_config_protein.yaml \
    --data-path data/proteins/raw_structures_hq_40k \
    --manifest-file data/proteins/afdb_clusters/manifest_hq_40k_len512.csv \
    --num-chunks 20
```
After this step, your processed data will be ready in separate chunk directories (e.g., `../data/proteins/processed_graphs_40k_chunk_0/`, `..._chunk_1/`).

### Step 5: Start Pre-training

Finally, start the training. The training script (`core/train_pretrain.py`) is designed to automatically detect the chunked dataset directories and load the data efficiently using the `LazyUniversalDataset`.

We use `nohup` to ensure the training continues even if your SSH connection is lost.

```bash
# Make sure you are in the project's root directory (MIND/)
# This command starts training on GPU 6 and saves all logs to 'mind_training.log'.
CUDA_VISIBLE_DEVICES=6 nohup python -m core.train_pretrain \
    --config-yaml-path core/pretraining_config_protein.yaml \
    > mind_training.log 2>&1 &
```
