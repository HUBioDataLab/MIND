# COCONUT SDF preparation scripts

Scripts written by Ege (HuBioDataLab) to prepare the COCONUT 3D SDF before it is turned into
training chunks. They were never in this repository; they are copied **unmodified** from the
Atum archive (`/media/16TB/archive/yusuf/MIND/data/raw/metabolite/coconut/source_ege_hubiodatalab/`)
so that a clone can reproduce the preparation. They stream the SDF with RDKit and take file
names on the command line (defaults are relative to `data/COCONUT/`).

Pipeline used for the production set (order matters):

| Step | Script | Default input → output |
|---|---|---|
| 1 | `filter_less_10_atoms.py` | `coconut.sdf` → `coconut_min10atoms.sdf` (keep molecules with at least 10 heavy atoms; `heavy_only=True` by default) |
| 2 | `filter_allowed_elements.py` | `coconut_min10atoms.sdf` → `coconut_min10atoms_allowed_elements.sdf` (keep only H, C, O, N, S, Cl, Br, P, I, Na, F) |
| 3 (optional) | `select_random_sdf_entries.py` | random subset of an SDF (`-n`, `--seed`); produced the 600,000-molecule file |
| 4 | `process_chunked_dataset.py --dataset coconut` | SDF → chunks (see [DATA_PIPELINE.md](../../Docs/data/DATA_PIPELINE.md)) |

Helpers: `count_atoms_sdf.py` (element statistics), `mean_atoms_per_molecule.py`,
`find_problematic_chunks.py` (finds SDF entries whose conversion fails per chunk),
`motif_smarts_search.py` and `smiles_scraper.py` (exploration, not part of the pipeline).

The raw SDF has 5,656,564 molecule records and 329,335,057 atoms (Ege's `counts.log`).
`coconut_sdf_3d-09-2025.zip` is the release downloaded by `data/download_coconut.py`; the
adapter expects the extracted file as `<data_path>/COCONUT/coconut.sdf` (rename it by hand).

## Open question: which SDF produced the 497 production chunks?

The training chunks (`/home/yusuf/data/metabolites/processed_chunk_*`) hold 2,755,135 graphs
(about 5,549 per chunk, which fits roughly 500 chunks of a ~2.8 M-molecule SDF). The
600,000-molecule file `coconut_min10atoms_allowed_elements_selected_600000.sdf` has exactly
600,000 entries and therefore cannot be the source. The exact input file and chunk count are
not recorded anywhere in the repository; confirm with Ege before regenerating.
