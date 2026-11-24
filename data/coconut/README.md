# COCONUT SDF preparation scripts

Scripts from the HUBioDataLab COCONUT workflow that prepare the COCONUT 3D SDF before it is
turned into training chunks. They are included **unmodified** so that a clone can reproduce the
preparation. They stream the SDF with RDKit and take file names on the command line (defaults
are relative to `data/COCONUT/`).

Pipeline for the production set (order matters):

| Step | Script | Default input → output |
|---|---|---|
| 1 | `filter_less_10_atoms.py` | `coconut.sdf` → `coconut_min10atoms.sdf` (keep molecules with at least 10 heavy atoms; `heavy_only=True` by default) |
| 2 | `filter_allowed_elements.py` | `coconut_min10atoms.sdf` → `coconut_min10atoms_allowed_elements.sdf` (keep only H, C, O, N, S, Cl, Br, P, I, Na, F) |
| 3 (optional) | `select_random_sdf_entries.py` | random subset of an SDF (`-n`, `--seed`); produced the 600,000-molecule file |
| 4 | `data_loading/build_coconut.py` (`make build-coconut`) | raw SDF or zip → recorded chunks; applies the same two filters itself, so steps 1-2 are only needed to reproduce the intermediate SDFs (see [DATA_PIPELINE.md](../../Docs/data/DATA_PIPELINE.md#coconut-build)) |

Helpers: `count_atoms_sdf.py` (element statistics), `mean_atoms_per_molecule.py`,
`find_problematic_chunks.py` (finds SDF entries whose conversion fails per chunk),
`motif_smarts_search.py` and `smiles_scraper.py` (exploration, not part of the pipeline).

The raw SDF has 5,656,564 molecule records and 329,335,057 atoms (measured with `count_atoms_sdf.py`).
`coconut_sdf_3d-09-2025.zip` is the release downloaded by `data/download_coconut.py`; the
adapter expects the extracted file as `<data_path>/COCONUT/coconut.sdf` (rename it by hand).

## Production chunk set

`make build-coconut` (`data_loading/build_coconut.py`) builds the production chunks from the raw SDF or the official zip:
`<DATA_ROOT>/metabolites/coconut_full_{train,val,test}_chunk_NNN`, each conformer a graph, the split assigned per molecule
(all conformers of a molecule share a split), with a build manifest (input checksum, filter counts, chunk counts) and a
completeness check (`--verify-only` repeats it from the files). The four-domain configs read this layout.
