# Non-Git migration inventory — Anzu, 12 September 2026

This inventory was obtained by read-only directory, size, symlink, configuration,
and package-metadata inspection. It is not a data-content integrity audit and no
transfer has been performed. Sizes are approximate unless bytes are specified.
Use the [migration procedure](MIGRATION.md) for transfer and verification.

## Workspace: Git contents and additional transfer contents

| Source under `/home/yusuf/MIND/` | Observed size / contents | Why preserve it |
|---|---|---|
| `.git/` and worktree | About 14 MB source Git metadata | Clone obtains pushed code; record local refs/index/edits in the migration snapshot |
| `Docs/`, root document pointers | Canonical docs plus original-document archive | Versioned in this update; clone includes them |
| `logs/` | 92 files, 1,385,127,249 bytes | Training and historical experiment evidence |
| `outputs/` | 948 regular files, 22,237,662,973 bytes; 283 symlinks | Checkpoints, W&B histories/configs, diagnostic output |
| `.agents/skills/`, `.claude/skills/`, `AGENTS.md`, `CLAUDE.md` | Shared skills, Claude symlinks and common rules | Versioned; no manual home installation is needed for these project skills |
| `data_loading/cache/` | 9 files, 2,741,884,861 bytes | Prepared cache artifacts not in Git |
| `papers/` | About 28 MB | Reference PDFs and provenance |
| `template/` | About 230 KB | Other-group source/report context |
| `repomix-output.xml`, helper/local files if present | Included by full-tree copy | External-code snapshot and untracked development context |
| `.pytest_cache/`, bytecode | Rebuildable | Included by the complete copy; not required for scientific reproducibility |

The workspace is approximately **25 GB**. The documentation ignore rules were
removed for publication: project docs and skills arrive through Git. Logs,
outputs, preparation caches, private state, and reference artifacts still need
rsync. The migration overlays the workspace onto a clone while excluding its
`.git/`; source Git metadata is saved separately in the migration snapshot.

## External datasets and provenance

| Source | Inventory | Migration status |
|---|---|---|
| `/home/yusuf/data/proteins/processed_14atom_chunk_0` … `_9` | 10 chunks, 50 files, 64,454,381,604 bytes (~60 GiB), including PT/metadata/raw PKL | Required to reproduce the current run without reprocessing |
| `/home/yusuf/data/proteins/cache_14atom/` | 10 PKL files, 4,950,001,695 bytes | Preserve preparation cache; omitted in the former plan |
| `/home/yusuf/data/proteins/raw_structures_hq_40k/` | 40,000 PDB files, ~3.7 GB | Required provenance / adapter checks |
| `/home/yusuf/data/proteins/afdb_clusters/` | ~45 MB; 40k manifest 5,880,306 bytes | Preserve manifests and source metadata |
| `/media/ubuntu/8TB/yusuf/protein_data/raw_structures_1m_1024/` | 1,117,868 structure files: 1,117,607 PDB + 261 CIF; ~160 GB | Future 1M expansion; actual file count is not exactly one million |
| `/media/ubuntu/8TB/yusuf/protein_data/afdb_clusters/` | 1M manifest 164,624,793 bytes; metadata gzip 40,980,878 bytes | Preserve separately from the 40k manifest directory |
| `/media/ubuntu/8TB/arya/rna_data/` | 12,933 CIF files, 17,487,196,786 bytes | Raw RNA provenance |
| `/media/ubuntu/8TB/yusuf/mind_virtual_v1/rna/final_23slot_chunk_000` … `_066` | 67 chunks, 225,757,098,914 bytes (~210 GiB) | Preserve source corpus; **not** the active full-training selection |
| Same RNA root: `june_equivalent_23slot_v2/` | 67 chunks, 135 files, 17,483,791,140 bytes | **Active** multidomain RNA dataset |
| Same RNA root: indexes, validation JSON, `manifests/`, `reports/`, `final_23slot_reports/` | Selection and derivation evidence | Copy the entire parent root to avoid orphaning provenance |
| `/home/yusuf/data/rna_small/` | 16 files, ~1.9 GB; normal and `*_all_*` development chunks | Fast-run dataset plus comparison artifacts; copy entire directory |
| `/media/ubuntu/8TB/arin/unimol_pyg_full/` | 200 chunks (`0..199`), 1,001 files, 192,915,032,541 bytes | Current run selects `0..9`; preserve all 200 for continuity |
| `/media/ubuntu/8TB/ege/processed/` | **Missing** | Metabolite disabled; unavailable, not “transferred” |
| `/home/arin/data/qm9_chunks/` | **Missing** | QM9 disabled; unavailable, not “transferred” |

`chunk_dir` is commonly a **prefix**, not a physical directory. For example,
`processed_14atom` does not exist, while `processed_14atom_chunk_0` through `_9`
do. Testing only `test -d "$chunk_dir"` would incorrectly report missing data.
Preserve every selected chunk's metadata and adjacent manifests/indexes.

`/home/arin/8TB_arin` is a symlink to `/media/ubuntu/8TB/arin`; the two UniMol
paths refer to the same data. Copy it once, then remap configuration paths.

## User-local tools and experiment evidence

| Source | Observed content | Handling |
|---|---|---|
| `/home/yusuf/.codex/` | ~545 MB; 5,888 files | Private snapshot, not a public Git artifact |
| `~/.codex/sessions/` | 35 files, ~108 MB | Conversation/development context; preserve |
| `~/.codex/skills/` | 62 files at original inventory; system and legacy validation skills | Preserve privately; use the repo validation skill rather than reactivating the legacy copy |
| `~/.codex/packages/`, `plugins/` | ~266 MB / ~27 MB | Skill/plugin resources may be referenced by settings; preserve, then verify discovery |
| Other Codex history/state/rules/attachments/memories | Present where configured | Snapshot as a whole when quiescent; do not copy a live SQLite DB without its consistency requirements |
| `/home/yusuf/.claude/` | Global settings, sessions, plugins and sensitive credential files | Same private-backup policy; distinct from repo `.claude/` |
| `/home/yusuf/.config/opencode/` | About 62 MB, inspected 13 September | Private OpenCode configuration/plugin resources |
| `/home/yusuf/.local/share/opencode/` | About 304 KB, inspected 13 September | Private sessions/authentication state; reauthenticate on Atum |
| `/home/yusuf/.local/state/opencode/` | Present, small | Private client state |
| `/home/yusuf/.cache/opencode/` | About 4.5 MB | Preserve snapshot; cache can be rebuilt |
| `/home/yusuf/.cache/wandb/` | ~1.1 MB | Referents for absolute output log symlinks; preserve |
| `/home/yusuf/.cache/mind/` | Small derived index maps | Preserve provenance; regenerate path-dependent maps if necessary without editing source datasets |
| `/tmp/mind*` | Prior audits, configs, logs, reports and some checkpoints | Ephemeral: inventory and archive outside `/tmp` before retiring Anzu |

Relevant temporary names include `mind_e5_*`, `mind_e6_e7_*`, `mind_e7_*`,
`mind_merge_gpu0_*`, `mind_handoffs`, `mind_e3_index_maps`,
`mind_arin_3473_audit`, and `mind_remote_triton_audit_3473cfc`. The
`mind_e5_fast_output_gpu0_6k` and `mind_e5_fixed_batch_final_output` directories
contain checkpoint pairs. A zero-byte overnight log does not prove that its
script ran successfully. The migration plan archives all matching accessible
temporary artifacts, not only these examples.

Authentication files, settings, session text, SQLite state, and W&B histories
may contain private information. This audit inspected filenames/package
metadata, not credential values. Transfer private snapshots over SSH to a
user-owned restricted directory; reauthenticate on Atum instead of blindly
activating copied tokens. Do not place private backups in the repository.

## Symlinks

Of the 283 output symlinks, 232 point absolutely into
`/home/yusuf/.cache/wandb/logs/`; the remaining 51 are relative W&B run links.
No dangling output symlinks were found during inventory. `rsync -aH` preserves
links; `-L` dereferences them and changes the archived topology. If the home
path changes on Atum, explicitly remap the old W&B cache prefix in the copied
links and verify every referent. Do not rewrite the Anzu originals.

## Environment and capacity

The working environment is `/opt/anaconda3/envs/mind_esa3d/` (~12 GB), with
Python 3.11.13, Torch 2.5.0+cu118, PyG 2.6.1, Lightning 2.5.2, NumPy 1.26.4,
Biopython 1.85, RDKit 2025.3.3, flash-attn 2.5.8, bitsandbytes 0.41.1, and
wandb 0.21.0. No editable installs or egg-link files were detected. Export the
actual environment; `GET/env.yml` describes an older Python 3.9/Torch 1.13.1
environment and is not a faithful lockfile for this run.

`/opt/anaconda3/pkgs/` (~8.3 GB) is an optional package-download cache, not
training evidence. CUDA-driver/kernel compatibility still needs a target-host
check even with identical Python package versions.

Allow roughly **0.7 TB for inventoried datasets/workspace**, plus private and
temporary snapshots and working headroom. This is not a reservation or a
bound: measure the full parent directories selected for transfer with `du` and
check free bytes **and inodes** on both hosts. New 1M PT generation can require
substantially more space and is a separate operation.

Known path edits after transfer: the Makefile's conda/data roots, both training
YAMLs, and the real-data cases in `tests/test_adapters.py`. Those tests currently
skip when their hard-coded Anzu paths are absent; a green test summary with
skipped real-data tests is not a verified data migration.
