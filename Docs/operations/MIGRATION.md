# MIND migration: Anzu → Atum

Updated: 13 September 2026; inventory collected 12 September. This is a **user-executed procedure**, not a record
of a completed transfer. Atum access, storage, and CUDA compatibility still need
verification. No transfer, model change, training, or data processing was started
in this documentation task. Retain Anzu until the final checklist passes.

The 100-epoch baseline is complete. See [training status and development handoff](../training/STATUS_AND_NEXT_STEPS_20260912.md)
and the [measured non-Git inventory](NON_GIT_INVENTORY.md). Moving servers does
not require PT regeneration, a new V design, or a 1M-protein run.

## 1. Start from a clone and keep Anzu reachable

After this documentation/skill update is committed and pushed, a clone contains
the current code, all project docs (including this procedure), original-document
archive, `AGENTS.md`, `CLAUDE.md`, and both shared MIND skills. It is a complete
starting point for following this procedure, but Git does not carry logs,
checkpoints, external data, private sessions, or the Python/CUDA environment.

On **Atum**, after confirming the desired storage location and Git SSH access:

```bash
export MIND_DEST=/home/yusuf/mind_transfer_20260912
mkdir -p "$MIND_DEST"
git clone git@github.com:HUBioDataLab/MIND.git "$MIND_DEST/MIND"
cd "$MIND_DEST/MIND"
git rev-parse HEAD
```

Keep Anzu and its disks reachable until export, transfer, checksums and target
validation finish. The source-side commands below can be run through a separate
SSH session to Anzu after you start working on Atum. Do not retire/reformat the
source after cloning. Confirm Atum's disk space, SSH account, conda root and GPU
before the long copies. These host-dependent values are not known in advance.

Transfer the source workspace over the clone **excluding `.git/`**, preserving
all non-Git evidence and local files. Save source Git metadata/refs separately
in the private migration snapshot. Also transfer external datasets/manifests,
user-local context, W&B symlink referents, temporary evidence and environment.

Set these on **Anzu**, replacing the unverified SSH destination and paths:

```bash
export MIND_TARGET=yusuf@atum
export MIND_DEST=/home/yusuf/mind_transfer_20260912
export MIND_TARGET_HOME=/home/yusuf
export MIND_SNAPSHOT=/home/yusuf/mind_migration_snapshot_20260912
```

Examples assume paths without spaces. Use a fresh dedicated destination, not
another developer's worktree. Do not use `--delete`, `--inplace`, or a destructive
Git reset. Inspect `du -sh` for each transfer source and `df -h`/`df -ih` on
both hosts. The inventory totals roughly 0.7 TB before temporary/private
snapshots and working headroom; selected parent directories may contain more.

## 2. Record source state

```bash
set -euo pipefail
umask 077
mkdir -p "$MIND_SNAPSHOT/records" "$MIND_SNAPSHOT/environment" \
  "$MIND_SNAPSHOT/private/user_state" "$MIND_SNAPSHOT/temporary"
cd /home/yusuf/MIND
git rev-parse HEAD > "$MIND_SNAPSHOT/records/git-sha.txt"
git status --short > "$MIND_SNAPSHOT/records/git-status.txt"
git diff --binary > "$MIND_SNAPSHOT/records/worktree.patch"
git diff --cached --binary > "$MIND_SNAPSHOT/records/index.patch"
git ls-files --others --ignored --exclude-standard \
  > "$MIND_SNAPSHOT/records/ignored-files.txt"
git ls-files --others --exclude-standard \
  > "$MIND_SNAPSHOT/records/untracked-files.txt"
git bundle create "$MIND_SNAPSHOT/records/all-refs.bundle" --all
rsync -aH /home/yusuf/MIND/.git/ "$MIND_SNAPSHOT/records/source-git/"
git diff --check
ssh "$MIND_TARGET" "mkdir -p '$MIND_DEST' && df -h '$MIND_DEST' && df -ih '$MIND_DEST'"
```

The bundle/source-Git snapshot protect local refs and metadata; patches protect
tracked edits; the workspace overlay protects ignored/untracked files. Project
docs and skills are now tracked. Before overlaying code, confirm the clone's
HEAD matches the source SHA recorded here. Publish intentional source edits
first, or review the saved patch explicitly; do not blindly overwrite a clone
of a different version. Copying `.git/` into the clone is not the default route.

Before the final sync, verify relevant writers are quiescent. For training,
identify the Python worker by full command/output directory, map its PID to
the GPU, and compare the intended log across two observations. Do not infer
completion from a remembered PID or a sandbox-only process list. The reviewed
full run is complete; other jobs and current agent sessions may still write.

## 3. Environment and private tool snapshots

Export the actual environment, not the older vendor `GET/env.yml`:

```bash
/opt/anaconda3/bin/conda env export -n mind_esa3d --no-builds \
  > "$MIND_SNAPSHOT/environment/mind_esa3d.yml"
/opt/anaconda3/bin/conda list -n mind_esa3d --explicit \
  > "$MIND_SNAPSHOT/environment/conda-explicit.txt"
/opt/anaconda3/envs/mind_esa3d/bin/python -m pip freeze \
  > "$MIND_SNAPSHOT/environment/pip-freeze.txt"
nvidia-smi > "$MIND_SNAPSHOT/environment/nvidia-smi.txt"
```

Versions are listed in the inventory and in the run's W&B `requirements.txt`.
Keep exports private; package URLs can contain local paths or credentials.
Inspect source-only `prefix`/package paths before target installation. Ordinary
rsync of a conda environment is not a verified relocatable installation.

**Idle/close Codex, Claude and OpenCode before their final snapshots.** A live SQLite
database copied inconsistently is not a safe backup. Copy the whole quiescent
directory, including adjacent WAL/state files. An initial copy during work is
preliminary; repeat after clients stop writing. Agents must not stop clients or
workloads on the user's behalf.

```bash
rsync -aH --no-devices --no-specials /home/yusuf/.codex/ \
  "$MIND_SNAPSHOT/private/user_state/codex/"
rsync -aH --no-devices --no-specials /home/yusuf/.claude/ \
  "$MIND_SNAPSHOT/private/user_state/claude/"
rsync -aH /home/yusuf/.cache/wandb/ \
  "$MIND_SNAPSHOT/private/user_state/wandb_cache/"
rsync -aH /home/yusuf/.cache/mind/ \
  "$MIND_SNAPSHOT/private/user_state/mind_cache/"
rsync -aH /home/yusuf/.config/opencode/ \
  "$MIND_SNAPSHOT/private/user_state/opencode_config/"
rsync -aH /home/yusuf/.local/share/opencode/ \
  "$MIND_SNAPSHOT/private/user_state/opencode_data/"
rsync -aH /home/yusuf/.local/state/opencode/ \
  "$MIND_SNAPSHOT/private/user_state/opencode_state/"
rsync -aH /home/yusuf/.cache/opencode/ \
  "$MIND_SNAPSHOT/private/user_state/opencode_cache/"
```

This preserves skills, sessions, rules, attachments, memories, history/state,
packages and plugins where present. Snapshots also may contain credentials such
as `auth.json`: retain user-only permissions, transfer over SSH, never commit or
share them, and **reauthenticate on Atum**. Do not overwrite an active Atum
client's database/authentication. Runtime sockets/devices are deliberately not
portable. Restore into a new/idle client profile only after checking versions
and paths; keep the untouched snapshot if automatic history import is unsupported.

### Project skills need no home-directory installation

Both skills are versioned in `.agents/skills/`: `validate-feature` and
`strict-contracts-and-validation`. Codex and OpenCode discover them there;
Claude discovers relative links in `.claude/skills/`. `CLAUDE.md` imports the
shared `AGENTS.md`. Clone/rsync preserves this layout automatically on Linux.
Follow [the agent workflow](../development/AGENT_WORKFLOW.md) to check discovery
after starting a trusted project session. No separate project skill copies or
tool-specific instruction bodies are needed.

Personal/plugin skills and histories are preserved by the private snapshots.
The old global `validate-feature` remains archived; do not activate it alongside
the repo version. When selectively restoring `~/.codex/skills`, exclude that
legacy folder. A whole-profile restore needs the same duplicate check. Snapshot
preservation does not guarantee cross-version client history import or automatic
plugin authentication.

## 4. Temporary experiment evidence

`/tmp` must not be the destination for durable backups. Archive matching
experiment artifacts before rebooting or retiring Anzu:

```bash
find /tmp -mindepth 1 -maxdepth 1 -name 'mind*' -printf '%f\0' \
  > "$MIND_SNAPSHOT/records/tmp-mind-entries.nul"
tar -C /tmp --null -T "$MIND_SNAPSHOT/records/tmp-mind-entries.nul" \
  -czf "$MIND_SNAPSHOT/temporary/mind-experiment-evidence.tar.gz"
tar -tzf "$MIND_SNAPSHOT/temporary/mind-experiment-evidence.tar.gz" \
  > "$MIND_SNAPSHOT/records/tmp-archive-contents.txt"
```

Permission errors or “file changed as we read it” invalidate a final snapshot.
Resolve the exact paths with their owner; do not silently omit them. The
inventory names relevant audit/checkpoint directories. Treat the archive as
private because old logs/configs can contain sensitive text.

## 5. Transfer complete trees

Use a user-created `tmux` session on Anzu so a laptop disconnect does not end the
copy. Set the section-1 variables inside it and enable `set -euo pipefail`.
`rsync --partial` supports reruns; tar-over-SSH alone is not a resumable primary
transfer method.

```bash
tmux new -s mind_migration
```

Copy the workspace over the matching clone, excluding **only its Git directory**:

```bash
rsync -aH --partial --info=progress2 --protect-args \
  --exclude='/.git/' /home/yusuf/MIND/ "$MIND_TARGET:$MIND_DEST/MIND/" \
  | tee "$MIND_SNAPSHOT/records/workspace-rsync.log"
rsync -aH --partial --info=progress2 --protect-args \
  "$MIND_SNAPSHOT/" "$MIND_TARGET:$MIND_DEST/migration_snapshot/"
```

This includes logs, checkpoints, nested W&B runs, project docs, hidden repository
skills, reference snapshots and local ignored files. The clone retains its Git
metadata; the source-Git backup travels inside `migration_snapshot/records/`.
Keep `pipefail` active when using `tee`; record and check every command's result.

Create dataset destinations:

```bash
ssh "$MIND_TARGET" "mkdir -p \
  '$MIND_DEST/data/proteins' \
  '$MIND_DEST/data/protein_1m/raw_structures_1m_1024' \
  '$MIND_DEST/data/protein_1m/afdb_clusters' \
  '$MIND_DEST/data/rna/raw_structures' \
  '$MIND_DEST/data/rna/virtual_v1' \
  '$MIND_DEST/data/rna_small' \
  '$MIND_DEST/data/small_molecules/unimol'"
```

Copy parent trees to preserve prefix-style chunks and adjacent metadata/indexes:

```bash
rsync -aH --partial --info=progress2 /home/yusuf/data/proteins/ \
  "$MIND_TARGET:$MIND_DEST/data/proteins/"
rsync -aH --partial --info=progress2 \
  /media/ubuntu/8TB/yusuf/protein_data/raw_structures_1m_1024/ \
  "$MIND_TARGET:$MIND_DEST/data/protein_1m/raw_structures_1m_1024/"
rsync -aH --partial --info=progress2 \
  /media/ubuntu/8TB/yusuf/protein_data/afdb_clusters/ \
  "$MIND_TARGET:$MIND_DEST/data/protein_1m/afdb_clusters/"
rsync -aH --partial --info=progress2 /media/ubuntu/8TB/arya/rna_data/ \
  "$MIND_TARGET:$MIND_DEST/data/rna/raw_structures/"
rsync -aH --partial --info=progress2 \
  /media/ubuntu/8TB/yusuf/mind_virtual_v1/rna/ \
  "$MIND_TARGET:$MIND_DEST/data/rna/virtual_v1/"
rsync -aH --partial --info=progress2 /home/yusuf/data/rna_small/ \
  "$MIND_TARGET:$MIND_DEST/data/rna_small/"
rsync -aH --partial --info=progress2 \
  /media/ubuntu/8TB/arin/unimol_pyg_full/ \
  "$MIND_TARGET:$MIND_DEST/data/small_molecules/unimol/"
```

The last command carries all **200** UniMol chunks; the baseline still selects
only 0–9. Keep 40k/1M manifest roots separate. Missing disabled QM9/metabolite
sources are **unavailable**, not successfully transferred empty directories.
The old UniMol `/home/arin/8TB_arin` path is a symlink to the `/media` source;
do not transfer that data twice.

## 6. Verify content and symlinks

Repeat workspace/private snapshots after relevant writers are quiescent. Then
run checksum dry comparisons for **each** source/target mapping, not only Git:

```bash
rsync -aHnci --protect-args /home/yusuf/MIND/ \
  --exclude='/.git/' "$MIND_TARGET:$MIND_DEST/MIND/"
rsync -aHnci --protect-args /home/yusuf/data/proteins/ \
  "$MIND_TARGET:$MIND_DEST/data/proteins/"
```

`-n` is read-only; `-c` reads contents. Inspect itemized output: exit zero alone
does not mean no differences. Resolve missing/content differences for every
dataset and snapshot. This may take substantial disk I/O. Counts/bytes are an
additional check, not a substitute for checksums:

```bash
find /path/to/one/transferred/tree -type f -printf '%s\n' \
  | awk '{n++; b += $1} END {printf "files=%d bytes=%.0f\n", n, b}'
```

Record matching SHA-256 on both hosts for manifests and these baseline artifacts:

The four source hashes were recorded on 13 September in
[BASELINE_SHA256SUMS.txt](BASELINE_SHA256SUMS.txt). From the copied repository
root on Atum, verify them directly with:

```bash
sha256sum -c Docs/operations/BASELINE_SHA256SUMS.txt
```

For additional critical artifacts, generate hashes on Anzu and compare on Atum:

```bash
cd /home/yusuf/MIND
sha256sum Docs/archive/pre_reorganization_20260912.tar.gz \
  logs/September2026/v_full_100epoch_20260901_214648.log \
  outputs/v_full_100epoch_20260901_214648/last.ckpt \
  outputs/v_full_100epoch_20260901_214648/best-checkpoint-epoch=99-val_total_loss=0.7948.ckpt
```

Use `rsync -aH`, not `-L`, to preserve symlink topology. There are **232 absolute
W&B links** into `/home/yusuf/.cache/wandb/logs/`. Restore the saved cache there
if Atum keeps the same home path; do not delete existing cache content. If home
changes, remap only that validated prefix in the **Atum copy**, never Anzu.
The other 51 relative output links travel with the complete output tree.

For the same home-path layout, restore the saved W&B referents on Atum:

```bash
# Atum: set MIND_DEST and MIND_TARGET_HOME to the confirmed local paths first.
mkdir -p "$MIND_TARGET_HOME/.cache/wandb"
rsync -aH "$MIND_DEST/migration_snapshot/private/user_state/wandb_cache/" \
  "$MIND_TARGET_HOME/.cache/wandb/"
```

The private Codex/Claude/OpenCode snapshots can be restored to their corresponding home
directories only when the target profiles are new/idle and source/target client
versions have been checked. Preserve a target profile backup if one already
exists. Review authentication/configuration separately and reauthenticate;
use the repo validation skill after the restore. Keep the source
snapshot intact for rollback and historical reference.

```bash
# Atum, copied repository:
find outputs -xtype l -print
find . -type l -printf '%p -> %l\n'
```

Check all required external referents, not only W&B. Keep private snapshots
restricted and out of Git after restore.

## 7. Remap paths without changing the experiment

Install the exported environment under Atum's conda root, inspect source-only
paths in the export, and compare imports/versions/GPU capability/FlashAttention.
Known Anzu torchvision-symbol and bitsandbytes warnings must not be assumed
harmless for every future feature. Preserve the original environment record.

For example, after reviewing the exported YAML's `prefix` and local package paths:

```bash
# Atum: use its actual conda executable/root and set MIND_DEST locally first.
conda env create -n mind_esa3d \
  -f "$MIND_DEST/migration_snapshot/environment/mind_esa3d.yml"
conda activate mind_esa3d
python -c 'import torch, torch_geometric; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch_geometric.__version__)'
```

Update both YAMLs, Makefile conda/data roots, and real-data adapter-test paths.
From the copied `$MIND_DEST/MIND` repository:

| Domain | New `chunk_dir` | Selection |
|---|---|---|
| Protein | `../data/proteins/processed_14atom` | Preserve each config's chunk range |
| Full RNA | `../data/rna/virtual_v1/june_equivalent_23slot_v2/processed_23slot` | Current June-equivalent selection |
| Fast RNA | `../data/rna_small/processed_23slot_5k` | Chunk 0 |
| UniMol | `../data/small_molecules/unimol/universal_unimol` | Full baseline remains 0–9 |

These are prefixes; sibling `_chunk_*` directories must exist even if the literal
prefix directory does not. Update `dataset_download_dir` and processing paths
as needed. Keep raw 1M data separate. Do not substitute the much larger full
RNA corpus for June-equivalent RNA because both were copied. Path-dependent
index maps/configs may need remapping or regeneration: retain originals and
compare selected molecule IDs. Reauthenticate W&B separately.

## 8. Post-migration checks and completion

Run bounded tests in the installed environment:

```bash
MPLCONFIGDIR=/tmp/mind_mpl python -m pytest -q \
  tests/test_adapters.py tests/test_rna_june_equivalent.py \
  tests/test_geometry_neutral_virtual_slots.py \
  tests/test_virtual_slot_diagnostics.py \
  tests/test_flash_varlen_attention_scope.py \
  tests/test_protein_dynamic_random_edges.py \
  tests/test_chemical_coordination.py tests/test_dynamic_sampler_epoch_plan.py
```

Inspect skipped cases: real-data tests hard-code Anzu paths and CUDA tests need
actual GPU access. Green tests with those cases skipped do not verify migration.
Existing tests also miss the encoder and trainer-loop defects in the handoff;
these are baseline defects, not transfer damage.

Preview without launching:

```bash
make -n train GPU=0 CONFIG=f EPOCHS=3 SEED=42 \
  LOG=atum_migration_smoke OUTPUT_DIR=outputs/atum_migration_smoke
```

After data, paths, and unit/kernel checks pass, the **user** may start the smoke:

```bash
make train GPU=0 CONFIG=f EPOCHS=3 SEED=42 \
  LOG=atum_migration_smoke OUTPUT_DIR=outputs/atum_migration_smoke
```

Select an actually available GPU. Ask whether the user ran it and wait for
confirmation before monitoring. Check finite losses, actual domain membership,
physical-edge validity, owner readout, real/V metrics, memory, and throughput.
A fresh three-epoch smoke need not equal a 100-epoch model. Removed masked-slot
distance metrics are not acceptance gates. Frozen-checkpoint evaluation requires
the explicit Make workflow proposed in the handoff; `make train` does not resume
or evaluate `last.ckpt` automatically.

- [ ] All workspace/data/snapshot mappings copied and checksum-verified.
- [ ] Clone matches recorded source SHA; source refs/local edits preserved separately.
- [ ] Versioned docs, archived originals, shared rules and both skills present.
- [ ] Logs, best/last checkpoints, W&B configs/history, and manifests match hashes.
- [ ] Codex/Claude/OpenCode histories and personal resources retained privately.
- [ ] Both repo skills discovered; Claude links and AGENTS.md import verified.
- [ ] Temporary evidence saved persistently; no required dangling symlinks.
- [ ] Target environment and real GPU attention path verified.
- [ ] Path remapping preserves baseline dataset selection; real-data tests not skipped.
- [ ] User-started smoke and frozen-checkpoint verification reviewed.
- [ ] Missing disabled datasets / optional package caches recorded explicitly.
- [ ] User approves retiring Anzu; retain source until then.

## 9. Later: 1M protein preparation

This is separate from migration. Keep existing PTs and evaluate/fix the runtime
issues first. The old “1.5 TB of PTs” figure is an estimate, not measured complete
output here. Confirm manifest, storage and acceptance gates before generation.

The default `make process-protein` points at 40k data and includes `--force`.
**Do not run it unchanged for 1M or for a harmless migration check.** When the
user explicitly requests processing, use a reviewed direct command (allowed for
data work) or a dedicated Make target with new raw/manifest/output/cache paths.
Check the orchestrator's `--help`, use a new output prefix, produce metadata for
each chunk, and inspect a bounded sample before scale-up. Parallel chunk ranges
must be disjoint. No data processing was performed in this documentation task.
