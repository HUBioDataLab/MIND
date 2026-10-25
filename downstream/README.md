# Downstream evaluation

Frozen-encoder probes, fine-tuning, a random-init control and ESM-2 baselines
for MIND encoders. The first task is EC number prediction (DeepFRI/GearNet
split: 538 EC terms, 15,550 / 1,729 / 1,919 chains, test cutoffs 30/40/50/70/95 %).

## Pipeline

| Step | Command | Kind |
|---|---|---|
| download | `python -m downstream.tasks.ec.prepare download` | data processing, CPU, network |
| process | `python -m downstream.tasks.ec.prepare process --workers 16` | data processing, CPU |
| snapshot | `python -m downstream.common.registry snapshot <ckpt> <name>` and `... alias current_best <name>` | file copy, CPU |
| embed | `python -m downstream.embed --task ec --model <snapshot> [--random-init]` | GPU inference |
| ESM-2 | `python -m downstream.embed --task ec --model esm2_t12_35M_UR50D --hf-cache-dir DIR` | GPU inference |
| probe | `python -m downstream.probe --task ec --model-id <id> --head linear\|mlp [--pooling P]` | CPU training |
| fine-tune | `python -m downstream.finetune --task ec --model <snapshot> [--random-init]` | GPU training |
| report | `python -m downstream.report --task ec` | CPU |

Training-like steps (embed, probe, fine-tune) are user-run through Make targets.

## Contracts

- `downstream/tasks/ec/task.yaml` must name a missing-atom policy explicitly
  (`missing_atoms`); the checked-in value `UNDECIDED` stops processing.
  See `downstream/common/missing_atoms.py` for `strict`,
  `drop_residue_missing_backbone` + `mask` / `drop_chain_if_fraction_above`.
- Experimental structures never reach `ProteinAdapter` unchecked: the adapter
  pads a missing sidechain atom with V and shifts later atoms into wrong slots.
  Incomplete residues are masked like a pretraining MLM block (all 14 slots,
  mask token 0, sidechain slots latent) or the chain is dropped.
- Graphs are built with the training functions (`_standardize_residue_to_14_atoms`,
  `_universal_to_pyg`, `preposition_visible_virtuals`, edge-free encode/decode,
  `attach_runtime_template_metadata`, protein random edges under a fixed per-chain seed).
- Only pre-PMA node states are used; the PMA graph embedding is untrained.
  Pooling covers real atoms the encoder saw with their true type
  (`active_node_mask & ~mlm_mask`): `mean_atoms`, `ca`, `mean_residues`.
- Chains longer than 1024 residues are skipped and recorded (`length_policy: skip`)
  or split into contiguous windows (`crop_windows`).
- Each processed split has a per-chain QC table (`qc_<split>.tsv`) and summary
  (`qc_<split>.json`); above `qc.max_failed_chain_fraction` the split is not written
  and the command exits 1.

## Layout

Data (outside Git): `/home/yusuf/data/downstream/<task>/<modality>/{raw,processed,splits,provenance.json}`.

Models and results (under `outputs/`, or `$MIND_OUTPUTS_ROOT`):

```
models/<snapshot>/{model.ckpt,card.json,hparams.json}   read-only copy, sha256 in card
models/<alias> -> <snapshot>                            e.g. current_best
downstream/<task>/<model_id>/embeddings/<split>.{npz,json}
downstream/<task>/<model_id>/<protocol>.<pooling>/seed<k>/{metrics.json,predictions.npz,head.pt,run.log,config.json}
```

`model_id` is the snapshot name, `<snapshot>.random_init` for the untrained
control, or the ESM-2 name. Literature EC numbers in the report are fine-tuned
results under other protocols and are labelled as such.
