"""Process the MoleculeNet tasks from the Uni-Mol LMDB files (data-processing CLI; run only on explicit request).

    python -m downstream.tasks.molnet.prepare process  --task mn_esol [--data-root DIR] [--splits train valid test]
    python -m downstream.tasks.molnet.prepare validate --task mn_esol [--data-root DIR] [--splits ...]

``process`` reads ``<source_root>/raw/extracted/molecular_property_prediction/<dataset>/{train,valid,test}.lmdb``
(read-only), writes ``<data_root>/processed/<revision>/{split}.pt`` plus ``qc_<split>.json/.tsv`` and appends a
``process`` record (command, config and input/output sha256, QC summaries) to ``<data_root>/provenance.json``.
It never overwrites an existing split file and raises on any corrupt record, count mismatch or duplicate molecule.

``validate`` runs the same reading and checks and prints the QC summaries without writing anything.

``download`` does not exist: the Uni-Mol archive was fetched and verified once (``<source_root>/provenance.json``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from downstream.tasks.molnet import common


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("download", "validate", "process"))
    parser.add_argument("--task", required=True, choices=sorted(common.DATASET_OF_TASK))
    parser.add_argument("--config", type=Path, default=common.TASK_YAML)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args(argv)
    if args.command == "download":
        parser.exit(2, "there is no download step for the mn_* tasks: the Uni-Mol archive is already extracted under "
                       "<source_root>/raw (see <source_root>/provenance.json)\n")
    config = common.load_task_config(args.config, data_root=args.data_root, dataset=common.DATASET_OF_TASK[args.task])
    splits = args.splits or config["splits"]
    unknown = [s for s in splits if s not in config["splits"]]
    if unknown:
        parser.error(f"unknown splits {unknown}; choose from {config['splits']}")
    return common.build_processed(config, splits=splits, dry_run=args.command == "validate")


if __name__ == "__main__":
    raise SystemExit(main())
