"""ss_q3 task surface (3-class helix / strand / coil, per residue) for the generic downstream CLIs; see dataset.py."""

from pathlib import Path

from downstream.tasks.ss import dataset

LABEL_SET = "ss3"
TASK_YAML = dataset.TASK_YAML
KIND = dataset.KIND
PER_RESIDUE = True
processed_dir = dataset.processed_dir
augment_test_metrics = lambda metrics_test, coverage: dataset.augment_test_metrics(metrics_test, coverage, LABEL_SET)  # noqa: E731


def load_task_config(path: Path = TASK_YAML, *, data_root=None):
    return dataset.load_task_config(path, data_root=data_root, label_set=LABEL_SET)


class ProcessedSplit(dataset.SSProcessedSplit):
    def __init__(self, path):
        super().__init__(path, LABEL_SET)
