"""go_mf task surface for the generic downstream CLIs (see downstream/tasks/go/common.py)."""

from downstream.tasks.go import common

BRANCH = "mf"
TASK_YAML = common.TASK_YAML
KIND = common.KIND
processed_dir = common.processed_dir
test_subsets = common.test_subsets


def load_task_config(path=TASK_YAML, *, data_root=None):
    return common.load_task_config(path, data_root=data_root, branch=BRANCH)


class ProcessedSplit(common.GoProcessedSplit):
    def __init__(self, path):
        super().__init__(path, branch=BRANCH)
