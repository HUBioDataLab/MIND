"""A global warnings.filterwarnings("ignore") hid every warning of the training process."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]



def test_importing_training_modules_installs_no_blanket_warning_filter():
    code = (
        "import warnings\n"
        "before = list(warnings.filters)\n"
        "import core.train_pretrain, data_loading.cache_to_pyg\n"
        "added = [f for f in warnings.filters if f not in before\n"
        "         and f[0] == 'ignore' and f[1] is None and f[2] is Warning and f[3] is None]\n"
        "assert not added, added\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-1500:]
