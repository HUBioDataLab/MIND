"""Read the TAPE secondary-structure LMDB files (the release PEER / TorchDrug use) without running pickled code.

Each value is a pickled dict (``primary``, ``ss3``, ``ss8``, ``valid_mask``, ``disorder``, ``id``, ...) whose arrays
are numpy arrays. ``RestrictedUnpickler`` resolves only the four numpy reconstruction globals, so a modified file can
not execute code; any other global raises. Field layout verified on the first 552 train records of the release.

``disorder`` is 1 for residues that have coordinates and 0 for residues the source marks unresolved (the name is
inverted relative to its meaning; unresolved residues carry the coil label 7 in ``ss8``). ``valid_mask`` is 1.0 for
every verified record (it never marks unresolved residues).
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Dict, Iterator

import numpy as np

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
}
REQUIRED_FIELDS = ("id", "primary", "ss3", "ss8", "valid_mask", "disorder")


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in a TAPE record")
        return super().find_class(module, name)


def loads(data: bytes):
    return RestrictedUnpickler(io.BytesIO(data)).load()


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def normalize_record(record: dict, where: str) -> Dict[str, object]:
    """Plain Python/NumPy record with validated fields (ids and sequences as ``str``, labels as int64)."""
    missing = [key for key in REQUIRED_FIELDS if key not in record]
    if missing:
        raise ValueError(f"{where}: record lacks {missing}; has {sorted(record)}")
    out = {"id": _text(record["id"]), "primary": _text(record["primary"])}
    for key in ("ss3", "ss8", "disorder"):
        out[key] = np.asarray(record[key]).astype(np.int64).reshape(-1)
    out["observed"] = out.pop("disorder")  # 1 = resolved in the source structure (see module docstring)
    if set(np.unique(out["observed"]).tolist()) - {0, 1}:
        raise ValueError(f"{where}: 'disorder' holds values other than 0/1")
    out["valid_mask"] = np.asarray(record["valid_mask"], dtype=np.float32).reshape(-1)
    if not (len(out["primary"]) == len(out["ss3"]) == len(out["ss8"]) == len(out["observed"]) == len(out["valid_mask"])):
        raise ValueError(f"{where}: field lengths differ ({out['id']})")
    return out


def read_lmdb(path: Path) -> Iterator[Dict[str, object]]:
    """Yield the normalized records of one ``secondary_structure_<split>.lmdb`` directory in stored order."""
    import lmdb

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not an LMDB directory")
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin() as txn:
            count = loads(txn.get(b"num_examples"))
            for index in range(int(count)):
                value = txn.get(str(index).encode())
                if value is None:
                    raise ValueError(f"{path}: record {index} of {count} is missing")
                yield normalize_record(loads(value), f"{path.name}[{index}]")
    finally:
        env.close()
