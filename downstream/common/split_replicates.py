"""Per-seed data splits: one processed split per run seed (tags ``seed0``, ``seed1``, ...).

Some benchmarks redraw their random train / valid / test split for every repeat and tie the model initialization to the
same seed (the NPASS benchmark of the NaFM reference code). Such a task carries ``split.seeds`` in its task.yaml and
``split_replicates: true`` in the ledger catalog; the split tag of a run is then a function of the run seed, so the
repeats of one cell are the runs of the different tags. The helpers here are the single definition of that mapping.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

TAG_PREFIX = "seed"
_TAG = re.compile(rf"^{TAG_PREFIX}(0|[1-9][0-9]*)$")


def tag_for_seed(seed: int) -> str:
    """``seed<k>``, the split tag of run seed ``k``."""
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError(f"a split seed must be a non-negative integer, got {seed!r}")
    return f"{TAG_PREFIX}{int(seed)}"


def seed_for_tag(tag: str) -> int:
    """The run seed of a split tag ``seed<k>``; any other tag raises."""
    match = _TAG.match(str(tag))
    if match is None:
        raise ValueError(f"split tag {tag!r} is not of the form {TAG_PREFIX}<k> (k = the run seed that draws the split)")
    return int(match.group(1))


def is_replicated(config: dict) -> bool:
    """Whether a task config draws one split per seed (it then lists ``split_seeds``)."""
    return "split_seeds" in config


def tag_for_run(config: dict, seed: int, *, explicit: Optional[str] = None) -> Optional[str]:
    """Split tag of run seed ``seed`` of a replicated task (``None`` for any other task).

    ``explicit`` (a ``--split-tag`` argument) must equal the tag of the seed: the model initialization and the split
    share the seed, so a run cannot be given another seed's split. A seed without a processed split raises.
    """
    if not is_replicated(config):
        if explicit:
            raise ValueError(f"task {config.get('task')!r} has no per-seed splits; --split-tag does not apply")
        return None
    tag = tag_for_seed(seed)
    if tag not in config["split_tags"]:
        raise ValueError(f"seed {seed} has no split: task {config['task']!r} draws splits for seeds "
                         f"{list(config['split_seeds'])} (task.yaml split.seeds)")
    if explicit is not None and explicit != tag:
        raise ValueError(f"--split-tag {explicit} does not belong to seed {seed} (its split tag is {tag}): the split "
                         "and the model initialization share the seed")
    return tag


def seeds_of_tag(tag: str, seeds: Sequence[int]) -> list:
    """The requested seeds whose split is ``tag`` (at most one)."""
    return [seed for seed in seeds if tag_for_seed(seed) == tag]
