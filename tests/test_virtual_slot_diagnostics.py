from collections import Counter, defaultdict

from core.metrics.virtual_slot_diagnostics_callback import (
    VirtualSlotDiagnosticsCallback,
)
from data_loading.atom_position_vocab import VIRTUAL_ATOMIC_NUMBER


def test_slot_prior_summary_exposes_context_free_real_accuracy():
    records = defaultdict(
        lambda: {"target": Counter(), "prediction": Counter()}
    )
    # Slot 1 prior is carbon. It gets all three real carbons right and misses N.
    records[1]["target"].update({6: 3, 7: 1})
    records[1]["prediction"].update({6: 4})
    records[1].update(
        correct=3,
        real_correct=3,
        virtual_correct=0,
        hetero_correct=0,
    )
    # Slot 2 prior is V. That helps all-slot accuracy but contributes zero to
    # the context-free real-only score.
    records[2]["target"].update({VIRTUAL_ATOMIC_NUMBER: 4, 8: 1})
    records[2]["prediction"].update({VIRTUAL_ATOMIC_NUMBER: 5})
    records[2].update(
        correct=4,
        real_correct=0,
        virtual_correct=4,
        hetero_correct=0,
    )

    summary = VirtualSlotDiagnosticsCallback.summarize_domain(records)

    assert summary["support"] == 9
    assert summary["real_support"] == 5
    assert summary["model_all_accuracy"] == 7 / 9
    assert summary["prior_all_accuracy"] == 7 / 9
    assert summary["model_real_accuracy"] == 3 / 5
    assert summary["prior_real_accuracy"] == 3 / 5
    assert summary["virtual_recall"] == 1.0
    assert summary["hetero_recall"] == 0.0
