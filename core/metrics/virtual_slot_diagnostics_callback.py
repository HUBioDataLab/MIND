"""Validation diagnostics for fixed protein/RNA virtual-slot representations."""

from collections import Counter, defaultdict

import torch
from pytorch_lightning.callbacks import Callback

from core.molecular_encoder import (
    PROTEIN_LOCAL_SLOT_OFFSET,
    RNA_LOCAL_SLOT_OFFSET,
    build_domain_local_slot_ids,
)
from data_loading.atom_position_vocab import VIRTUAL_ATOMIC_NUMBER


_ATOMIC_LABELS = {
    6: "C",
    7: "N",
    8: "O",
    15: "P",
    16: "S",
    119: "R",
    VIRTUAL_ATOMIC_NUMBER: "V",
}


def _new_slot_record():
    return {"target": Counter(), "prediction": Counter()}


class VirtualSlotDiagnosticsCallback(Callback):
    """Print slot-majority baselines beside the model's real-slot accuracy.

    The baseline never sees molecular context. It predicts the most frequent
    target class observed at each fixed slot. If model accuracy matches it,
    apparently stable SCr/BaseR can be explained by slot marginals alone.
    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self._reset()

    def _reset(self) -> None:
        self._slots = {
            "protein": defaultdict(_new_slot_record),
            "rna": defaultdict(_new_slot_record),
        }
        self._audit_weighted_sum = defaultdict(float)
        self._audit_support = defaultdict(int)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        if not self.enabled or not isinstance(outputs, dict):
            return
        predictions = outputs.get("mlm_preds_atomic_num")
        if predictions is None:
            return
        required = ("original_types", "mlm_mask", "target_slot_mask")
        if any(getattr(batch, name, None) is None for name in required):
            return

        predictions = torch.as_tensor(predictions).long().cpu()
        targets = batch.original_types.detach().long().cpu()
        selected = (
            batch.mlm_mask.detach().bool().cpu()
            & batch.target_slot_mask.detach().bool().cpu()
        )
        slot_ids = build_domain_local_slot_ids(
            batch, targets.numel(), batch.original_types.device
        ).cpu()

        domain_specs = (
            ("protein", PROTEIN_LOCAL_SLOT_OFFSET + 4,
             PROTEIN_LOCAL_SLOT_OFFSET + 13, 4),
            ("rna", RNA_LOCAL_SLOT_OFFSET + 12,
             RNA_LOCAL_SLOT_OFFSET + 22, 12),
        )
        for domain, first_id, last_id, ordinal_shift in domain_specs:
            for slot_id in range(first_id, last_id + 1):
                slot_mask = selected & (slot_ids == slot_id)
                if not bool(slot_mask.any()):
                    continue
                display_slot = slot_id - (
                    PROTEIN_LOCAL_SLOT_OFFSET
                    if domain == "protein"
                    else RNA_LOCAL_SLOT_OFFSET
                ) + 1 - ordinal_shift
                record = self._slots[domain][display_slot]
                record["target"].update(targets[slot_mask].tolist())
                record["prediction"].update(predictions[slot_mask].tolist())
                self._attach_joint_counts(
                    domain,
                    display_slot,
                    targets[slot_mask],
                    predictions[slot_mask],
                )

        for name, value_and_support in outputs.get(
            "virtual_slot_context_audit", {}
        ).items():
            value, support = value_and_support
            support = int(support)
            self._audit_weighted_sum[name] += float(value) * support
            self._audit_support[name] += support

    @staticmethod
    def _format_counts(counts: Counter) -> str:
        return ",".join(
            f"{_ATOMIC_LABELS.get(int(atom), str(atom))}:{count}"
            for atom, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        )

    @staticmethod
    def summarize_domain(slot_records) -> dict:
        total = correct = prior_correct = 0
        real_total = real_correct = prior_real_correct = 0
        virtual_total = virtual_correct = 0
        hetero_total = hetero_correct = 0
        rows = []
        for slot, record in sorted(slot_records.items()):
            targets = record["target"]
            predictions = record["prediction"]
            support = sum(targets.values())
            if support == 0:
                continue
            prior = min(
                (atom for atom, count in targets.items()
                 if count == max(targets.values())),
                default=VIRTUAL_ATOMIC_NUMBER,
            )
            # Actual joint correctness cannot be recovered from two marginal
            # counters. It is stored explicitly by the accumulator when
            # present; older/pure test records may omit it.
            slot_correct = int(record.get("correct", 0))
            slot_real_total = support - targets[VIRTUAL_ATOMIC_NUMBER]
            slot_real_correct = int(record.get("real_correct", 0))
            slot_virtual_correct = int(record.get("virtual_correct", 0))
            slot_hetero_total = sum(targets[atom] for atom in (7, 8, 16))
            slot_hetero_correct = int(record.get("hetero_correct", 0))

            total += support
            correct += slot_correct
            prior_correct += targets[prior]
            real_total += slot_real_total
            real_correct += slot_real_correct
            if prior != VIRTUAL_ATOMIC_NUMBER:
                prior_real_correct += targets[prior]
            virtual_total += targets[VIRTUAL_ATOMIC_NUMBER]
            virtual_correct += slot_virtual_correct
            hetero_total += slot_hetero_total
            hetero_correct += slot_hetero_correct
            rows.append(
                {
                    "slot": slot,
                    "support": support,
                    "targets": VirtualSlotDiagnosticsCallback._format_counts(targets),
                    "prior": _ATOMIC_LABELS.get(int(prior), str(prior)),
                    "prediction_top": VirtualSlotDiagnosticsCallback._format_counts(
                        predictions
                    ),
                    "model_accuracy": slot_correct / support,
                }
            )
        return {
            "rows": rows,
            "model_all_accuracy": correct / total if total else 0.0,
            "prior_all_accuracy": prior_correct / total if total else 0.0,
            "model_real_accuracy": real_correct / real_total if real_total else 0.0,
            "prior_real_accuracy": (
                prior_real_correct / real_total if real_total else 0.0
            ),
            "virtual_recall": (
                virtual_correct / virtual_total if virtual_total else 0.0
            ),
            "hetero_recall": (
                hetero_correct / hetero_total if hetero_total else 0.0
            ),
            "support": total,
            "real_support": real_total,
            "virtual_support": virtual_total,
            "hetero_support": hetero_total,
        }

    def _attach_joint_counts(
        self,
        domain: str,
        display_slot: int,
        targets: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        record = self._slots[domain][display_slot]
        matches = predictions == targets
        record["correct"] = int(record.get("correct", 0)) + int(matches.sum())
        real = targets != VIRTUAL_ATOMIC_NUMBER
        record["real_correct"] = int(record.get("real_correct", 0)) + int(
            (matches & real).sum()
        )
        virtual = ~real
        record["virtual_correct"] = int(
            record.get("virtual_correct", 0)
        ) + int((matches & virtual).sum())
        hetero = (targets == 7) | (targets == 8) | (targets == 16)
        record["hetero_correct"] = int(
            record.get("hetero_correct", 0)
        ) + int((matches & hetero).sum())

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.enabled:
            return
        if not any(self._slots[domain] for domain in ("protein", "rna")):
            self._reset()
            return

        print("\n" + "=" * 96)
        print("VIRTUAL SLOT DIAGNOSTIC — model versus context-free slot prior")
        print("=" * 96)
        for domain in ("protein", "rna"):
            summary = self.summarize_domain(self._slots[domain])
            if not summary["rows"]:
                continue
            print(f"\n{domain.upper()} fixed target slots")
            print("slot | support | target distribution | prior | model prediction distribution | model acc")
            for row in summary["rows"]:
                print(
                    f"{row['slot']:>4} | {row['support']:>7} | "
                    f"{row['targets']:<28} | {row['prior']:^5} | "
                    f"{row['prediction_top']:<30} | "
                    f"{row['model_accuracy']:.4f}"
                )
            print(
                "overall: "
                f"model(all)={summary['model_all_accuracy']:.4f}, "
                f"slot-prior(all)={summary['prior_all_accuracy']:.4f}, "
                f"model(real)={summary['model_real_accuracy']:.4f}, "
                f"slot-prior(real)={summary['prior_real_accuracy']:.4f}, "
                f"V-recall={summary['virtual_recall']:.4f}, "
                f"hetero(N/O/S)-recall={summary['hetero_recall']:.4f}"
            )

        if self._audit_support:
            print("\nLATENT OWNER READOUT COUNTERFACTUAL (positive CE/accuracy drops are desired after learning)")
            for name in sorted(self._audit_support):
                support = self._audit_support[name]
                mean = self._audit_weighted_sum[name] / max(support, 1)
                print(f"  {name}: {mean:+.6f} (support={support})")
        print("=" * 96)
        self._reset()
