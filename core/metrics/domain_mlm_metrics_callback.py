"""Lightning callback for the per-domain masked-atom metrics.

Logs ``val_<domain>_mlm/<metric>`` (and ``test_<domain>_mlm/...``) for every
domain present in the epoch and prints a compact text table every
``print_every_n_epochs`` epochs (default 5, the protein/RNA callback cadence;
test epochs always print). Definitions: core/metrics/domain_mlm_metrics.py.

The distance-head and embedding-health captures are off by default; when
enabled they are armed on the model here and only taken in eval mode.
"""

from __future__ import annotations

from typing import Dict, Optional

from pytorch_lightning.callbacks import Callback

from core.metrics.domain_mlm_metrics import (
    DOMAIN_ALIASES,
    DomainMLMMetrics,
    flatten_scalar_metrics,
    format_text_tables,
)
from core.metrics.per_atom_metrics import require_wandb_logger_or_announce

# Config dataset keys of multi_domain.datasets -> metric domain names.
_CONFIG_DOMAIN_KEYS = {"pdb": "protein", "metabolite": "coconut"}


def enabled_config_domains(config) -> list:
    """Domains a run is configured to see (multi_domain datasets marked enabled)."""
    multi_domain = getattr(config, "multi_domain", None) or {}
    if not multi_domain.get("enabled", False):
        dataset = str(getattr(config, "dataset", "")).lower()
        return [DOMAIN_ALIASES[dataset]] if dataset in DOMAIN_ALIASES else []
    domains = []
    for key, settings in (multi_domain.get("datasets") or {}).items():
        if settings.get("enabled", False):
            name = _CONFIG_DOMAIN_KEYS.get(key, key)
            if name not in DOMAIN_ALIASES:
                raise ValueError(f"multi_domain dataset {key!r} has no domain-metric name")
            domains.append(DOMAIN_ALIASES[name])
    return domains


class DomainMLMMetricsCallback(Callback):
    def __init__(
        self,
        enabled: bool = True,
        log_every_n_epochs: int = 1,
        print_every_n_epochs: int = 5,
        distance_diagnostics: bool = False,
        embedding_sample_nodes_per_batch: int = 0,
        embedding_max_nodes_per_domain: int = 4096,
    ) -> None:
        super().__init__()
        if log_every_n_epochs < 1 or print_every_n_epochs < 1:
            raise ValueError("log/print cadence must be at least one epoch")
        self.enabled = enabled
        self.log_every_n_epochs = log_every_n_epochs
        self.print_every_n_epochs = print_every_n_epochs
        self.distance_diagnostics = bool(distance_diagnostics)
        self.embedding_sample_nodes_per_batch = int(embedding_sample_nodes_per_batch)
        self.metrics = {
            "val": DomainMLMMetrics(embedding_max_nodes_per_domain),
            "test": DomainMLMMetrics(embedding_max_nodes_per_domain),
        }
        self.expected_domains: list = []
        self.last_results: Dict[str, dict] = {}
        self.last_text: Optional[str] = None
        self._no_wandb_announced = False

    def setup(self, trainer, pl_module, stage=None) -> None:
        if not self.enabled:
            return
        if "mlm" not in pl_module.config.pretraining_tasks:
            self.enabled = False
            print("ℹ️  Domain MLM metrics disabled: MLM is not a pretraining task")
            return
        self.expected_domains = enabled_config_domains(pl_module.config)
        pl_module.pretraining_tasks.capture_distance_diagnostics = self.distance_diagnostics
        pl_module.eval_embedding_sample_nodes = self.embedding_sample_nodes_per_batch
        print(
            "✅ DomainMLMMetricsCallback enabled "
            f"(domains={self.expected_domains or 'as seen'}, distance_diagnostics={self.distance_diagnostics}, "
            f"embedding_nodes_per_batch={self.embedding_sample_nodes_per_batch})"
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.enabled:
            self.metrics["val"].update_from_batch(batch, outputs)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.enabled:
            self.metrics["test"].update_from_batch(batch, outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled:
            return
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            self.metrics["val"].reset()
            return
        self._finish("val", trainer, pl_module, print_table=epoch % self.print_every_n_epochs == 0)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.enabled:
            self._finish("test", trainer, pl_module, print_table=True)

    def _finish(self, stage, trainer, pl_module, print_table: bool) -> None:
        accumulator = self.metrics[stage]
        results = accumulator.compute()
        accumulator.reset()
        for domain in self.expected_domains:
            if domain not in results:
                print(f"ℹ️  Domain MLM metrics ({stage}): no masked {domain} atom this epoch; nothing logged for it")
        self.last_results = results
        self.last_text = format_text_tables(
            results, f"DOMAIN MLM METRICS ({stage}) — epoch {trainer.current_epoch}"
        )
        if not results:
            return
        wandb_logger = require_wandb_logger_or_announce(self, trainer, pl_module, "Domain MLM metrics")
        if wandb_logger is not None:
            wandb_logger.log_metrics(flatten_scalar_metrics(results, stage), step=trainer.global_step)
        if print_table:
            print(self.last_text)
