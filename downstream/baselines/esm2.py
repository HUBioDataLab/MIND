"""ESM-2 sequence baseline (HF transformers ``EsmModel``), pooled like MIND ``mean_residues``.

The sequence is exactly the residues kept in the processed MIND graph, so both
encoders see the same chain coverage. Chains longer than the ESM-2 context
(1022 residues + BOS/EOS) are embedded in contiguous windows and averaged
residue-weighted. Weights are downloaded by HF on first use into the explicit
``cache_dir`` (the environment's default HF cache is not writable here).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

ESM2_MODELS = {
    "esm2_t6_8M_UR50D": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D": "facebook/esm2_t12_35M_UR50D",
}
MAX_RESIDUES = 1022
POOLING = "mean_residues"


def load_esm2(name: str, device, *, cache_dir: str, local_files_only: bool = False):
    if name not in ESM2_MODELS:
        raise ValueError(f"unknown ESM-2 model {name!r}; choose from {sorted(ESM2_MODELS)}")
    from transformers import AutoTokenizer, EsmModel

    repo = ESM2_MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(repo, cache_dir=cache_dir, local_files_only=local_files_only)
    model = EsmModel.from_pretrained(
        repo, cache_dir=cache_dir, local_files_only=local_files_only, add_pooling_layer=False
    )
    return tokenizer, model.to(device).eval()


def _windows(sequence: str) -> List[str]:
    count = max(1, -(-len(sequence) // MAX_RESIDUES))
    bounds = np.linspace(0, len(sequence), count + 1).round().astype(int)
    return [sequence[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


@torch.no_grad()
def embed_sequences(tokenizer, model, sequences: Sequence[str], device, *, batch_size: int = 8) -> np.ndarray:
    """Mean of last-layer residue states (BOS/EOS/padding excluded), one row per sequence."""
    pieces = [(row, piece) for row, sequence in enumerate(sequences) for piece in _windows(sequence)]
    if any(not piece for _, piece in pieces):
        raise ValueError("empty sequence")
    sums = np.zeros((len(sequences), model.config.hidden_size), dtype=np.float64)
    counts = np.zeros(len(sequences))
    for start in range(0, len(pieces), batch_size):
        chunk = pieces[start:start + batch_size]
        encoded = tokenizer([p for _, p in chunk], return_tensors="pt", padding=True,
                            return_special_tokens_mask=True)
        special = encoded.pop("special_tokens_mask").bool()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = model(**encoded).last_hidden_state.float().cpu()
        residue_mask = encoded["attention_mask"].cpu().bool() & ~special
        for (row, piece), states, mask in zip(chunk, hidden, residue_mask):
            if int(mask.sum()) != len(piece):
                raise RuntimeError(f"tokenizer produced {int(mask.sum())} residue tokens for {len(piece)} residues")
            sums[row] += states[mask].double().sum(dim=0).numpy()
            counts[row] += len(piece)
    return (sums / counts[:, None]).astype(np.float32)
