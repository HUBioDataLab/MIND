"""SaProt baseline (Su et al., ICLR 2024): structure-aware tokens, pooled like MIND ``mean_residues``.

Each residue is one token: the amino acid (upper case) followed by its Foldseek
3Di state (lower case), e.g. ``Md``. ``#`` replaces the 3Di letter for residues
MIND masks for an incomplete sidechain and for residues Foldseek has no state
for (SaProt's unknown-structure letter); the amino acid is always kept. The
residues are exactly those of the processed MIND split, with 3Di states from the
cache written by ``downstream.baselines.foldseek_3di``.

Token ids are looked up directly in SaProt's vocabulary (no string splitting),
and a residue that is not in the vocabulary raises. Chains longer than 1022
residues are embedded in contiguous windows and averaged residue-weighted, as
for ESM-2. Weights are downloaded by HF on first use into the explicit
``cache_dir``, at the revision pinned in ``registry.yaml``.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

from downstream.baselines.esm2 import MAX_RESIDUES, check_loaded_revision, pinned_revision
from downstream.baselines.foldseek_3di import NO_STATE, THREE_DI_ALPHABET

SAPROT_MODELS = {
    "saprot_650m_pdb": "westlake-repl/SaProt_650M_PDB",   # AF2 + PDB structures; experimental chains here
    "saprot_650m_af2": "westlake-repl/SaProt_650M_AF2",   # AF2 structures only (pLDDT < 70 masked as '#')
    "saprot_35m_af2": "westlake-repl/SaProt_35M_AF2",
}
POOLING = "mean_residues"
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def residue_tokens(sequence: str, three_di: str, masked: Sequence[int]) -> List[str]:
    """SaProt tokens ``<AA><3di|#>`` for one chain; lengths and alphabets must match exactly."""
    if len(sequence) != len(three_di):
        raise ValueError(f"sequence has {len(sequence)} residues but 3Di string has {len(three_di)}")
    masked = set(int(i) for i in masked)
    if masked and (min(masked) < 0 or max(masked) >= len(sequence)):
        raise ValueError(f"masked index outside 0..{len(sequence) - 1}")
    tokens = []
    for index, (amino, state) in enumerate(zip(sequence, three_di)):
        if amino not in AMINO_ACIDS:
            raise ValueError(f"residue {index}: amino acid {amino!r} is not one of the 20 standard letters")
        if state != NO_STATE and state not in THREE_DI_ALPHABET:
            raise ValueError(f"residue {index}: 3Di letter {state!r} is not a Foldseek state")
        tokens.append(amino + (NO_STATE if index in masked or state == NO_STATE else state.lower()))
    return tokens


def load_saprot(name: str, device, *, cache_dir: str, local_files_only: bool = False):
    if name not in SAPROT_MODELS:
        raise ValueError(f"unknown SaProt model {name!r}; choose from {sorted(SAPROT_MODELS)}")
    from transformers import EsmModel, EsmTokenizer

    repo = SAPROT_MODELS[name]
    revision = pinned_revision(name, repo)
    tokenizer = EsmTokenizer.from_pretrained(repo, revision=revision, cache_dir=cache_dir,
                                             local_files_only=local_files_only)
    model = EsmModel.from_pretrained(
        repo, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only, add_pooling_layer=False
    )
    check_loaded_revision(model, revision, repo)
    return tokenizer, model.to(device).eval()


def _windows(tokens: Sequence[str]) -> List[List[str]]:
    count = max(1, -(-len(tokens) // MAX_RESIDUES))
    bounds = np.linspace(0, len(tokens), count + 1).round().astype(int)
    return [list(tokens[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]


def token_ids(tokenizer, tokens: Sequence[str]) -> List[int]:
    """``<cls> tokens <eos>`` ids; every residue token must be in the vocabulary."""
    vocab = tokenizer.get_vocab()
    unknown = sorted({t for t in tokens if t not in vocab})
    if unknown:
        raise ValueError(f"tokens not in the SaProt vocabulary: {unknown[:10]}")
    return [tokenizer.cls_token_id, *(vocab[t] for t in tokens), tokenizer.eos_token_id]


@torch.no_grad()
def embed_tokens(tokenizer, model, chains: Sequence[Sequence[str]], device, *, batch_size: int = 8) -> np.ndarray:
    """Mean of last-layer residue states (CLS/EOS/padding excluded), one row per chain."""
    pieces = [(row, piece) for row, tokens in enumerate(chains) for piece in _windows(tokens)]
    if any(not piece for _, piece in pieces):
        raise ValueError("empty chain")
    sums = np.zeros((len(chains), model.config.hidden_size), dtype=np.float64)
    counts = np.zeros(len(chains))
    for start in range(0, len(pieces), batch_size):
        chunk = pieces[start:start + batch_size]
        ids = [token_ids(tokenizer, piece) for _, piece in chunk]
        width = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), width), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(ids), width), dtype=torch.long)
        for row, x in enumerate(ids):
            input_ids[row, :len(x)] = torch.tensor(x)
            attention[row, :len(x)] = 1
        hidden = model(input_ids=input_ids.to(device), attention_mask=attention.to(device)).last_hidden_state
        hidden = hidden.float().cpu()
        for (row, piece), states in zip(chunk, hidden):
            sums[row] += states[1:1 + len(piece)].double().sum(dim=0).numpy()
            counts[row] += len(piece)
    return (sums / counts[:, None]).astype(np.float32)


def split_tokens(records: Sequence[dict]) -> List[List[str]]:
    """Token lists for cache records (``foldseek_3di.load_cache`` output)."""
    return [residue_tokens(r["seq"], r["3di"], r["masked"]) for r in records]
