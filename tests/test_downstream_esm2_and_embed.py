"""ESM-2 pooling on a tiny offline model, and the MIND split-embedding wiring on CPU."""

import numpy as np
import pytest
import torch

from downstream.baselines import esm2
from downstream.embed import embed_split_mind
from downstream.tasks.ec import dataset
from test_downstream_ec_dataset import _write_task
from test_downstream_encoder import _tiny_checkpoint

AMINO = "LAGVSERTIDPKQNFYMHWC"


@pytest.fixture
def tiny_esm(tmp_path):
    from transformers import EsmConfig, EsmModel, EsmTokenizer

    vocab = ["<cls>", "<pad>", "<eos>", "<unk>", *AMINO, "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = EsmTokenizer(str(tmp_path / "vocab.txt"))
    torch.manual_seed(0)
    config = EsmConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=32, max_position_embeddings=64, pad_token_id=1, mask_token_id=len(vocab) - 1,
                       position_embedding_type="rotary", token_dropout=False)
    return tokenizer, EsmModel(config, add_pooling_layer=False).eval()


def test_esm2_mean_excludes_special_tokens_and_padding(tiny_esm):
    tokenizer, model = tiny_esm
    sequences = ["ACDEFG", "KL"]
    batched = esm2.embed_sequences(tokenizer, model, sequences, "cpu", batch_size=2)
    single = esm2.embed_sequences(tokenizer, model, sequences[1:], "cpu", batch_size=1)
    np.testing.assert_allclose(batched[1], single[0], atol=1e-5)  # padding does not leak
    encoded = tokenizer(["KL"], return_tensors="pt")
    with torch.no_grad():
        states = model(**encoded).last_hidden_state[0, 1:-1].mean(dim=0).numpy()
    np.testing.assert_allclose(single[0], states, atol=1e-5)


def test_esm2_windows_long_sequences(tiny_esm, monkeypatch):
    tokenizer, model = tiny_esm
    monkeypatch.setattr(esm2, "MAX_RESIDUES", 4)
    assert esm2._windows("ACDEFGHIK") == ["ACD", "EFG", "HIK"]
    out = esm2.embed_sequences(tokenizer, model, ["ACDEFGHIK"], "cpu")
    parts = esm2.embed_sequences(tokenizer, model, ["ACD", "EFG", "HIK"], "cpu")
    np.testing.assert_allclose(out[0], parts.mean(axis=0), atol=1e-5)


def test_unknown_esm_name_fails():
    with pytest.raises(ValueError, match="unknown ESM-2 model"):
        esm2.load_esm2("esm2_t48_15B_UR50D", "cpu", cache_dir="/nonexistent")


def test_embed_split_mind_on_cpu_matches_chain_count(cpu_flash, tmp_path):
    config = _write_task(tmp_path / "data")
    assert dataset.build_processed(config, splits=["valid"], workers=1) == 0
    split = dataset.ProcessedSplit(dataset.processed_dir(config) / "valid.pt")
    model = _tiny_checkpoint(tmp_path / "tiny.ckpt").eval()
    first = embed_split_mind(model, split, "cpu", seed=0, poolings=["mean_atoms", "ca"], max_atoms=100)
    second = embed_split_mind(model, split, "cpu", seed=0, poolings=["mean_atoms", "ca"], max_atoms=100)
    assert first["mean_atoms"].shape == (len(split.chains), 32)
    for mode in first:
        np.testing.assert_array_equal(first[mode], second[mode])
