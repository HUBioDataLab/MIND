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


def hash_only_cache(root, repo, revision, tokenizer, model):
    """HF cache holding only ``snapshots/<revision>`` (no ``refs/main``), as ``fetch_models`` leaves it."""
    snapshot = root / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    tokenizer.save_pretrained(snapshot)
    model.save_pretrained(snapshot)
    return root


def test_esm2_loads_the_registry_revision_from_a_hash_only_cache(tiny_esm, tmp_path):
    from downstream.baselines.fetch_models import load_registry

    name = "esm2_t6_8M_UR50D"
    revision = load_registry()[name]["revision"]
    cache = hash_only_cache(tmp_path / "hub", esm2.ESM2_MODELS[name], revision, *tiny_esm)
    _, model = esm2.load_esm2(name, "cpu", cache_dir=str(cache), local_files_only=True)
    assert model.config._commit_hash == revision


def test_registry_and_loader_repo_must_agree(monkeypatch):
    monkeypatch.setitem(esm2.ESM2_MODELS, "esm2_t6_8M_UR50D", "someone/else")
    with pytest.raises(ValueError, match="registry"):
        esm2.load_esm2("esm2_t6_8M_UR50D", "cpu", cache_dir="/nonexistent", local_files_only=True)


def test_a_loaded_revision_other_than_the_pin_fails(tiny_esm):
    _, model = tiny_esm
    model.config._commit_hash = "b" * 40
    with pytest.raises(RuntimeError, match="revision"):
        esm2.check_loaded_revision(model, "a" * 40, "facebook/esm2_t6_8M_UR50D")
    model.config._commit_hash = None
    with pytest.raises(RuntimeError, match="revision"):
        esm2.check_loaded_revision(model, "a" * 40, "facebook/esm2_t6_8M_UR50D")


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


def _lba_task_with_processed_splits(root, tag="identity30"):
    """LBA config plus tiny processed train/val/test files for one split tag (ESM-readable sequences)."""
    from downstream.common import complex_graph as cg
    from downstream.tasks.lba import dataset as lba
    from test_downstream_lba import SETTINGS, simple_item

    config = lba.load_task_config(data_root=root)
    out = lba.processed_dir(config, tag)
    out.mkdir(parents=True)
    for split in config["splits"]:
        results = [lba.process_item(simple_item(f"{split}{i}", target=5.0 + i), SETTINGS) for i in range(2)]
        torch.save({
            "schema": cg.complex_edge_free_schema(3, {"builder": "tests"}), "policy": "dropbb_masksc",
            "split": split, "split_tag": tag, "ids": [r.complex_id for r in results],
            "targets": torch.tensor([r.target for r in results], dtype=torch.float32),
            "samples": [r.sample for r in results], "ligands": [r.ligand for r in results],
            "sequences": [r.sequence for r in results], "qc": {},
        }, out / f"{split}.pt")
    return config


@pytest.mark.parametrize("flags, tag, representation", [
    ([], "identity30", "joint"),                                       # defaults: record must still name them
    (["--split-tag", "identity60", "--representation", "joint"], "identity60", "joint"),
])
def test_lba_embedding_record_names_the_split_tag_and_representation_even_when_defaulted(
        tmp_path, tiny_esm, monkeypatch, flags, tag, representation):
    import json

    from downstream import embed
    from downstream.common.registry import embeddings_path

    _lba_task_with_processed_splits(tmp_path / "data", tag)
    monkeypatch.setattr(esm2, "load_esm2", lambda *a, **k: tiny_esm)
    args = ["--task", "lba", "--model", "esm2_t6_8M_UR50D", "--device", "cpu", "--hf-cache-dir", str(tmp_path),
            "--data-root", str(tmp_path / "data"), "--outputs-root", str(tmp_path / "out"), "--splits", "val",
            *flags]
    assert embed.main(args) == 0
    out = embeddings_path("lba", "esm2_t6_8M_UR50D", f"{tag}.{representation}.val", root=tmp_path / "out")
    record = json.loads(out.with_suffix(".json").read_text())
    assert record["split_tag"] == tag and record["representation"] == representation


def test_non_lba_embedding_record_has_no_split_tag(tmp_path, tiny_esm, monkeypatch):
    import json

    from downstream import embed
    from downstream.common.registry import embeddings_path

    config = _write_task(tmp_path / "data", max_residues=30)
    assert dataset.build_processed(config, splits=["valid"], workers=1) == 0
    monkeypatch.setattr(esm2, "load_esm2", lambda *a, **k: tiny_esm)
    monkeypatch.setattr(dataset, "load_task_config", lambda *a, **k: config)
    args = ["--task", "ec", "--model", "esm2_t6_8M_UR50D", "--device", "cpu", "--hf-cache-dir", str(tmp_path),
            "--outputs-root", str(tmp_path / "out"), "--splits", "valid"]
    assert embed.main(args) == 0
    record = json.loads(embeddings_path("ec", "esm2_t6_8M_UR50D", "valid", root=tmp_path / "out")
                        .with_suffix(".json").read_text())
    assert record["split_tag"] is None and record["representation"] is None
