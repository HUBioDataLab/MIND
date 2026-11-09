"""SaProt baseline: Foldseek 3Di alignment to MIND-kept residues, token strings, pooling, embed wiring.

CPU only and offline; the real-Foldseek and real-tokenizer checks skip when the
binary / cached tokenizer are absent.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from downstream import embed
from downstream.baselines import foldseek_3di as fs
from downstream.baselines import saprot
from downstream.common.registry import embeddings_path, sha256_file
from downstream.common.structure_io import parse_structure_text
from downstream.tasks.ec import dataset
from downstream_helpers import pdb_line, residue_atoms
from test_downstream_ec_dataset import _write_task

FOLDSEEK = os.environ.get("MIND_FOLDSEEK", "/home/yusuf/tools/foldseek/10-941cd33/foldseek/bin/foldseek")
HF_CACHE = os.path.join(os.environ.get("MIND_MODELS_ROOT", "/home/yusuf/models"), "huggingface", "hub")
AMINO = "ACDEFGHIKLMNPQRSTVWY"
MASK = {"policy": "drop_residue_missing_backbone", "incomplete_sidechain": "mask"}


def cand(resseq, name, letter, absent=False):
    return fs.Candidate((resseq, "", name), letter, absent)


# ---------------------------------------------------------------------------
# Token strings
# ---------------------------------------------------------------------------

def test_residue_tokens_mask_and_no_state_become_hash():
    tokens = saprot.residue_tokens("MKAG", "DV#P", masked=[1])
    assert tokens == ["Md", "K#", "A#", "Gp"]


@pytest.mark.parametrize("sequence, three_di, masked, message", [
    ("MKA", "DV", [], "3Di string has 2"),
    ("MXA", "DVP", [], "not one of the 20"),
    ("MKA", "DVb", [], "not a Foldseek state"),
    ("MKA", "DVP", [3], "masked index"),
])
def test_residue_tokens_fail_loudly(sequence, three_di, masked, message):
    with pytest.raises(ValueError, match=message):
        saprot.residue_tokens(sequence, three_di, masked)


# ---------------------------------------------------------------------------
# Alignment of Foldseek output to the kept residues
# ---------------------------------------------------------------------------

def test_alignment_identity_and_nonstandard_absent():
    candidates = [cand(1, "ALA", "A"), cand(2, "CSO", None, True), cand(3, "GLY", "G"), cand(4, "SER", "S")]
    kept = [(1, "", "ALA"), (3, "", "GLY"), (4, "", "SER")]
    # Foldseek dropped the modified residue
    assert fs.align_kept_residues("c", candidates, "AGS", "PQR", kept, "AGS") == "PQR"
    # Foldseek kept it (as its parent letter): the kept residues skip its state
    assert fs.align_kept_residues("c", candidates, "ACGS", "PVQR", kept, "AGS") == "PQR"


def test_alignment_shadowed_kept_residue_gets_hash():
    candidates = [cand(1, "ALA", "A"), cand(2, "DDZ", None, True), cand(3, "GLY", "G")]
    kept = [(1, "", "ALA"), (2, "", "CYS"), (3, "", "GLY")]
    assert fs.align_kept_residues("c", candidates, "AXG", "PVQ", kept, "ACG", {(2, "", "CYS")}) == "P#Q"


@pytest.mark.parametrize("aa, three_di, message", [
    ("AWS", "PQR", "no mismatch-free alignment"),       # sequence mismatch
    ("AGSS", "PQRR", "more than the"),                   # Foldseek longer than the structure
    ("AG", "PQ", "only 0 may be absent"),                # a standard residue missing from Foldseek
    ("AGS", "PQ", "3Di states"),                         # malformed Foldseek entry
])
def test_alignment_mismatches_raise(aa, three_di, message):
    candidates = [cand(1, "ALA", "A"), cand(2, "GLY", "G"), cand(3, "SER", "S")]
    kept = [(1, "", "ALA"), (2, "", "GLY"), (3, "", "SER")]
    with pytest.raises(fs.AlignmentError, match=message):
        fs.align_kept_residues("c", candidates, aa, three_di, kept, "AGS")


def test_alignment_ambiguous_kept_residue_raises():
    candidates = [cand(1, "CSO", None, True), cand(2, "ALA", "A"), cand(3, "CSO", None, True)]
    with pytest.raises(fs.AlignmentError, match="ambiguously"):
        fs.align_kept_residues("c", candidates, "AA", "PQ", [(2, "", "ALA")], "A")


def test_alignment_kept_residue_without_ca_raises():
    with pytest.raises(fs.AlignmentError, match="not a parsed CA residue"):
        fs.align_kept_residues("c", [cand(1, "ALA", "A")], "A", "P", [(5, "", "GLY")], "G")


def _microheterogeneity_pdb():
    """ALA1 GLY2, DDZ3 then CYS3 (same number), SER4 missing OG, LYS5, a water."""
    residues = [("ALA", 1, "ATOM", residue_atoms("ALA", 0)), ("GLY", 2, "ATOM", residue_atoms("GLY", 1)),
                ("DDZ", 3, "HETATM", residue_atoms("CYS", 2)), ("CYS", 3, "ATOM", residue_atoms("CYS", 2)),
                ("SER", 4, "ATOM", residue_atoms("SER", 3, drop=("OG",))), ("LYS", 5, "ATOM", residue_atoms("LYS", 4)),
                ("HOH", 6, "HETATM", [("O", "O", (40.0, 0.0, 0.0))])]
    lines, serial = [], 1
    for resname, resseq, record, atoms in residues:
        for name, element, xyz in atoms:
            lines.append(pdb_line(serial, name, resname, "A", resseq, xyz, element, record=record))
            serial += 1
    return "\n".join(lines + ["TER", "END"]) + "\n"


def test_candidates_follow_gemmi_first_conformer():
    model, _ = parse_structure_text(_microheterogeneity_pdb(), "pdb", "t")
    candidates, shadowed = fs.candidate_residues(next(iter(model)))
    assert [c.key for c in candidates] == [(1, "", "ALA"), (2, "", "GLY"), (3, "", "DDZ"), (4, "", "SER"),
                                           (5, "", "LYS")]
    assert [c.letter for c in candidates] == ["A", "G", None, "S", "K"]
    assert shadowed == {(3, "", "CYS")}


def _job(path, foldseek_chains, sequence):
    return ("1XYZ-A", str(path), None, foldseek_chains, sequence, {"missing_atoms": MASK, "mse_to_met": True})


def test_chain_3di_end_to_end_with_masked_and_shadowed(tmp_path):
    path = tmp_path / "1XYZ-A_1.pdb"
    path.write_text(_microheterogeneity_pdb())
    out = fs.chain_3di(_job(path, {"A": ("AGXSK", "PQRST")}, "AGCSK"))
    assert out["seq"] == "AGCSK" and out["3di"] == "PQ#ST" and out["masked"] == [3]
    assert saprot.residue_tokens(out["seq"], out["3di"], out["masked"]) == ["Ap", "Gq", "C#", "S#", "Kt"]
    with pytest.raises(fs.AlignmentError, match="differs from the processed split"):
        fs.chain_3di(_job(path, {"A": ("AGXSK", "PQRST")}, "AGASK"))
    with pytest.raises(fs.AlignmentError, match="no entry for a chain with 5"):
        fs.chain_3di(_job(path, {}, "AGCSK"))


def test_chain_3di_short_chain_without_foldseek_entry_is_all_hash(tmp_path):
    path = tmp_path / "1XYZ-A_1.pdb"
    lines = [pdb_line(i + 1, n, "ALA" if i < 5 else "GLY", "A", 1 + i // 5, xyz, e)
             for i, (n, e, xyz) in enumerate(residue_atoms("ALA", 0) + residue_atoms("GLY", 1))]
    path.write_text("\n".join(lines + ["END"]) + "\n")
    out = fs.chain_3di(_job(path, {}, "AG"))
    assert out["3di"] == "##"


def test_parse_descriptor_tsv_rejects_bad_rows(tmp_path):
    good = tmp_path / "ok.tsv"
    good.write_text("f.pdb_A\tAG\tDV\t0,0\n")
    assert fs.parse_descriptor_tsv(good) == {"f.pdb_A": ("AG", "DV")}
    bad = tmp_path / "bad.tsv"
    bad.write_text("f.pdb_A\tAG\tDB\t0,0\n")
    with pytest.raises(fs.AlignmentError, match="outside the alphabet"):
        fs.parse_descriptor_tsv(bad)


@pytest.mark.skipif(not Path(FOLDSEEK).is_file(), reason="Foldseek binary not installed")
def test_real_foldseek_matches_first_conformer_rule(tmp_path):
    path = tmp_path / "1XYZ-A_1.pdb"
    path.write_text(_microheterogeneity_pdb())
    work = tmp_path / "work"
    work.mkdir()
    out = fs.run_foldseek(FOLDSEEK, [path], threads=1, workdir=work)
    (aa, three_di), = out[path.name].values()
    assert len(aa) == 5 and aa[:2] == "AG" and aa[3:] == "SK"   # DDZ read, CYS shadowed
    result = fs.chain_3di(_job(path, out[path.name], "AGCSK"))
    assert result["3di"][2] == "#" and len(result["3di"]) == 5


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _saprot_vocab():
    return ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>",
            *[a + s for a in AMINO + "X#" for s in "acdefghiklmnpqrstvwy#"]]


@pytest.fixture
def tiny_saprot(tmp_path):
    from transformers import EsmConfig, EsmModel, EsmTokenizer

    vocab = _saprot_vocab()
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = EsmTokenizer(str(tmp_path / "vocab.txt"))
    torch.manual_seed(0)
    config = EsmConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=32, max_position_embeddings=64, pad_token_id=1, mask_token_id=4,
                       position_embedding_type="rotary", token_dropout=True)
    return tokenizer, EsmModel(config, add_pooling_layer=False).eval()


def test_token_ids_match_string_tokenizer_and_reject_unknown(tiny_saprot):
    tokenizer, _ = tiny_saprot
    tokens = ["Md", "K#", "Av"]
    assert saprot.token_ids(tokenizer, tokens) == tokenizer("".join(tokens))["input_ids"]
    with pytest.raises(ValueError, match="not in the SaProt vocabulary"):
        saprot.token_ids(tokenizer, ["Mb"])


def test_embed_mean_excludes_special_and_padding(tiny_saprot):
    tokenizer, model = tiny_saprot
    chains = [["Md", "K#", "Av", "Gp", "Sq"], ["Wd", "Y#"]]
    batched = saprot.embed_tokens(tokenizer, model, chains, "cpu", batch_size=2)
    single = saprot.embed_tokens(tokenizer, model, chains[1:], "cpu", batch_size=1)
    np.testing.assert_allclose(batched[1], single[0], atol=1e-5)
    ids = torch.tensor([saprot.token_ids(tokenizer, chains[1])])
    with torch.no_grad():
        manual = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, 1:-1].mean(0)
    np.testing.assert_allclose(single[0], manual.numpy(), atol=1e-5)
    np.testing.assert_array_equal(batched, saprot.embed_tokens(tokenizer, model, chains, "cpu", batch_size=2))


def test_embed_windows_long_chains(tiny_saprot, monkeypatch):
    tokenizer, model = tiny_saprot
    monkeypatch.setattr(saprot, "MAX_RESIDUES", 4)
    tokens = ["Md", "Kv", "Ap", "Gq", "Sd", "Ww", "Yy", "Ll", "Ii"]
    assert saprot._windows(tokens) == [tokens[:3], tokens[3:6], tokens[6:]]
    whole = saprot.embed_tokens(tokenizer, model, [tokens], "cpu")
    parts = saprot.embed_tokens(tokenizer, model, [tokens[:3], tokens[3:6], tokens[6:]], "cpu")
    np.testing.assert_allclose(whole[0], parts.mean(axis=0), atol=1e-5)


def test_unknown_saprot_name_fails():
    with pytest.raises(ValueError, match="unknown SaProt model"):
        saprot.load_saprot("saprot_15b", "cpu", cache_dir="/nonexistent")


def _real_tokenizer():
    from transformers import EsmTokenizer
    try:
        return EsmTokenizer.from_pretrained(saprot.SAPROT_MODELS["saprot_650m_pdb"], cache_dir=HF_CACHE,
                                            local_files_only=True)
    except Exception:  # not cached on this machine
        return None


def test_real_saprot_tokenizer_matches_direct_ids():
    tokenizer = _real_tokenizer()
    if tokenizer is None:
        pytest.skip("SaProt tokenizer not in the local HF cache")
    tokens = saprot.residue_tokens("MKAGWC", "DV#PLA", masked=[4])
    assert saprot.token_ids(tokenizer, tokens) == tokenizer("".join(tokens))["input_ids"]
    assert len(tokenizer.get_vocab()) == 446


# ---------------------------------------------------------------------------
# downstream.embed wiring
# ---------------------------------------------------------------------------

def _fake_cache(config, split_name, *, sha=None):
    split = dataset.ProcessedSplit(dataset.processed_dir(config) / f"{split_name}.pt")
    records = [{"chain": c, "seq": s, "3di": ("DVPL" * len(s))[:len(s)], "masked": [0]}
               for c, s in zip(split.chains, split.sequences)]
    path = fs.cache_dir(config) / f"{split_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {"processed_sha256": sha or sha256_file(split.path), "foldseek_version": "test"}
    path.write_text(json.dumps({"format": fs.CACHE_FORMAT, "provenance": provenance, "chains": records}))
    return split


def test_embed_main_saprot_writes_embeddings(tmp_path, monkeypatch, tiny_saprot):
    config = _write_task(tmp_path / "data", max_residues=30)
    assert dataset.build_processed(config, splits=["valid", "test"], workers=1) == 0
    split = _fake_cache(config, "valid")
    _fake_cache(config, "test", sha="0" * 64)
    monkeypatch.setattr(dataset, "load_task_config", lambda *a, **k: config)
    monkeypatch.setattr(saprot, "load_saprot", lambda *a, **k: tiny_saprot)
    args = ["--task", "ec", "--model", "saprot_650m_pdb", "--device", "cpu", "--hf-cache-dir", str(tmp_path),
            "--outputs-root", str(tmp_path / "out"), "--threads", str(torch.get_num_threads())]
    assert embed.main(args + ["--splits", "valid"]) == 0
    out = embeddings_path("ec", "saprot_650m_pdb", "valid", root=tmp_path / "out")
    data = np.load(out)
    assert data["emb_mean_residues"].shape == (len(split.chains), 16)
    assert list(data["chains"]) == split.chains
    record = json.loads(out.with_suffix(".json").read_text())
    assert record["foldseek_version"] == "test" and record["model"] == "westlake-repl/SaProt_650M_PDB"
    assert record["foldseek_3di_sha256"] == sha256_file(fs.cache_dir(config) / "valid.json")
    with pytest.raises(ValueError, match="another version"):
        embed.main(args + ["--splits", "test"])


@pytest.mark.parametrize("task", sorted(set(embed.TASKS) - {"ec"}))
def test_embed_main_saprot_refuses_tasks_without_3di_cache(task, capsys):
    """SaProt needs the Foldseek 3Di cache, which exists for the ec task only."""
    with pytest.raises(SystemExit) as raised:
        embed.main(["--task", task, "--model", "saprot_650m_pdb", "--device", "cpu"])
    assert raised.value.code == 2
    assert "ec task only" in capsys.readouterr().err


def test_load_cache_rejects_other_chain_order(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"format": fs.CACHE_FORMAT, "provenance": {},
                                "chains": [{"chain": "B", "seq": "A", "3di": "D", "masked": []}]}))
    with pytest.raises(ValueError, match="chain list differs"):
        fs.load_cache(path, ["A"], ["A"])
    with pytest.raises(ValueError, match="sequence/3Di differs"):
        fs.load_cache(path, ["B"], ["G"])
