from pathlib import Path

import pytest

from core.pretraining_config import create_pretraining_config, load_config_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "core" / "pretraining_config_multidomain_1m_coconut.yaml"
SINGLE_DOMAIN = {
    "protein": ("pdb", "protein"),
    "rna": ("rna", "rna"),
    "unimol": ("unimol", "unimol"),
    "coconut": ("metabolite", "coconut"),
}
ALLOWED_TOP_LEVEL_DIFFERENCES = {"wandb_run_name", "multi_domain"}


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_override_merges_nested_keys_and_replaces_domain_weights(tmp_path):
    _write(tmp_path / "base.yaml", (
        "lr: 0.1\n"
        "multi_domain:\n"
        "  datasets: {a: {enabled: true, x: 1}, b: {enabled: true}}\n"
        "  atom_sampling: {domain_weights: {a: 1, b: 2}, epoch_num_batches: 10}\n"
    ))
    child = _write(tmp_path / "child.yaml", (
        "base_config: base.yaml\n"
        "multi_domain:\n"
        "  datasets: {b: {enabled: false}}\n"
        "  atom_sampling: {domain_weights: {a: 1}}\n"
    ))
    config = load_config_yaml(child)
    assert config["lr"] == 0.1
    assert config["multi_domain"]["datasets"] == {"a": {"enabled": True, "x": 1}, "b": {"enabled": False}}
    assert config["multi_domain"]["atom_sampling"] == {"domain_weights": {"a": 1}, "epoch_num_batches": 10}
    assert "base_config" not in config


def test_override_key_missing_from_base_raises(tmp_path):
    _write(tmp_path / "base.yaml", "multi_domain: {datasets: {a: {enabled: true}}}\n")
    child = _write(tmp_path / "child.yaml", (
        "base_config: base.yaml\nmulti_domain: {datasets: {a: {enabeld: false}}}\n"
    ))
    with pytest.raises(KeyError, match="multi_domain.datasets.a.enabeld"):
        load_config_yaml(child)


def test_base_config_cycle_raises(tmp_path):
    _write(tmp_path / "a.yaml", "base_config: b.yaml\nlr: 1\n")
    _write(tmp_path / "b.yaml", "base_config: a.yaml\nlr: 2\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config_yaml(tmp_path / "a.yaml")


def test_plain_config_is_returned_unchanged():
    assert load_config_yaml(BASE) == __import__("yaml").safe_load(BASE.read_text())


@pytest.mark.parametrize("domain", sorted(SINGLE_DOMAIN))
def test_single_domain_config_differs_from_base_only_in_domain_selection(domain):
    dataset_key, weight_key = SINGLE_DOMAIN[domain]
    base = load_config_yaml(BASE)
    config = load_config_yaml(REPO_ROOT / "core" / f"pretraining_config_{domain}_only.yaml")

    differing = {key for key in base.keys() | config.keys() if base.get(key) != config.get(key)}
    assert differing <= ALLOWED_TOP_LEVEL_DIFFERENCES

    enabled = {name for name, spec in config["multi_domain"]["datasets"].items() if spec.get("enabled")}
    assert enabled == {dataset_key}
    assert config["multi_domain"]["datasets"][dataset_key] == base["multi_domain"]["datasets"][dataset_key]

    sampling, base_sampling = config["multi_domain"]["atom_sampling"], base["multi_domain"]["atom_sampling"]
    assert sampling["domain_weights"] == {weight_key: 1}
    assert {k: v for k, v in sampling.items() if k not in {"domain_weights", "epoch_atom_budget"}} == {
        k: v for k, v in base_sampling.items() if k not in {"domain_weights", "epoch_atom_budget"}
    }
    assert {k: v for k, v in config["multi_domain"].items() if k not in {"datasets", "atom_sampling"}} == {
        k: v for k, v in base["multi_domain"].items() if k not in {"datasets", "atom_sampling"}
    }
    create_pretraining_config(**config)


@pytest.mark.parametrize("stem", ["pretraining_config_multidomain_1m", "pretraining_config_multidomain_1m_coconut"])
def test_unimol_hidden_child_differs_from_its_base_only_in_the_hiding_flag_and_run_name(stem):
    base = load_config_yaml(REPO_ROOT / "core" / f"{stem}.yaml")
    child = load_config_yaml(REPO_ROOT / "core" / f"{stem}_unimol_hidden.yaml")

    differing = {key for key in base.keys() | child.keys() if base.get(key) != child.get(key)}
    assert differing == {"wandb_run_name", "unimol_masked_geometry_hiding"}
    assert child["wandb_run_name"] != base["wandb_run_name"]
    assert base["unimol_masked_geometry_hiding"] == {"enabled": False, "bond_threshold_angstrom": 2.1}
    assert child["unimol_masked_geometry_hiding"] == {"enabled": True, "bond_threshold_angstrom": 2.1}
    assert create_pretraining_config(**child).unimol_masked_geometry_hiding["enabled"] is True


def test_every_shipped_yaml_passes_the_unknown_key_check():
    from core.pretraining_config import check_yaml_keys
    for path in sorted((REPO_ROOT / "core").glob("pretraining_config_*.yaml")):
        check_yaml_keys(load_config_yaml(path), path.name)


def test_unknown_yaml_key_is_refused_but_checkpoint_style_configs_still_load():
    from core.pretraining_config import check_yaml_keys
    values = load_config_yaml(BASE)
    check_yaml_keys(values, "base")
    with pytest.raises(ValueError, match="unknown config key.*lr_typo"):
        check_yaml_keys({**values, "lr_typo": 1}, "typo.yaml")
    # old checkpoints carry keys that are no longer fields (e.g. temperature-era leftovers);
    # create_pretraining_config must keep accepting them.
    assert create_pretraining_config(**{**values, "lr_typo": 1, "use_hybrid_edges": True}).lr_typo == 1
