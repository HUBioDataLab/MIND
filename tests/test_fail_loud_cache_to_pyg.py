"""cache -> PyG conversion: failures stop the build; intended size filtering and clean EOF do not."""

import pickle

import pytest

from data_loading.cache_to_pyg import OptimizedUniversalDataset, load_molecules_iteratively
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule


def _molecule(mol_id, num_atoms):
    atoms = [
        UniversalAtom(element="C", position=(1.5 * i, 0.0, 0.0), pos_code="sm",
                      block_idx=0, atom_idx_in_block=i, entity_idx=0)
        for i in range(num_atoms)
    ]
    return UniversalMolecule(id=mol_id, dataset_type="unimol",
                             blocks=[UniversalBlock(symbol="F", atoms=atoms)], properties={})


def _write(path, molecules):
    with open(path, "wb") as handle:
        for molecule in molecules:
            pickle.dump(molecule, handle)
    return str(path)


def _build(tmp_path, molecules, **kwargs):
    source = _write(tmp_path / "universal_unimol_chunk_0.pkl", molecules)
    return OptimizedUniversalDataset(
        root=str(tmp_path / "root"), universal_cache_path=source,
        conversion_num_workers=1, **kwargs,
    )


def test_unconvertible_molecule_fails_the_build_and_names_it(tmp_path):
    with pytest.raises(RuntimeError, match=r"1 of 3 .*empty_one") as caught:
        _build(tmp_path, [_molecule("a", 4), _molecule("empty_one", 0), _molecule("b", 5)])
    assert "no atoms" in str(caught.value)
    assert not list((tmp_path / "root" / "processed").glob("*.pt"))


def test_size_filtered_molecules_are_intended_and_do_not_fail(tmp_path):
    dataset = _build(tmp_path, [_molecule("a", 4), _molecule("big", 40), _molecule("b", 5)],
                     molecule_max_atoms=10)
    assert len(dataset) == 2


def test_truncated_pickle_record_is_an_error_not_a_short_dataset(tmp_path):
    path = _write(tmp_path / "t.pkl", [_molecule("a", 4), _molecule("b", 5)])
    raw = open(path, "rb").read()
    open(path, "wb").write(raw[:-10])
    with pytest.raises(pickle.UnpicklingError):
        list(load_molecules_iteratively(path))


def test_clean_pickle_stream_ends_normally(tmp_path):
    path = _write(tmp_path / "c.pkl", [_molecule("a", 4), _molecule("b", 5)])
    assert [m.id for m in load_molecules_iteratively(path)] == ["a", "b"]
