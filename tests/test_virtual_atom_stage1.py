import json
import math
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from core.local_frame import (
    LocalFrameError,
    protein_n_ca_c_frame,
    rna_c2_c3_c4_frame,
)
from core.virtual_atom_templates import (
    MINIMUM_TEMPLATE_SEPARATION_ANGSTROM,
    protein_sidechain_template,
    rna_base_template,
)
from data_loading.audit_rna_quarantine_provenance import (
    classify_block,
    has_finite_unique_sugar_frame,
)
from data_loading.audit_rna_anchor_selection import (
    build_candidate_frame,
    is_mirror_rna_molecule,
    select_numerically_dominant_candidate,
)
from data_loading.atom_position_vocab import (
    is_p_only_rna_block,
    rna_molecule_contains_p_only_block,
)
from data_loading.cache_to_pyg import OptimizedUniversalRNADataset
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule


def _protein_anchors(dtype=torch.float64):
    return {
        "N": torch.tensor([-0.7, 1.2, 0.4], dtype=dtype),
        "CA": torch.tensor([0.3, -0.2, 1.1], dtype=dtype),
        "C": torch.tensor([1.5, 0.4, 0.7], dtype=dtype),
    }


def _rotation_matrix(dtype=torch.float64):
    axis = torch.tensor([1.0, 2.0, -1.0], dtype=dtype)
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis
    skew = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=dtype
    )
    identity = torch.eye(3, dtype=dtype)
    angle = torch.tensor(0.73, dtype=dtype)
    return (
        identity * torch.cos(angle)
        + (1.0 - torch.cos(angle)) * torch.outer(axis, axis)
        + torch.sin(angle) * skew
    )


def _assert_fixed_template_contract(points, expected_count):
    assert tuple(points.shape) == (expected_count, 3)
    assert points.dtype == torch.float64
    assert torch.isfinite(points).all()
    assert torch.unique(points, dim=0).shape[0] == expected_count
    assert torch.pdist(points).min() >= MINIMUM_TEMPLATE_SEPARATION_ANGSTROM


def test_type_independent_templates_have_exact_safe_shapes_and_are_bitwise_stable():
    protein_a = protein_sidechain_template(dtype=torch.float64)
    protein_b = protein_sidechain_template(dtype=torch.float64)
    rna_a = rna_base_template(dtype=torch.float64)
    rna_b = rna_base_template(dtype=torch.float64)

    _assert_fixed_template_contract(protein_a, 10)
    _assert_fixed_template_contract(rna_a, 11)
    assert torch.equal(protein_a, protein_b)
    assert torch.equal(rna_a, rna_b)

    # Each call owns its tensor; mutating one caller's result cannot silently
    # alter the geometry returned to another residue or nucleotide.
    protein_a[0, 0] = 999.0
    rna_a[0, 0] = 999.0
    assert torch.equal(protein_b, protein_sidechain_template(dtype=torch.float64))
    assert torch.equal(rna_b, rna_base_template(dtype=torch.float64))


def test_template_api_has_no_residue_nucleotide_or_atom_count_input():
    import inspect

    for function in (protein_sidechain_template, rna_base_template):
        assert tuple(inspect.signature(function).parameters) == ("dtype", "device")


def test_templates_remain_separated_from_canonical_frame_anchors():
    protein_anchors_local = torch.tensor(
        [[-0.55, 1.36, 0.0], [0.0, 0.0, 0.0], [1.53, 0.0, 0.0]],
        dtype=torch.float64,
    )
    rna_anchors_local = torch.tensor(
        [[-0.333, 1.486, 0.0], [0.0, 0.0, 0.0], [1.521, 0.0, 0.0]],
        dtype=torch.float64,
    )
    assert torch.cdist(
        protein_sidechain_template(dtype=torch.float64), protein_anchors_local
    ).min() >= MINIMUM_TEMPLATE_SEPARATION_ANGSTROM
    assert torch.cdist(
        rna_base_template(dtype=torch.float64), rna_anchors_local
    ).min() >= MINIMUM_TEMPLATE_SEPARATION_ANGSTROM


def test_protein_and_rna_template_global_points_are_rigid_motion_equivariant():
    rotation = _rotation_matrix()
    translation = torch.tensor([4.0, -3.5, 8.2], dtype=torch.float64)

    protein_anchors = _protein_anchors()
    protein_frame = protein_n_ca_c_frame(protein_anchors)
    transformed_protein_anchors = {
        name: point @ rotation.transpose(0, 1) + translation
        for name, point in protein_anchors.items()
    }
    transformed_protein_frame = protein_n_ca_c_frame(transformed_protein_anchors)
    protein_local = protein_sidechain_template(dtype=torch.float64)
    assert torch.allclose(
        transformed_protein_frame.local_to_global(protein_local),
        protein_frame.local_to_global(protein_local) @ rotation.transpose(0, 1)
        + translation,
        atol=1.0e-12,
    )

    rna_codes, rna_positions = _rna_frame_table()
    rna_frame = rna_c2_c3_c4_frame(rna_codes, rna_positions)
    transformed_rna_frame = rna_c2_c3_c4_frame(
        rna_codes, rna_positions @ rotation.transpose(0, 1) + translation
    )
    rna_local = rna_base_template(dtype=torch.float64)
    assert torch.allclose(
        transformed_rna_frame.local_to_global(rna_local),
        rna_frame.local_to_global(rna_local) @ rotation.transpose(0, 1)
        + translation,
        atol=1.0e-12,
    )
def test_protein_frame_is_deterministic_right_handed_and_round_trips():
    anchors = _protein_anchors()
    frame_a = protein_n_ca_c_frame(anchors)
    frame_b = protein_n_ca_c_frame(anchors)

    assert torch.equal(frame_a.origin, frame_b.origin)
    assert torch.equal(frame_a.basis, frame_b.basis)
    assert torch.allclose(
        frame_a.basis.transpose(0, 1) @ frame_a.basis,
        torch.eye(3, dtype=frame_a.basis.dtype),
        atol=1.0e-12,
    )
    assert math.isclose(torch.linalg.det(frame_a.basis).item(), 1.0, abs_tol=1.0e-12)

    local_points = torch.tensor(
        [[0.8, 1.1, -0.4], [-1.2, 0.3, 2.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    recovered = frame_a.global_to_local(frame_a.local_to_global(local_points))
    assert torch.allclose(recovered, local_points, atol=1.0e-12)


def test_protein_frame_is_translation_and_rotation_equivariant():
    anchors = _protein_anchors()
    frame = protein_n_ca_c_frame(anchors)
    rotation = _rotation_matrix()
    translation = torch.tensor([4.0, -3.5, 8.2], dtype=torch.float64)
    transformed_anchors = {
        name: point @ rotation.transpose(0, 1) + translation
        for name, point in anchors.items()
    }
    transformed_frame = protein_n_ca_c_frame(transformed_anchors)

    assert torch.allclose(
        transformed_frame.origin,
        frame.origin @ rotation.transpose(0, 1) + translation,
        atol=1.0e-12,
    )
    assert torch.allclose(
        transformed_frame.basis, rotation @ frame.basis, atol=1.0e-12
    )

    local_points = torch.tensor(
        [[0.2, 1.4, -0.9], [1.7, -0.1, 0.6]], dtype=torch.float64
    )
    expected_global = (
        frame.local_to_global(local_points) @ rotation.transpose(0, 1)
        + translation
    )
    assert torch.allclose(
        transformed_frame.local_to_global(local_points),
        expected_global,
        atol=1.0e-12,
    )


def test_protein_frame_reports_explicit_quarantine_reasons():
    cases = [
        ({"CA": [0.0, 0.0, 0.0], "C": [1.0, 0.0, 0.0]}, "missing_anchor"),
        (
            {"N": [0.0, 1.0, 0.0], "CA": [0.0, 0.0, 0.0], "C": [0.0, 0.0, 0.0]},
            "coincident_anchor",
        ),
        (
            {"N": [-1.0, 0.0, 0.0], "CA": [0.0, 0.0, 0.0], "C": [1.0, 0.0, 0.0]},
            "collinear_anchors",
        ),
        (
            {"N": [0.0, 1.0, 0.0], "CA": [0.0, 0.0, 0.0], "C": [float("nan"), 0.0, 0.0]},
            "nonfinite_anchor",
        ),
    ]
    for anchors, expected_reason in cases:
        try:
            protein_n_ca_c_frame(anchors)
        except LocalFrameError as error:
            assert error.reason == expected_reason
        else:
            raise AssertionError(f"Expected LocalFrameError: {expected_reason}")


def _rna_atom(pos_code, position=(0.0, 0.0, 0.0)):
    return SimpleNamespace(pos_code=pos_code, position=position)


def test_rna_quarantine_audit_separates_p_only_from_terminal_phosphate():
    category, _, _ = classify_block([_rna_atom("P")], block_index=0)
    assert category == "p_only"

    sugar_and_base = [
        _rna_atom("O5'"),
        _rna_atom("C5'"),
        _rna_atom("C4'"),
        _rna_atom("O4'"),
        _rna_atom("C3'", (1.0, 1.0, 0.0)),
        _rna_atom("O3'"),
        _rna_atom("C2'", (0.0, 0.0, 0.0)),
        _rna_atom("O2'"),
        _rna_atom("C1'", (0.0, 1.0, 0.0)),
        _rna_atom("N1"),
    ]
    category, missing, duplicates = classify_block(sugar_and_base, block_index=0)
    assert category == "missing_phosphate_at_block_zero"
    assert missing == ("P", "OP1", "OP2")
    assert duplicates == {}
    assert has_finite_unique_sugar_frame(sugar_and_base)


def test_rna_sugar_frame_audit_rejects_missing_and_collinear_anchors():
    valid = [
        _rna_atom("C1'", (0.0, 1.0, 0.0)),
        _rna_atom("C2'", (0.0, 0.0, 0.0)),
        _rna_atom("C3'", (1.0, 0.0, 0.0)),
    ]
    assert has_finite_unique_sugar_frame(valid)
    assert not has_finite_unique_sugar_frame(valid[:-1])
    assert not has_finite_unique_sugar_frame(
        [
            _rna_atom("C1'", (-1.0, 0.0, 0.0)),
            _rna_atom("C2'", (0.0, 0.0, 0.0)),
            _rna_atom("C3'", (1.0, 0.0, 0.0)),
        ]
    )


def test_rna_candidate_frame_is_right_handed_and_atom_order_independent():
    atoms = [
        _rna_atom("C2'", (0.1, 1.2, -0.2)),
        _rna_atom("O2'", (-0.4, 1.0, 0.8)),
        _rna_atom("C3'", (0.0, 0.0, 0.0)),
        _rna_atom("C4'", (1.4, 0.2, 0.1)),
    ]
    frame = build_candidate_frame(atoms, ("C2'", "C3'", "C4'"))
    reordered = build_candidate_frame(
        list(reversed(atoms)), ("C2'", "C3'", "C4'")
    )

    assert frame.determinant > 0.0
    assert frame.orthonormal_max_abs_error < 1.0e-12
    assert frame.perpendicular_height > 0.0
    assert frame.sin_angle > 0.0
    assert torch.equal(torch.from_numpy(frame.origin), torch.from_numpy(reordered.origin))
    assert torch.equal(torch.from_numpy(frame.basis), torch.from_numpy(reordered.basis))


def test_rna_anchor_selection_requires_one_candidate_to_win_both_worst_cases():
    def report(minimum_height, minimum_sin):
        return {
            "accepted_block_valid_percent": 100.0,
            "counts": {},
            "geometry": {
                "perpendicular_height_angstrom": {"min": minimum_height},
                "sin_angle": {"min": minimum_sin},
            },
        }

    reports = {
        "winner": report(1.1, 0.9),
        "weaker_a": report(1.0, 0.8),
        "weaker_b": report(0.9, 0.85),
    }
    assert select_numerically_dominant_candidate(reports) == "winner"

    split_winners = {
        "height_winner": report(1.1, 0.8),
        "angle_winner": report(1.0, 0.9),
    }
    assert select_numerically_dominant_candidate(split_winners) is None


def test_mirror_rna_policy_requires_a_strict_molecule_level_majority():
    assert is_mirror_rna_molecule(33, 33)
    assert is_mirror_rna_molecule(29, 29)
    assert is_mirror_rna_molecule(6, 10)
    assert not is_mirror_rna_molecule(5, 10)
    assert not is_mirror_rna_molecule(1, 76)
    assert not is_mirror_rna_molecule(0, 0)


def _rna_frame_table(dtype=torch.float64):
    return (
        ["P", "C4'", "O2'", "C2'", "C3'", "N1"],
        torch.tensor(
            [
                [-1.3, 0.7, 2.2],
                [1.4, 0.2, 0.1],
                [-0.4, 1.0, 0.8],
                [0.1, 1.2, -0.2],
                [0.0, 0.0, 0.0],
                [1.7, 1.3, -0.6],
            ],
            dtype=dtype,
        ),
    )


def test_rna_torch_frame_is_deterministic_right_handed_and_order_independent():
    codes, positions = _rna_frame_table()
    frame_a = rna_c2_c3_c4_frame(codes, positions)
    frame_b = rna_c2_c3_c4_frame(codes, positions)
    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    frame_reordered = rna_c2_c3_c4_frame(
        [codes[index] for index in permutation.tolist()], positions[permutation]
    )

    assert torch.equal(frame_a.origin, frame_b.origin)
    assert torch.equal(frame_a.basis, frame_b.basis)
    assert torch.equal(frame_a.origin, frame_reordered.origin)
    assert torch.equal(frame_a.basis, frame_reordered.basis)
    assert torch.allclose(
        frame_a.basis.transpose(0, 1) @ frame_a.basis,
        torch.eye(3, dtype=frame_a.basis.dtype),
        atol=1.0e-12,
    )
    assert math.isclose(torch.linalg.det(frame_a.basis).item(), 1.0, abs_tol=1.0e-12)

    local_points = torch.tensor(
        [[0.8, 1.1, -0.4], [-1.2, 0.3, 2.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    recovered = frame_a.global_to_local(frame_a.local_to_global(local_points))
    assert torch.allclose(recovered, local_points, atol=1.0e-12)


def test_rna_torch_frame_is_translation_and_rotation_equivariant():
    codes, positions = _rna_frame_table()
    frame = rna_c2_c3_c4_frame(codes, positions)
    rotation = _rotation_matrix()
    translation = torch.tensor([4.0, -3.5, 8.2], dtype=torch.float64)
    transformed_positions = positions @ rotation.transpose(0, 1) + translation
    transformed = rna_c2_c3_c4_frame(codes, transformed_positions)

    assert torch.allclose(
        transformed.origin,
        frame.origin @ rotation.transpose(0, 1) + translation,
        atol=1.0e-12,
    )
    assert torch.allclose(transformed.basis, rotation @ frame.basis, atol=1.0e-12)


def test_rna_torch_frame_matches_numpy_selection_audit():
    codes, positions = _rna_frame_table()
    atoms = [
        _rna_atom(code, tuple(position.tolist()))
        for code, position in zip(codes, positions)
    ]
    numpy_frame = build_candidate_frame(atoms, ("C2'", "C3'", "C4'"))
    torch_frame = rna_c2_c3_c4_frame(codes, positions)

    assert torch.allclose(
        torch_frame.origin,
        torch.from_numpy(numpy_frame.origin),
        atol=1.0e-12,
    )
    assert torch.allclose(
        torch_frame.basis,
        torch.from_numpy(numpy_frame.basis),
        atol=1.0e-12,
    )


def test_rna_torch_frame_reports_explicit_quarantine_reasons():
    cases = [
        (["C3'", "C4'"], [[0, 0, 0], [1, 0, 0]], "missing_anchor"),
        (
            ["C2'", "C2'", "C3'", "C4'"],
            [[0, 1, 0], [0, 2, 0], [0, 0, 0], [1, 0, 0]],
            "duplicate_anchor",
        ),
        (
            ["C2'", "C3'", "C4'"],
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            "coincident_anchor",
        ),
        (
            ["C2'", "C3'", "C4'"],
            [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
            "collinear_anchors",
        ),
        (
            ["C2'", "C3'", "C4'"],
            [[0, 1, 0], [0, 0, 0], [float("nan"), 0, 0]],
            "nonfinite_anchor",
        ),
        (
            ["C2'", "C3'", "C4'"],
            [[0, 1, 0], [0, 0, 0]],
            "invalid_atom_table_shape",
        ),
    ]
    for codes, positions, expected_reason in cases:
        try:
            rna_c2_c3_c4_frame(codes, positions)
        except LocalFrameError as error:
            assert error.reason == expected_reason
        else:
            raise AssertionError(f"Expected LocalFrameError: {expected_reason}")


def _universal_rna_atom(pos_code, *, is_virtual=False):
    return UniversalAtom(
        element="V" if is_virtual else "P",
        position=(0.0, 0.0, 0.0),
        pos_code=pos_code,
        block_idx=0,
        atom_idx_in_block=0,
        entity_idx=0,
        is_virtual=is_virtual,
        virtual_type="V" if is_virtual else None,
    )


def test_p_only_filter_ignores_virtual_padding_but_not_real_atoms():
    p_atom = _universal_rna_atom("P")
    virtual_atoms = [
        _universal_rna_atom(f"V{index}", is_virtual=True)
        for index in range(1, 12)
    ]
    assert is_p_only_rna_block([p_atom])
    assert is_p_only_rna_block([p_atom, *virtual_atoms])
    assert not is_p_only_rna_block([p_atom, _universal_rna_atom("OP1")])
    assert not is_p_only_rna_block([p_atom, _universal_rna_atom("sm")])


def test_p_only_filter_is_molecule_level_flagged_and_cache_signed():
    p_only_molecule = UniversalMolecule(
        id="p_only",
        dataset_type="rna",
        blocks=[UniversalBlock(symbol="A", atoms=[_universal_rna_atom("P")])],
    )
    assert rna_molecule_contains_p_only_block(p_only_molecule)

    converter = object.__new__(OptimizedUniversalRNADataset)
    converter.exclude_rna_p_only_molecules = True
    converter.molecule_max_atoms = None
    assert converter._create_pyg_data_object(p_only_molecule) is None
    assert converter._diagnose_skip(p_only_molecule) == (
        "RNA molecule contains a P-only nucleotide block"
    )

    converter.exclude_rna_p_only_molecules = False
    converter.use_hybrid_edges = False
    converter.cutoff_distance = 8.0
    converter.max_neighbors = 32
    converter.pre_filter = None
    converter.pre_transform = None
    unfiltered_data = converter._create_pyg_data_object(p_only_molecule)
    assert unfiltered_data is not None
    assert unfiltered_data.num_nodes == 1

    converter.universal_cache_path = "/tmp/universal_rna_chunk_0.pkl"
    converter.max_samples = None
    converter.exclude_rna_p_only_molecules = True
    filtered_name = converter.processed_file_names[0]
    converter.exclude_rna_p_only_molecules = False
    unfiltered_name = converter.processed_file_names[0]
    assert "_rna_no_p_only_" in filtered_name
    assert "_rna_no_p_only_" not in unfiltered_name
    assert filtered_name != unfiltered_name


def test_rna_p_only_filter_end_to_end_writes_reject_manifest():
    p_only_molecule = UniversalMolecule(
        id="reject_me",
        dataset_type="rna",
        blocks=[UniversalBlock(symbol="A", atoms=[_universal_rna_atom("P")])],
    )
    retained_atoms = [
        _universal_rna_atom("P"),
        _universal_rna_atom("OP1"),
    ]
    retained_atoms[1].position = (1.0, 0.0, 0.0)
    retained_atoms[1].atom_idx_in_block = 1
    retained_molecule = UniversalMolecule(
        id="retain_me",
        dataset_type="rna",
        blocks=[UniversalBlock(symbol="A", atoms=retained_atoms)],
    )

    with tempfile.TemporaryDirectory(prefix="mind_rna_p_only_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_pkl = tmp_path / "tiny_rna.pkl"
        with source_pkl.open("wb") as handle:
            pickle.dump(p_only_molecule, handle)
            pickle.dump(retained_molecule, handle)

        dataset = OptimizedUniversalRNADataset(
            root=str(tmp_path / "filtered"),
            universal_cache_path=str(source_pkl),
            cutoff_distance=8.0,
            max_neighbors=32,
            exclude_rna_p_only_molecules=True,
        )
        assert len(dataset) == 1
        assert dataset.get(0).mol_id == "retain_me"
        assert "_rna_no_p_only_" in Path(dataset.processed_paths[0]).name

        manifests = list(Path(dataset.processed_dir).glob("*_rejects.json"))
        assert len(manifests) == 1
        report = json.loads(manifests[0].read_text())
        assert report["input_molecules"] == 2
        assert report["rejected_molecules"] == 1
        assert report["retained_molecules"] == 1
        assert report["rejections"] == [
            {"blocks": 1, "molecule_id": "reject_me", "p_only_blocks": 1}
        ]
        assert report["source_pkl_rewritten"] is False
