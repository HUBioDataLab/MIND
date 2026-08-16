import numpy as np

from data_loading.audit_virtual_atom_leakage import (
    GATE_FEATURES,
    build_feature_matrices,
    stable_group_split,
)


def _protein_backbone_batch():
    first = np.array(
        [
            [-0.55, 1.36, 0.0],
            [0.0, 0.0, 0.0],
            [1.53, 0.0, 0.0],
            [2.15, -0.7, 0.2],
        ],
        dtype=np.float32,
    )
    second = first.copy()
    second[0] = [-0.58, 1.31, 0.0]
    second[3] = [2.10, 0.8, -0.4]
    return np.stack((first, second))


def _rna_backbone_batch():
    first = np.array(
        [
            [2.9, -1.6, -2.4],
            [3.1, -2.3, -2.2],
            [2.7, -1.6, -3.1],
            [2.2, -1.0, -1.6],
            [2.2, -0.9, -0.8],
            [1.52, 0.0, 0.0],
            [1.89, 1.19, -0.4],
            [0.0, 0.0, 0.0],
            [-0.52, -0.67, 1.14],
            [-0.33, 1.49, 0.0],
            [-0.67, 1.96, 1.06],
            [0.80, 2.05, -0.31],
        ],
        dtype=np.float32,
    )
    second = first.copy()
    second[0:3] += np.array([0.4, -0.2, 0.3], dtype=np.float32)
    second[9] = [-0.36, 1.44, 0.0]
    return np.stack((first, second))


def test_stable_group_split_never_splits_one_molecule():
    for domain in ("protein", "rna"):
        values = [stable_group_split(domain, "same_molecule") for _ in range(20)]
        assert len(set(values)) == 1
        assert values[0] in {"train", "validation"}


def test_invalid_validation_fraction_is_rejected():
    for fraction in (0.0, 1.0, -0.1, 1.1):
        try:
            stable_group_split("protein", "molecule", validation_fraction=fraction)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid validation fraction accepted: {fraction}")


def test_protein_fixed_template_features_ignore_backbone_variation():
    features = build_feature_matrices("protein", _protein_backbone_batch())
    for name in ("template_xyz", "template_slot_order", "template_internal_graph"):
        assert np.array_equal(features[name][0], features[name][1])
    assert not np.array_equal(
        features["visible_backbone_only"][0],
        features["visible_backbone_only"][1],
    )
    assert not np.array_equal(
        features["frame_anchors_plus_template_relative"][0],
        features["frame_anchors_plus_template_relative"][1],
    )


def test_rna_fixed_template_features_ignore_backbone_variation():
    features = build_feature_matrices("rna", _rna_backbone_batch())
    for name in ("template_xyz", "template_slot_order", "template_internal_graph"):
        assert np.array_equal(features[name][0], features[name][1])
    assert not np.array_equal(
        features["visible_backbone_only"][0],
        features["visible_backbone_only"][1],
    )
    assert not np.array_equal(
        features["frame_anchors_plus_template_relative"][0],
        features["frame_anchors_plus_template_relative"][1],
    )


def test_gate_feature_names_are_explicit_and_complete():
    assert GATE_FEATURES == (
        "template_xyz",
        "template_slot_order",
        "template_internal_graph",
    )
