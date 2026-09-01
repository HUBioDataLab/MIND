import copy
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from core.pretraining_config import (
    MASKED_SLOT_TEMPLATE_DEFAULTS,
    create_pretraining_config,
    normalize_masked_slot_template_config,
)
from core.block_masking import legacy_oxt_eligible_node_mask, select_whole_blocks
from core.masked_slot_mapping import map_masked_template_slots
from core.molecular_encoder import build_domain_local_slot_ids
from core.masked_coordinate_corruption import (
    apply_masked_slot_template_,
    build_masked_coordinate_corruption,
    rebuild_template_incident_edges,
)
from core.pretraining_model import PretrainingESAModel
from core.pretraining_tasks import PretrainingTasks
from core.virtual_atom_templates import (
    protein_sidechain_template,
    rna_base_template,
)
from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    RNA_BACKBONE_ATOMS,
    VIRTUAL_ATOMIC_NUMBER,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _MaskingHarness:
    def __init__(self, config):
        self.config = config
        self.current_epoch = 4

    def _graph_dtype_list(self, batch):
        return PretrainingESAModel._graph_dtype_list(batch)


def _legacy_config_without_template_field():
    return SimpleNamespace(
        pretraining_tasks=["long_range_distance", "short_range_distance", "mlm"],
        protein_block_mlm=True,
        protein_block_mlm_ratio=0.5,
        mlm_mask_ratio=0.15,
    )


def _explicit_flag_off_config():
    return create_pretraining_config(
        pretraining_tasks=["long_range_distance", "short_range_distance", "mlm"],
        protein_block_mlm=True,
        protein_block_mlm_ratio=0.5,
        mlm_mask_ratio=0.15,
        masked_slot_template={
            "enabled": False,
            "protein": False,
            "rna": False,
        },
    )


def _mixed_protein_rna_batch():
    protein_nodes = 28  # two exact 14-slot residues
    rna_nodes = 23      # one exact 23-slot nucleotide
    num_nodes = protein_nodes + rna_nodes
    z = (torch.arange(num_nodes, dtype=torch.long) % 10) + 1
    pos = torch.arange(num_nodes * 3, dtype=torch.float32).reshape(num_nodes, 3)
    pos = pos / 7.0
    source = torch.arange(num_nodes, dtype=torch.long)
    target = torch.roll(source, shifts=-1)
    edge_index = torch.stack(
        (torch.cat((source, target)), torch.cat((target, source))), dim=0
    )
    edge_attr = torch.linspace(
        0.1, 1.0, edge_index.shape[1], dtype=torch.float32
    ).unsqueeze(1)
    return SimpleNamespace(
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        block_idx=torch.tensor([0] * 14 + [1] * 14 + [0] * 23),
        batch=torch.tensor([0] * protein_nodes + [1] * rna_nodes),
        dataset_type=["pdb", "rna"],
        num_graphs=2,
        num_nodes=num_nodes,
    )


def _mixed_protein_rna_unimol_block_batch():
    protein_nodes = 28   # two 14-slot residues
    rna_nodes = 69       # three 23-slot nucleotides
    unimol_nodes = 20    # remains atom-level
    num_nodes = protein_nodes + rna_nodes + unimol_nodes
    source = torch.arange(num_nodes, dtype=torch.long)
    target = torch.roll(source, shifts=-1)
    edge_index = torch.stack((source, target), dim=0)
    return SimpleNamespace(
        z=(torch.arange(num_nodes, dtype=torch.long) % 10) + 1,
        pos=torch.arange(num_nodes * 3, dtype=torch.float32).reshape(
            num_nodes, 3
        ),
        edge_index=edge_index,
        edge_attr=torch.ones((num_nodes, 1), dtype=torch.float32),
        block_idx=torch.tensor(
            [0] * 14
            + [1] * 14
            + [0] * 23
            + [1] * 23
            + [2] * 23
            + [0] * unimol_nodes
        ),
        batch=torch.tensor(
            [0] * protein_nodes + [1] * rna_nodes + [2] * unimol_nodes
        ),
        dataset_type=["pdb", "rna", "unimol"],
        num_graphs=3,
        num_nodes=num_nodes,
    )


def _run_masking(config, *, is_validation, input_batch=None):
    batch = copy.deepcopy(
        _mixed_protein_rna_batch() if input_batch is None else input_batch
    )
    before = copy.deepcopy(batch)
    harness = _MaskingHarness(config)
    torch.manual_seed(91_337)
    PretrainingESAModel._ensure_mlm_attributes(
        harness,
        batch,
        batch_idx=7,
        is_validation=is_validation,
    )
    rng_state_after_masking = torch.random.get_rng_state().clone()
    short_range_loss_mask = (
        torch.rand(batch.edge_index.shape[1]) < 0.15
    )
    return batch, before, rng_state_after_masking, short_range_loss_mask


def _losses_for_identical_model_inputs(batch, short_range_loss_mask):
    config = create_pretraining_config(graph_dim=16)
    torch.manual_seed(41_009)
    tasks = PretrainingTasks(config)
    node_embeddings = torch.randn(batch.num_nodes, config.graph_dim)

    losses = {
        "mlm": tasks.mlm_loss(node_embeddings, batch),
        "short_range_distance": tasks.short_range_distance_loss(
            node_embeddings,
            batch.edge_index,
            batch.edge_attr.squeeze(-1),
            short_range_loss_mask,
        ),
    }
    # The current long-range sampler is stochastic for batches above 50 nodes.
    # A fixed seed lets this test detect changes to either its inputs or RNG use.
    torch.manual_seed(62_011)
    losses["long_range_distance"] = tasks.long_range_distance_loss(
        node_embeddings, batch
    )
    return losses, torch.random.get_rng_state().clone()


def _assert_flag_off_matches_legacy(*, is_validation, input_batch=None):
    legacy = _run_masking(
        _legacy_config_without_template_field(),
        is_validation=is_validation,
        input_batch=input_batch,
    )
    explicit = _run_masking(
        _explicit_flag_off_config(),
        is_validation=is_validation,
        input_batch=input_batch,
    )
    legacy_batch, legacy_before, legacy_rng, legacy_distance_mask = legacy
    explicit_batch, explicit_before, explicit_rng, explicit_distance_mask = explicit

    for field in (
        "z",
        "original_types",
        "masked_types",
        "mlm_mask",
        "pos",
        "edge_index",
        "edge_attr",
        "block_idx",
        "batch",
    ):
        assert torch.equal(getattr(legacy_batch, field), getattr(explicit_batch, field))

    # Coordinate/graph tensors must remain exactly as loaded when the flag is off.
    for field in ("pos", "edge_index", "edge_attr", "block_idx", "batch"):
        assert torch.equal(getattr(legacy_batch, field), getattr(legacy_before, field))
        assert torch.equal(getattr(explicit_batch, field), getattr(explicit_before, field))

    # Identical RNG consumption also guarantees the subsequently sampled
    # short-range loss edge mask remains bitwise identical.
    assert torch.equal(legacy_rng, explicit_rng)
    assert torch.equal(legacy_distance_mask, explicit_distance_mask)

    legacy_losses, legacy_loss_rng = _losses_for_identical_model_inputs(
        legacy_batch, legacy_distance_mask
    )
    explicit_losses, explicit_loss_rng = _losses_for_identical_model_inputs(
        explicit_batch, explicit_distance_mask
    )
    for task_name in legacy_losses:
        assert torch.equal(legacy_losses[task_name], explicit_losses[task_name]), (
            task_name,
            legacy_losses[task_name],
            explicit_losses[task_name],
        )
    assert torch.equal(legacy_loss_rng, explicit_loss_rng)

    if input_batch is None:
        # One 14-node protein residue plus floor(23 * 0.15)=3 RNA atoms.
        assert int(explicit_batch.mlm_mask.sum()) == 17
        protein_mask = explicit_batch.mlm_mask[:28].reshape(2, 14)
        assert sorted(protein_mask.sum(dim=1).tolist()) == [0, 14]
        assert int(explicit_batch.mlm_mask[28:].sum()) == 3

    for forbidden in (
        "clean_pos",
        "input_pos",
        "template_node_mask",
        "template_edge_mask",
        "distance_loss_mask",
    ):
        assert not hasattr(explicit_batch, forbidden)


def test_absent_and_explicit_false_config_normalize_identically():
    absent = create_pretraining_config().masked_slot_template
    explicit = _explicit_flag_off_config().masked_slot_template
    assert absent == MASKED_SLOT_TEMPLATE_DEFAULTS
    assert explicit == MASKED_SLOT_TEMPLATE_DEFAULTS
    assert absent is not explicit


def test_fast_and_multidomain_yaml_select_one_virtual_slot_policy():
    expected_prepositioned = {
        "enabled": True,
        "protein": True,
        "rna": True,
        "visible_virtuals_prepositioned": True,
    }
    selected_policies = []
    for relative_path in (
        "core/pretraining_config_fast.yaml",
        "core/pretraining_config_multidomain.yaml",
    ):
        config = yaml.safe_load((REPO_ROOT / relative_path).read_text())
        assert config["protein_block_mlm"] is True
        assert config["rna_block_mlm"] is True
        assert config["rna_block_mlm_ratio"] == 0.15
        assert config["use_domain_local_slot_embedding"] is True
        assert config["virtual_atom_loss_weight"] == 0.2
        assert config["protein_block_mlm_exclude_legacy_oxt"] is True
        parsed = create_pretraining_config(**config)
        neutral = getattr(parsed, "geometry_neutral_virtual_slots", None) or {}
        prepositioned_enabled = (
            parsed.masked_slot_template == expected_prepositioned
        )
        neutral_enabled = (
            neutral.get("enabled") is True
            and neutral.get("protein") is True
            and neutral.get("rna") is True
            and parsed.masked_slot_template == MASKED_SLOT_TEMPLATE_DEFAULTS
        )
        assert prepositioned_enabled != neutral_enabled
        selected_policies.append(
            "prepositioned" if prepositioned_enabled else "geometry_neutral"
        )
        assert parsed.rna_block_mlm is True

    assert len(set(selected_policies)) == 1


def test_flag_on_requires_matching_domain_block_mlm():
    try:
        create_pretraining_config(
            masked_slot_template={
                "enabled": True,
                "protein": True,
                "rna": False,
            }
        )
    except ValueError as error:
        assert "protein_block_mlm" in str(error)
    else:
        raise AssertionError("protein template accepted atom-level protein MLM")

    config = create_pretraining_config(
        pretraining_tasks=["mlm"],
        protein_block_mlm=True,
        rna_block_mlm=True,
        masked_slot_template={
            "enabled": True,
            "protein": True,
            "rna": True,
        },
    )
    assert config.masked_slot_template == {
        "enabled": True,
        "protein": True,
        "rna": True,
        "visible_virtuals_prepositioned": False,
    }


def test_invalid_masked_slot_template_contract_is_rejected():
    invalid_values = (
        True,
        {"enabled": False, "protein": True, "rna": False},
        {"enabled": False, "unknown": False},
        {"enabled": 0, "protein": False, "rna": False},
    )
    for value in invalid_values:
        try:
            normalize_masked_slot_template_config(value)
        except (TypeError, ValueError, NotImplementedError):
            pass
        else:
            raise AssertionError(f"invalid config accepted: {value!r}")


def test_flag_off_training_masking_and_graph_inputs_are_bitwise_legacy():
    _assert_flag_off_matches_legacy(is_validation=False)


def test_flag_off_validation_masking_and_graph_inputs_are_bitwise_legacy():
    _assert_flag_off_matches_legacy(is_validation=True)


def test_shared_block_selector_returns_one_consistent_selection_object():
    block_ids = torch.tensor([7] * 14 + [11] * 14 + [19] * 14)
    generator_a = torch.Generator(device="cpu")
    generator_b = torch.Generator(device="cpu")
    generator_a.manual_seed(8103)
    generator_b.manual_seed(8103)
    selection_a = select_whole_blocks(block_ids, 2 / 3, generator=generator_a)
    selection_b = select_whole_blocks(block_ids, 2 / 3, generator=generator_b)

    assert torch.equal(selection_a.selected_block_ids, selection_b.selected_block_ids)
    assert torch.equal(selection_a.selected_node_mask, selection_b.selected_node_mask)
    assert selection_a.selected_block_ids.numel() == 2
    assert int(selection_a.selected_node_mask.sum()) == 28
    for block_id in torch.unique(block_ids):
        local_mask = selection_a.selected_node_mask[block_ids == block_id]
        assert bool(local_mask.all()) or bool((~local_mask).all())
    assert torch.equal(
        torch.unique(block_ids[selection_a.selected_node_mask]),
        torch.sort(selection_a.selected_block_ids).values,
    )


def test_template_on_protein_block_selection_excludes_legacy_oxt_block():
    block_ids = torch.tensor([0] * 14 + [1] * 14 + [2] * 14)
    pos_codes = (
        list(PROTEIN_BACKBONE_ATOMS) + ["V1"] + [f"V{i}" for i in range(2, 11)]
        + list(PROTEIN_BACKBONE_ATOMS) + ["CB", "OXT"] + [f"V{i}" for i in range(3, 11)]
        + list(PROTEIN_BACKBONE_ATOMS) + ["CB"] + [f"V{i}" for i in range(2, 11)]
    )
    eligible = legacy_oxt_eligible_node_mask(block_ids, pos_codes)
    assert bool(eligible[:14].all())
    assert not bool(eligible[14:28].any())
    assert bool(eligible[28:].all())

    generator = torch.Generator(device="cpu")
    generator.manual_seed(19_337)
    selection = select_whole_blocks(
        block_ids,
        1.0,
        generator=generator,
        eligible_node_mask=eligible,
    )
    assert torch.equal(selection.selected_block_ids.sort().values, torch.tensor([0, 2]))
    assert int(selection.selected_node_mask.sum()) == 28
    assert not bool(selection.selected_node_mask[14:28].any())

    batch = SimpleNamespace(
        z=torch.full((42,), 6, dtype=torch.long),
        pos=torch.zeros((42, 3), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
        block_idx=block_ids,
        batch=torch.zeros(42, dtype=torch.long),
        dataset_type=["pdb"],
        pos_code=pos_codes,
        num_graphs=1,
        num_nodes=42,
    )
    config = create_pretraining_config(
        pretraining_tasks=["mlm"],
        protein_block_mlm=True,
        protein_block_mlm_ratio=1.0,
        masked_slot_template={
            "enabled": True,
            "protein": True,
            "rna": False,
        },
    )
    PretrainingESAModel._ensure_mlm_attributes(
        _MaskingHarness(config), batch, batch_idx=3, is_validation=True
    )
    assert bool(batch.mlm_mask[:14].all())
    assert not bool(batch.mlm_mask[14:28].any())
    assert bool(batch.mlm_mask[28:].all())

    # A paired control can use the established OXT exclusion independently of
    # templates, so both branches receive exactly the same validation targets.
    control_batch = SimpleNamespace(
        z=torch.full((42,), 6, dtype=torch.long),
        pos=torch.zeros((42, 3), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
        block_idx=block_ids,
        batch=torch.zeros(42, dtype=torch.long),
        dataset_type=["pdb"],
        pos_code=pos_codes,
        num_graphs=1,
        num_nodes=42,
    )
    control_config = create_pretraining_config(
        pretraining_tasks=["mlm"],
        protein_block_mlm=True,
        protein_block_mlm_ratio=1.0,
        protein_block_mlm_exclude_legacy_oxt=True,
    )
    PretrainingESAModel._ensure_mlm_attributes(
        _MaskingHarness(control_config),
        control_batch,
        batch_idx=3,
        is_validation=True,
    )
    assert torch.equal(control_batch.mlm_mask, batch.mlm_mask)


def test_domain_local_slot_ids_do_not_depend_on_masked_atom_types():
    batch = _mixed_protein_rna_unimol_block_batch()
    ids_before = build_domain_local_slot_ids(batch, batch.num_nodes, batch.z.device)
    assert torch.equal(ids_before[:28], torch.tensor(list(range(1, 15)) * 2))
    expected_rna = torch.tensor(list(range(15, 38)) * 3)
    assert torch.equal(ids_before[28:97], expected_rna)
    assert not bool(ids_before[97:].any())

    batch.z.zero_()
    ids_after = build_domain_local_slot_ids(batch, batch.num_nodes, batch.z.device)
    assert torch.equal(ids_after, ids_before)

    malformed = copy.deepcopy(_mixed_protein_rna_batch())
    malformed.block_idx[13] = 1  # first protein run becomes 13 nodes
    malformed.block_idx[14] = 1  # second run becomes 15 nodes
    malformed_ids = build_domain_local_slot_ids(
        malformed, malformed.num_nodes, malformed.z.device
    )
    assert not bool(malformed_ids[:28].any())
    assert torch.equal(malformed_ids[28:], torch.arange(15, 38))


def test_virtual_weighted_mlm_matches_other_group_formula_and_legacy_default():
    logits = torch.randn(4, 121, generator=torch.Generator().manual_seed(2_901))
    targets = torch.tensor([6, VIRTUAL_ATOMIC_NUMBER, 8, VIRTUAL_ATOMIC_NUMBER])
    data = SimpleNamespace(
        mlm_mask=torch.ones(4, dtype=torch.bool),
        original_types=targets,
        masked_types=torch.zeros(4, dtype=torch.long),
    )

    legacy_tasks = PretrainingTasks(
        create_pretraining_config(graph_dim=121, virtual_atom_loss_weight=1.0)
    )
    legacy_tasks.mlm_head = torch.nn.Identity()
    assert torch.equal(
        legacy_tasks.mlm_loss(logits, data),
        torch.nn.functional.cross_entropy(logits, targets),
    )

    weighted_tasks = PretrainingTasks(
        create_pretraining_config(graph_dim=121, virtual_atom_loss_weight=0.2)
    )
    weighted_tasks.mlm_head = torch.nn.Identity()
    real = targets != VIRTUAL_ATOMIC_NUMBER
    expected = (
        torch.nn.functional.cross_entropy(logits[real], targets[real])
        + 0.2
        * torch.nn.functional.cross_entropy(logits[~real], targets[~real])
    )
    assert torch.allclose(weighted_tasks.mlm_loss(logits, data), expected)


def test_standard_mlm_loss_with_virtual_atom_weighting():
    logits = torch.randn(6, 121, generator=torch.Generator().manual_seed(8_117))
    targets = torch.tensor([6, 6, 8, 16, 7, VIRTUAL_ATOMIC_NUMBER])
    # Protein ids 5..14 are sidechain slots; id 1 is backbone N.
    slot_ids = torch.tensor([5, 6, 7, 8, 1, 9])
    data = SimpleNamespace(
        mlm_mask=torch.ones(6, dtype=torch.bool),
        original_types=targets,
        masked_types=torch.zeros(6, dtype=torch.long),
        domain_local_slot_ids=slot_ids,
    )
    tasks = PretrainingTasks(
        create_pretraining_config(
            graph_dim=121,
            hidden_dims=[121],
            protein_block_mlm=True,
            use_domain_local_slot_embedding=True,
            virtual_atom_loss_weight=0.2,
            geometry_neutral_virtual_slots={
                "enabled": True,
                "protein": True,
                "rna": False,
            },
        )
    )
    tasks.mlm_head = torch.nn.Identity()

    per_node = torch.nn.functional.cross_entropy(
        logits, targets, reduction="none"
    )
    expected_real = per_node[:5].mean()
    expected = expected_real + 0.2 * per_node[5]

    assert torch.allclose(tasks.mlm_loss(logits, data), expected)


def _run_rna_block_masking(*, is_validation, current_epoch=4):
    config = SimpleNamespace(
        pretraining_tasks=["mlm"],
        protein_block_mlm=True,
        protein_block_mlm_ratio=0.5,
        rna_block_mlm=True,
        rna_block_mlm_ratio=2 / 3,
        mlm_mask_ratio=0.15,
    )
    harness = _MaskingHarness(config)
    harness.current_epoch = current_epoch
    batch = _mixed_protein_rna_unimol_block_batch()
    before = copy.deepcopy(batch)
    torch.manual_seed(72_019)
    PretrainingESAModel._ensure_mlm_attributes(
        harness,
        batch,
        batch_idx=9,
        is_validation=is_validation,
    )
    return batch, before


def _assert_rna_block_contract(*, is_validation):
    first, before = _run_rna_block_masking(is_validation=is_validation)
    second, _ = _run_rna_block_masking(is_validation=is_validation)
    assert torch.equal(first.mlm_mask, second.mlm_mask)

    protein_mask = first.mlm_mask[:28].reshape(2, 14)
    rna_mask = first.mlm_mask[28:97].reshape(3, 23)
    unimol_mask = first.mlm_mask[97:]
    assert sorted(protein_mask.sum(dim=1).tolist()) == [0, 14]
    assert sorted(rna_mask.sum(dim=1).tolist()) == [0, 23, 23]
    assert int(unimol_mask.sum()) == 3

    # The MLM node mask is the sole source of truth future templating will
    # consume: selected RNA blocks are complete, never partially selected.
    selected_rna_blocks = torch.where(rna_mask.all(dim=1))[0]
    assert selected_rna_blocks.numel() == 2
    assert not rna_mask.any(dim=1)[~rna_mask.all(dim=1)].any()

    assert torch.equal(first.original_types, before.z)
    assert torch.equal(first.z[first.mlm_mask], torch.zeros_like(first.z[first.mlm_mask]))
    assert torch.equal(first.z[~first.mlm_mask], before.z[~first.mlm_mask])
    for field in ("pos", "edge_index", "edge_attr", "block_idx", "batch"):
        assert torch.equal(getattr(first, field), getattr(before, field))
    for forbidden in ("clean_pos", "input_pos", "template_node_mask"):
        assert not hasattr(first, forbidden)


def test_rna_block_mlm_training_masks_whole_nucleotides_reproducibly():
    _assert_rna_block_contract(is_validation=False)


def test_rna_block_mlm_validation_masks_whole_nucleotides_deterministically():
    _assert_rna_block_contract(is_validation=True)


def test_validation_mask_is_stable_across_epochs():
    first, _ = _run_rna_block_masking(
        is_validation=True, current_epoch=0
    )
    later, _ = _run_rna_block_masking(
        is_validation=True, current_epoch=37
    )
    assert torch.equal(first.mlm_mask, later.mlm_mask)


def test_rna_block_mode_defers_sample_masking_to_batch_time():
    from core.train_pretrain import create_pretraining_data_transforms

    atom_config = create_pretraining_config(
        pretraining_tasks=["mlm"],
        protein_block_mlm=False,
        rna_block_mlm=False,
    )
    rna_block_config = create_pretraining_config(
        pretraining_tasks=["mlm"],
        protein_block_mlm=False,
        rna_block_mlm=True,
    )
    assert create_pretraining_data_transforms(atom_config) is not None
    assert create_pretraining_data_transforms(rna_block_config) is None


def _canonical_slot_mapping_batch():
    protein_blocks = (
        list(PROTEIN_BACKBONE_ATOMS)
        + ["CB"]
        + [f"V{index}" for index in range(2, 11)],
        list(PROTEIN_BACKBONE_ATOMS)
        + [f"V{index}" for index in range(1, 11)],
    )
    # Historical RNA caches use this valid but pre-current-adapter order.
    rna_g_slots = [
        "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"
    ]
    rna_block = list(RNA_BACKBONE_ATOMS) + rna_g_slots
    graph_pos_codes = [protein_blocks[0] + protein_blocks[1], rna_block]
    num_nodes = sum(len(codes) for codes in graph_pos_codes)
    return SimpleNamespace(
        z=torch.full((num_nodes,), 6, dtype=torch.long),
        pos=torch.arange(num_nodes * 3, dtype=torch.float32).reshape(num_nodes, 3),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
        block_idx=torch.tensor([0] * 14 + [1] * 14 + [0] * 23),
        batch=torch.tensor([0] * 28 + [1] * 23),
        dataset_type=["pdb", "rna"],
        pos_code=graph_pos_codes,
        num_graphs=2,
        num_nodes=num_nodes,
    )


def test_runtime_slot_mapping_finds_exact_selected_protein_and_rna_slots():
    batch = _canonical_slot_mapping_batch()
    config = SimpleNamespace(
        pretraining_tasks=["mlm"],
        protein_block_mlm=True,
        protein_block_mlm_ratio=0.5,
        rna_block_mlm=True,
        rna_block_mlm_ratio=1.0,
        mlm_mask_ratio=0.15,
    )
    torch.manual_seed(4_033)
    PretrainingESAModel._ensure_mlm_attributes(
        _MaskingHarness(config), batch, batch_idx=2, is_validation=True
    )
    before_mapping = copy.deepcopy(batch)
    mapping = map_masked_template_slots(batch)

    assert len(mapping.mapped_blocks) == 2
    assert not mapping.quarantined_blocks
    by_domain = {block.domain: block for block in mapping.mapped_blocks}
    assert by_domain["protein"].node_indices.numel() == 10
    assert by_domain["rna"].node_indices.numel() == 11
    assert len(by_domain["protein"].pos_codes) == 10
    assert len(by_domain["rna"].pos_codes) == 11
    assert by_domain["rna"].pos_codes == (
        "N9", "C8", "N7", "C6", "N1", "C2", "N3", "C4", "C5", "O6", "N2"
    )
    assert int(mapping.node_mask.sum()) == 21
    assert not bool((mapping.node_mask & ~batch.mlm_mask).any())

    # Whole-block MLM includes backbone, but template mapping deliberately
    # retains only the 10/11 side/base nodes.
    assert int(batch.mlm_mask.sum()) == 14 + 23
    assert int(batch.mlm_mask.sum() - mapping.node_mask.sum()) == 4 + 12
    for field in (
        "z", "pos", "edge_index", "edge_attr", "block_idx", "batch", "mlm_mask"
    ):
        assert torch.equal(getattr(batch, field), getattr(before_mapping, field))
    assert batch.pos_code == before_mapping.pos_code


def test_runtime_slot_mapping_quarantines_partial_and_malformed_blocks():
    bad_protein = (
        list(PROTEIN_BACKBONE_ATOMS)
        + ["CB", "V1"]
        + [f"V{index}" for index in range(3, 11)]
    )
    valid_rna = list(RNA_BACKBONE_ATOMS) + [
        "N9", "C8", "N7", "C6", "N1", "C2", "N3", "C4", "C5", "O6", "N2"
    ]
    pos_codes = bad_protein + valid_rna
    num_nodes = len(pos_codes)
    batch = SimpleNamespace(
        z=torch.full((num_nodes,), 6, dtype=torch.long),
        batch=torch.tensor([0] * len(bad_protein) + [1] * len(valid_rna)),
        block_idx=torch.zeros(num_nodes, dtype=torch.long),
        mlm_mask=torch.tensor(
            [True] * len(bad_protein) + [True] + [False] * (len(valid_rna) - 1)
        ),
        pos_code=pos_codes,
        dataset_type=["pdb", "rna"],
        num_graphs=2,
    )
    mapping = map_masked_template_slots(batch)
    assert not mapping.mapped_blocks
    assert not bool(mapping.node_mask.any())
    reasons = {
        block.domain: set(block.reasons)
        for block in mapping.quarantined_blocks
    }
    assert reasons["protein"] == {"virtual_slot_index_mismatch"}
    assert reasons["rna"] == {"partially_masked_block"}


def test_runtime_slot_mapping_accepts_modified_rna_real_atom_names():
    rna_codes = list(RNA_BACKBONE_ATOMS) + [
        "C6", "N1", "C2", "N3", "C4", "C5", "N4", "O2", "CM5",
        "V10", "V11",
    ]
    batch = SimpleNamespace(
        z=torch.full((len(rna_codes),), 6, dtype=torch.long),
        batch=torch.zeros(len(rna_codes), dtype=torch.long),
        block_idx=torch.zeros(len(rna_codes), dtype=torch.long),
        mlm_mask=torch.ones(len(rna_codes), dtype=torch.bool),
        pos_code=rna_codes,
        dataset_type="rna",
        num_graphs=1,
    )

    mapping = map_masked_template_slots(batch)

    assert len(mapping.mapped_blocks) == 1
    assert not mapping.quarantined_blocks
    assert mapping.mapped_blocks[0].pos_codes[-3:] == ("CM5", "V10", "V11")


def test_runtime_slot_mapping_rejects_missing_backbone_and_bad_metadata():
    rna_codes = [
        code for code in RNA_BACKBONE_ATOMS if code != "O2'"
    ] + [
        "N9", "C8", "N7", "C6", "N1", "C2", "N3", "C4", "C5", "O6", "N2"
    ]
    batch = SimpleNamespace(
        z=torch.full((len(rna_codes),), 6, dtype=torch.long),
        batch=torch.zeros(len(rna_codes), dtype=torch.long),
        block_idx=torch.zeros(len(rna_codes), dtype=torch.long),
        mlm_mask=torch.ones(len(rna_codes), dtype=torch.bool),
        pos_code=rna_codes,
        dataset_type="rna",
        num_graphs=1,
    )
    mapping = map_masked_template_slots(batch)
    assert not mapping.mapped_blocks
    assert "missing_backbone_position" in mapping.quarantined_blocks[0].reasons

    batch.pos_code = rna_codes[:-1]
    try:
        map_masked_template_slots(batch)
    except ValueError as error:
        assert "pos_code length" in str(error)
    else:
        raise AssertionError("slot mapping accepted misaligned pos_code metadata")


def _coordinate_corruption_batch(*, select_both_protein_blocks=False):
    batch = _canonical_slot_mapping_batch()
    pos = torch.zeros_like(batch.pos)

    def set_protein_frame(start, offset):
        pos[start + 0] = offset + torch.tensor([0.0, 1.0, 0.0])  # N
        pos[start + 1] = offset + torch.tensor([0.0, 0.0, 0.0])  # CA
        pos[start + 2] = offset + torch.tensor([1.0, 0.0, 0.0])  # C
        pos[start + 3] = offset + torch.tensor([1.0, 1.0, 0.0])  # O
        for local_index in range(4, 14):
            pos[start + local_index] = offset + torch.tensor(
                [0.1 * local_index, 0.2, 0.3]
            )

    set_protein_frame(0, torch.tensor([0.0, 0.0, 0.0]))
    set_protein_frame(14, torch.tensor([10.0, 0.0, 0.0]))

    rna_start = 28
    rna_offset = torch.tensor([20.0, 0.0, 0.0])
    rna_codes = batch.pos_code[1]
    for local_index in range(23):
        pos[rna_start + local_index] = rna_offset + torch.tensor(
            [0.05 * local_index, 0.1, 0.2]
        )
    rna_index = {code: index for index, code in enumerate(rna_codes)}
    pos[rna_start + rna_index["C3'"]] = rna_offset
    pos[rna_start + rna_index["C4'"]] = rna_offset + torch.tensor([1.0, 0.0, 0.0])
    pos[rna_start + rna_index["C2'"]] = rna_offset + torch.tensor([0.0, 1.0, 0.0])
    pos[rna_start + rna_index["O2'"]] = rna_offset + torch.tensor([0.0, 0.0, 1.0])

    batch.pos = pos
    flat_codes = [code for graph_codes in batch.pos_code for code in graph_codes]
    for node_index, code in enumerate(flat_codes):
        if code.startswith("V"):
            batch.z[node_index] = VIRTUAL_ATOMIC_NUMBER
    batch.original_types = batch.z.clone()
    batch.mlm_mask = torch.tensor(
        [True] * 14
        + [select_both_protein_blocks] * 14
        + [True] * 23
    )
    return batch


def _attach_graph_local_chain_edges(batch):
    edge_parts = []
    for graph_id in range(batch.num_graphs):
        nodes = torch.where(batch.batch == graph_id)[0]
        forward = torch.stack((nodes[:-1], nodes[1:]), dim=0)
        edge_parts.extend((forward, forward.flip(0)))
    batch.edge_index = torch.cat(edge_parts, dim=1)
    batch.edge_attr = torch.linalg.vector_norm(
        batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]],
        dim=1,
    ).unsqueeze(1)
    return batch


def test_coordinate_corruption_templates_targets_and_relocates_only_visible_virtuals():
    batch = _coordinate_corruption_batch()
    before = copy.deepcopy(batch)
    mapping = map_masked_template_slots(batch)
    result = build_masked_coordinate_corruption(batch, mapping=mapping)

    assert len(result.applied_blocks) == 2
    assert not result.mapping_quarantines
    assert not result.coordinate_quarantines
    assert int(result.target_template_node_mask.sum()) == 10 + 11
    assert int(result.context_virtual_node_mask.sum()) == 10
    assert int(result.template_node_mask.sum()) == 10 + 11 + 10
    assert torch.equal(result.clean_pos, before.pos)
    assert torch.equal(result.input_pos[~result.template_node_mask], before.pos[~result.template_node_mask])
    assert bool((result.input_pos[result.template_node_mask] != before.pos[result.template_node_mask]).any(dim=1).all())

    protein_block = next(block for block in result.applied_blocks if block.domain == "protein")
    rna_block = next(block for block in result.applied_blocks if block.domain == "rna")
    assert torch.equal(
        result.input_pos[protein_block.node_indices],
        protein_sidechain_template(dtype=batch.pos.dtype),
    )
    assert torch.equal(
        result.input_pos[rna_block.node_indices],
        rna_base_template(dtype=batch.pos.dtype) + torch.tensor([20.0, 0.0, 0.0]),
    )
    context_block = result.context_virtual_blocks[0]
    assert context_block.domain == "protein"
    assert context_block.block_id == 1
    assert torch.equal(
        result.input_pos[context_block.node_indices],
        protein_sidechain_template(dtype=batch.pos.dtype)
        + torch.tensor([10.0, 0.0, 0.0]),
    )
    assert torch.equal(result.input_pos[14:18], before.pos[14:18])
    assert torch.allclose(
        torch.cdist(result.input_pos[protein_block.node_indices], result.input_pos[protein_block.node_indices]),
        torch.cdist(protein_sidechain_template(), protein_sidechain_template()),
        atol=1e-6,
        rtol=0.0,
    )

    # The function is pure: neither coordinates nor MLM targets are attached
    # to or modified on the input batch.
    assert torch.equal(batch.pos, before.pos)
    assert torch.equal(batch.z, before.z)
    assert torch.equal(batch.original_types, before.original_types)
    assert torch.equal(batch.mlm_mask, before.mlm_mask)
    assert not hasattr(batch, "clean_pos")
    assert not hasattr(batch, "input_pos")
    assert not hasattr(batch, "template_node_mask")


def test_coordinate_corruption_is_type_independent_for_ala_and_gly_slots():
    batch = _coordinate_corruption_batch(select_both_protein_blocks=True)
    result = build_masked_coordinate_corruption(batch)
    protein_blocks = [
        block for block in result.applied_blocks if block.domain == "protein"
    ]
    assert len(protein_blocks) == 2
    by_block = {block.block_id: block for block in protein_blocks}
    ala_points = result.input_pos[by_block[0].node_indices]
    gly_points_local = (
        result.input_pos[by_block[1].node_indices]
        - torch.tensor([10.0, 0.0, 0.0])
    )
    assert torch.allclose(ala_points, gly_points_local, atol=1e-6, rtol=0.0)


def test_visible_ala_keeps_real_cb_and_maps_v2_to_v10_by_absolute_slot():
    batch = _coordinate_corruption_batch()
    batch.mlm_mask[:14] = False
    batch.mlm_mask[14:28] = True
    before = batch.pos.clone()

    result = build_masked_coordinate_corruption(batch)

    assert not bool(result.context_virtual_node_mask[4])
    assert torch.equal(result.input_pos[4], before[4])
    assert bool(result.context_virtual_node_mask[5:14].all())
    assert torch.equal(
        result.input_pos[5:14],
        protein_sidechain_template(dtype=batch.pos.dtype)[1:],
    )


def test_coordinate_corruption_quarantines_bad_frames_and_rna_orientation():
    batch = _coordinate_corruption_batch()
    # Make selected protein N/CA/C collinear.
    batch.pos[0] = torch.tensor([-1.0, 0.0, 0.0])
    batch.pos[1] = torch.tensor([0.0, 0.0, 0.0])
    batch.pos[2] = torch.tensor([1.0, 0.0, 0.0])
    # Flip the independent O2' chirality marker below the RNA frame plane.
    rna_start = 28
    o2_index = batch.pos_code[1].index("O2'")
    batch.pos[rna_start + o2_index] = torch.tensor([20.0, 0.0, -1.0])
    before = batch.pos.clone()

    result = build_masked_coordinate_corruption(batch)
    assert not result.applied_blocks
    assert not bool(result.target_template_node_mask.any())
    assert int(result.context_virtual_node_mask.sum()) == 10
    assert torch.equal(result.clean_pos, before)
    assert torch.equal(
        result.input_pos[~result.context_virtual_node_mask],
        before[~result.context_virtual_node_mask],
    )
    reasons = {
        block.domain: block.reason
        for block in result.coordinate_quarantines
    }
    assert reasons == {
        "protein": "frame:collinear_anchors",
        "rna": "noncanonical_rna_orientation",
    }


def test_edge_rebuild_changes_only_edges_incident_to_template_nodes():
    batch = _attach_graph_local_chain_edges(_coordinate_corruption_batch())
    original_edge_index = batch.edge_index.clone()
    original_edge_attr = batch.edge_attr.clone()
    corruption = build_masked_coordinate_corruption(batch)
    touches_template = (
        corruption.template_node_mask[original_edge_index[0]]
        | corruption.template_node_mask[original_edge_index[1]]
    )

    rebuilt = rebuild_template_incident_edges(
        batch,
        corruption.input_pos,
        corruption.template_node_mask,
        cutoff=5.0,
        max_neighbors=48,
    )
    kept_count = int((~touches_template).sum())

    assert rebuilt.removed_edges == int(touches_template.sum())
    assert torch.equal(
        rebuilt.edge_index[:, :kept_count],
        original_edge_index[:, ~touches_template],
    )
    assert torch.equal(
        rebuilt.edge_attr[:kept_count], original_edge_attr[~touches_template]
    )
    assert not bool(rebuilt.rebuilt_edge_mask[:kept_count].any())
    assert bool(rebuilt.rebuilt_edge_mask[kept_count:].all())

    new_edges = rebuilt.edge_index[:, kept_count:]
    assert bool((
        corruption.template_node_mask[new_edges[0]]
        | corruption.template_node_mask[new_edges[1]]
    ).all())
    assert bool((batch.batch[new_edges[0]] == batch.batch[new_edges[1]]).all())
    assert bool((new_edges[0] != new_edges[1]).all())
    new_distances = torch.linalg.vector_norm(
        corruption.input_pos[new_edges[0]]
        - corruption.input_pos[new_edges[1]],
        dim=1,
    )
    assert bool((new_distances <= 5.0).all())
    assert torch.allclose(
        rebuilt.edge_attr[kept_count:].squeeze(1),
        new_distances,
        atol=0.0,
        rtol=0.0,
    )


def _enabled_runtime_template_config():
    return create_pretraining_config(
        pretraining_tasks=["long_range_distance", "short_range_distance", "mlm"],
        task_weights={
            "long_range_distance": 1.0,
            "short_range_distance": 1.0,
            "mlm": 1.0,
        },
        graph_dim=16,
        protein_block_mlm=True,
        protein_block_mlm_ratio=0.5,
        rna_block_mlm=True,
        rna_block_mlm_ratio=1.0,
        masked_slot_template={
            "enabled": True,
            "protein": True,
            "rna": True,
        },
    )


def _run_enabled_runtime_template():
    config = _enabled_runtime_template_config()
    batch = _attach_graph_local_chain_edges(_coordinate_corruption_batch())
    original = copy.deepcopy(batch)
    torch.manual_seed(4_033)
    PretrainingESAModel._ensure_mlm_attributes(
        _MaskingHarness(config), batch, batch_idx=2, is_validation=True
    )
    masked_state = copy.deepcopy(batch)
    stats = apply_masked_slot_template_(batch, config)
    return batch, original, masked_state, stats, config


def test_flag_on_runtime_applies_one_shared_protein_and_rna_template_selection():
    batch, original, masked_state, stats, _ = _run_enabled_runtime_template()

    assert stats["templated_blocks"] == 2
    assert stats["templated_nodes"] == 31
    assert stats["target_template_nodes"] == 21
    assert stats["context_virtual_blocks"] == 1
    assert stats["context_virtual_nodes"] == 10
    assert stats["removed_edges"] > 0
    assert stats["added_edges"] > 0
    assert stats["excluded_short_edges"] == stats["added_edges"]
    assert stats["excluded_long_pairs"] == 1085
    assert torch.equal(batch.clean_pos, original.pos)
    assert torch.equal(batch.input_pos, batch.pos)
    assert torch.equal(
        batch.pos[~batch.template_node_mask],
        original.pos[~batch.template_node_mask],
    )
    assert int(batch.template_node_mask.sum()) == 10 + 11 + 10
    assert not bool((batch.target_template_node_mask & ~batch.mlm_mask).any())
    assert not bool((batch.context_virtual_node_mask & batch.mlm_mask).any())
    assert torch.equal(batch.z, masked_state.z)
    assert torch.equal(batch.original_types, masked_state.original_types)
    assert torch.equal(batch.mlm_mask, masked_state.mlm_mask)

    # Calling the guard twice in one Lightning step is an idempotent no-op.
    pos_after_first_call = batch.pos.clone()
    edges_after_first_call = batch.edge_index.clone()
    repeated_stats = apply_masked_slot_template_(batch, _enabled_runtime_template_config())
    assert repeated_stats == stats
    assert torch.equal(batch.pos, pos_after_first_call)
    assert torch.equal(batch.edge_index, edges_after_first_call)


def test_enabled_runtime_is_deterministic_for_validation_mask_and_edges():
    first, _, _, first_stats, _ = _run_enabled_runtime_template()
    second, _, _, second_stats, _ = _run_enabled_runtime_template()
    for field in (
        "mlm_mask",
        "template_node_mask",
        "clean_pos",
        "input_pos",
        "pos",
        "edge_index",
        "edge_attr",
        "template_rebuilt_edge_mask",
    ):
        assert torch.equal(getattr(first, field), getattr(second, field))
    for key in ("templated_blocks", "templated_nodes", "removed_edges", "added_edges"):
        assert first_stats[key] == second_stats[key]


def test_prepositioned_context_keeps_visible_v_coordinates_and_rebuilds_only_targets():
    config = _enabled_runtime_template_config()
    config.masked_slot_template["visible_virtuals_prepositioned"] = True
    batch = _attach_graph_local_chain_edges(_coordinate_corruption_batch())
    torch.manual_seed(4_033)
    PretrainingESAModel._ensure_mlm_attributes(
        _MaskingHarness(config), batch, batch_idx=2, is_validation=True
    )
    before_pos = batch.pos.clone()

    stats = apply_masked_slot_template_(batch, config)

    assert stats["target_template_nodes"] == 21
    assert stats["context_virtual_nodes"] == 10
    assert int(batch.template_node_mask.sum()) == 31
    assert int(batch.template_edge_rebuild_node_mask.sum()) == 21
    assert torch.equal(
        batch.pos[batch.context_virtual_node_mask],
        before_pos[batch.context_virtual_node_mask],
    )
    assert bool((
        batch.template_edge_rebuild_node_mask[batch.edge_index[0]]
        | batch.template_edge_rebuild_node_mask[batch.edge_index[1]]
    )[batch.template_rebuilt_edge_mask].all())


def test_distance_losses_ignore_all_template_coordinates():
    batch, _, _, _, config = _run_enabled_runtime_template()
    torch.manual_seed(81_207)
    tasks = PretrainingTasks(config)
    node_embeddings = torch.randn(batch.num_nodes, config.graph_dim)
    node_embeddings[batch.template_node_mask] = torch.nan

    edge_touches_template = (
        batch.template_node_mask[batch.edge_index[0]]
        | batch.template_node_mask[batch.edge_index[1]]
    )
    assert bool((~edge_touches_template).any())

    # Exercise the actual model loss dispatcher. NaN embeddings on every
    # template node make either loss non-finite if even one artificial
    # coordinate is accidentally retained as a supervised distance target.
    config.pretraining_tasks = ["long_range_distance", "short_range_distance"]
    config.task_weights = {
        "long_range_distance": 1.0,
        "short_range_distance": 1.0,
    }
    harness = SimpleNamespace(config=config, pretraining_tasks=tasks)
    losses, total_loss = PretrainingESAModel._compute_pretraining_losses(
        harness,
        batch,
        graph_embeddings=None,
        node_embeddings=node_embeddings,
        is_training=False,
    )
    assert set(losses) == {"long_range_distance", "short_range_distance"}
    assert all(bool(torch.isfinite(loss)) for loss in losses.values())
    assert bool(torch.isfinite(total_loss))


def test_enabled_template_rejects_coordinate_denoising_config():
    try:
        create_pretraining_config(
            pretraining_tasks=["mlm", "coordinate_denoising"],
            protein_block_mlm=True,
            masked_slot_template={
                "enabled": True,
                "protein": True,
                "rna": False,
            },
        )
    except ValueError as error:
        assert "coordinate_denoising" in str(error)
    else:
        raise AssertionError("template and coordinate denoising were enabled together")


def test_enabled_template_excludes_audited_rna_quarantine_from_mlm_target():
    rna_codes = [
        code for code in RNA_BACKBONE_ATOMS if code != "O2'"
    ] + [
        "N9", "C8", "N7", "C6", "N1", "C2", "N3", "C4", "C5", "O6", "N2"
    ]
    num_nodes = len(rna_codes)
    original_types = torch.full((num_nodes,), 6, dtype=torch.long)
    batch = SimpleNamespace(
        z=torch.zeros(num_nodes, dtype=torch.long),
        original_types=original_types.clone(),
        masked_types=torch.zeros(num_nodes, dtype=torch.long),
        pos=torch.arange(num_nodes * 3, dtype=torch.float32).reshape(num_nodes, 3),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
        block_idx=torch.zeros(num_nodes, dtype=torch.long),
        batch=torch.zeros(num_nodes, dtype=torch.long),
        dataset_type=["rna"],
        pos_code=rna_codes,
        mlm_mask=torch.ones(num_nodes, dtype=torch.bool),
        num_graphs=1,
        num_nodes=num_nodes,
    )
    config = _enabled_runtime_template_config()
    config.masked_slot_template["visible_virtuals_prepositioned"] = True

    stats = apply_masked_slot_template_(batch, config)

    assert stats["excluded_quarantined_target_blocks"] == 1
    assert stats["excluded_quarantined_target_nodes"] == num_nodes
    assert stats["target_template_nodes"] == 0
    assert not bool(batch.mlm_mask.any())
    assert torch.equal(batch.z, original_types)
    assert torch.equal(batch.masked_types, original_types)


def test_enabled_template_is_a_noop_for_unimol_only_batch():
    batch = SimpleNamespace(
        z=torch.tensor([6, 8, 7], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.4, 0.0]]
        ),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[1.2], [1.8], [1.4]], dtype=torch.float32),
        block_idx=torch.zeros(3, dtype=torch.long),
        batch=torch.zeros(3, dtype=torch.long),
        dataset_type=["unimol"],
        pos_code=["C1", "O1", "N1"],
        mlm_mask=torch.tensor([True, False, False]),
        num_graphs=1,
        num_nodes=3,
    )
    before = copy.deepcopy(batch)
    stats = apply_masked_slot_template_(batch, _enabled_runtime_template_config())
    assert stats.pop("template_total_ms") >= 0.0
    assert stats == {
        "templated_blocks": 0,
        "templated_nodes": 0,
        "target_template_nodes": 0,
        "excluded_quarantined_target_blocks": 0,
        "excluded_quarantined_target_nodes": 0,
        "context_virtual_blocks": 0,
        "context_virtual_nodes": 0,
        "context_mapping_quarantines": 0,
        "context_coordinate_quarantines": 0,
        "removed_edges": 0,
        "added_edges": 0,
        "excluded_short_edges": 0,
        "excluded_long_pairs": 0,
        "edge_rebuild_ms": 0.0,
    }
    for field in ("z", "pos", "edge_index", "edge_attr", "mlm_mask"):
        assert torch.equal(getattr(batch, field), getattr(before, field))
    assert not hasattr(batch, "template_node_mask")
