from types import SimpleNamespace

import torch
from torch import nn

from core.pretraining_tasks import PretrainingTasks


def _tasks() -> PretrainingTasks:
    config = SimpleNamespace(
        graph_dim=4,
        distance_bins=4,
        max_distance=20.0,
        distance_bin_boundaries=None,
        short_range_distance_bins=4,
        short_range_max_distance=5.0,
        short_range_bin_boundaries=None,
        atom_types=121,
    )
    return PretrainingTasks(config)


class _RecordingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4, bias=False)
        self.last_input = None

    def forward(self, edge_embeddings):
        self.last_input = edge_embeddings.detach().clone()
        return self.linear(edge_embeddings)


def test_short_range_head_never_sees_edges_beyond_five_angstrom():
    tasks = _tasks()
    head = _RecordingHead()
    tasks.distance_head = head
    node_embeddings = torch.randn(4, 4, requires_grad=True)
    edge_index = torch.tensor(
        [[0, 0, 0, 0], [1, 2, 3, 1]], dtype=torch.long
    )
    distances = torch.tensor([2.0, 5.0, 5.01, float("inf")])
    selected = torch.ones(4, dtype=torch.bool)

    loss = tasks.short_range_distance_loss(
        node_embeddings, edge_index, distances, selected
    )

    expected_input = torch.stack(
        [
            torch.cat((node_embeddings[0], node_embeddings[1])),
            torch.cat((node_embeddings[0], node_embeddings[2])),
        ]
    ).detach()
    torch.testing.assert_close(head.last_input, expected_input)
    loss.backward()
    assert float(node_embeddings.grad[3].abs().sum()) == 0.0


def test_short_range_loss_returns_connected_zero_when_no_edge_is_in_range():
    tasks = _tasks()
    node_embeddings = torch.randn(3, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    distances = torch.tensor([5.01, float("nan")])
    selected = torch.ones(2, dtype=torch.bool)

    loss = tasks.short_range_distance_loss(
        node_embeddings, edge_index, distances, selected
    )

    assert float(loss) == 0.0
    loss.backward()
    assert node_embeddings.grad is not None
    assert float(node_embeddings.grad.abs().sum()) == 0.0


def test_long_range_loss_never_pairs_atoms_from_different_graphs():
    tasks = _tasks()
    tasks.long_range_distance_head = _RecordingHead()
    captured_distances = []

    def capture_bins(distances):
        captured_distances.append(distances.detach().clone())
        return torch.zeros_like(distances, dtype=torch.long)

    tasks._distances_to_bins = capture_bins
    node_embeddings = torch.randn(4, 4, requires_grad=True)
    data = SimpleNamespace(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [103.0, 0.0, 0.0],
            ]
        ),
        batch=torch.tensor([0, 0, 1, 1]),
    )

    loss = tasks.long_range_distance_loss(node_embeddings, data)

    torch.testing.assert_close(
        torch.sort(captured_distances[0]).values, torch.tensor([1.0, 3.0])
    )
    loss.backward()
    assert node_embeddings.grad is not None


def test_long_range_loss_excludes_geometry_invalid_nodes():
    tasks = _tasks()
    tasks.long_range_distance_head = _RecordingHead()
    captured_distances = []

    def capture_bins(distances):
        captured_distances.append(distances.detach().clone())
        return torch.zeros_like(distances, dtype=torch.long)

    tasks._distances_to_bins = capture_bins
    node_embeddings = torch.randn(3, 4, requires_grad=True)
    data = SimpleNamespace(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ]
        ),
        batch=torch.zeros(3, dtype=torch.long),
        geometry_valid_mask=torch.tensor([True, True, False]),
    )

    loss = tasks.long_range_distance_loss(node_embeddings, data)

    torch.testing.assert_close(captured_distances[0], torch.tensor([1.0]))
    loss.backward()
    assert float(node_embeddings.grad[2].abs().sum()) == 0.0


def test_long_range_sampling_remains_graph_local_when_pair_cap_is_hit():
    batch_mapping = torch.cat(
        [torch.zeros(60, dtype=torch.long), torch.ones(60, dtype=torch.long)]
    )
    eligible_indices = torch.arange(120)

    left, right = PretrainingTasks._sample_graph_local_pairs(
        batch_mapping, eligible_indices, max_pairs=1000
    )

    assert left.numel() == 1000
    assert right.numel() == 1000
    assert bool(torch.all(batch_mapping[left] == batch_mapping[right]))
    assert bool(torch.all(left != right))


def test_long_range_loss_returns_connected_zero_without_an_intra_graph_pair():
    tasks = _tasks()
    node_embeddings = torch.randn(2, 4, requires_grad=True)
    data = SimpleNamespace(
        pos=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        batch=torch.tensor([0, 1]),
    )

    loss = tasks.long_range_distance_loss(node_embeddings, data)

    assert float(loss) == 0.0
    loss.backward()
    assert node_embeddings.grad is not None
    assert float(node_embeddings.grad.abs().sum()) == 0.0
