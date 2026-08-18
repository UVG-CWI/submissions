from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


GQE_ROOT = Path(__file__).resolve().parents[3] / "external" / "GQE-Net"
sys.path.insert(0, str(GQE_ROOT))

import model_GQE_Net_final as gqe


class GQETensorEquivalenceTests(unittest.TestCase):
    def setUp(self):
        gqe.devices = "cpu"

    def test_knn_reuses_one_topk_without_changing_result(self):
        torch.manual_seed(3)
        points = torch.randn(2, 3, 32)
        inner = -2 * torch.matmul(points.transpose(2, 1), points)
        squared = torch.sum(points**2, dim=1, keepdim=True)
        pairwise = -squared - inner - squared.transpose(2, 1)
        expected_distance = pairwise.topk(k=20, dim=-1)[0]
        expected_index = pairwise.topk(k=20, dim=-1)[1]
        index, distance = gqe.knn(points, 20)
        torch.testing.assert_close(distance, expected_distance, rtol=0.0, atol=0.0)
        torch.testing.assert_close(index, expected_index, rtol=0.0, atol=0.0)

    def test_broadcast_weight_matches_materialized_repeat(self):
        torch.manual_seed(4)
        feature = torch.randn(2, 130, 16, 20)
        weight = torch.randn(2, 1, 16, 20)
        expected = feature * weight.repeat(1, 130, 1, 1)
        actual = feature * weight
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_graph_feature_expand_matches_repeat(self):
        torch.manual_seed(6)
        feature = torch.randn(2, 4, 24)
        position = torch.randn(2, 3, 24)
        index, distance = gqe.knn(position, 20)
        edge, mixed, _, _ = gqe.get_graph_feature(
            feature,
            position,
            index,
            distance,
            20,
        )

        batch_size, channels, point_count = feature.shape
        index_base = torch.arange(batch_size).view(-1, 1, 1) * point_count
        flat_index = (index + index_base).reshape(-1)
        point_feature = feature.transpose(2, 1).contiguous()
        neighbor = point_feature.reshape(batch_size * point_count, channels)[flat_index]
        neighbor = neighbor.reshape(batch_size, point_count, 20, channels)
        center = point_feature.reshape(batch_size, point_count, 1, channels).repeat(1, 1, 20, 1)
        expected_edge = (neighbor - center).permute(0, 3, 1, 2).contiguous()
        expected_mixed = torch.cat((neighbor - center, center), dim=3).permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(edge, expected_edge, rtol=0.0, atol=0.0)
        torch.testing.assert_close(mixed, expected_mixed, rtol=0.0, atol=0.0)

    def test_normal_keeps_batch_dimension_for_batch_one(self):
        torch.manual_seed(5)
        points = torch.randn(1, 24, 3)
        index, _ = gqe.knn(points.transpose(2, 1), 20)
        normals = gqe.Normal(points, index, 20)
        self.assertEqual(tuple(normals.shape), (1, 24, 3))

    def test_full_model_supports_batch_one(self):
        torch.manual_seed(7)
        model = gqe.GAPCN().eval()
        with torch.no_grad():
            output = model(torch.randn(1, 4, 24))
        self.assertEqual(tuple(output.shape), (1, 24, 1))


if __name__ == "__main__":
    unittest.main()
