from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"performance equivalence tests require scipy: {exc}") from exc


BASELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_DIR))

from infer_gqenet_dqpc import run_models, select_centers
from recolor_gaussian import gaussian_recolor
from summarize_validation import collect_runtime
from temporal_smooth import map_colors


def gaussian_reference(source_xyz, source_rgb, target_xyz, k, sigma, copy_distance):
    k_eff = min(k, source_xyz.shape[0])
    dist, idx = cKDTree(source_xyz).query(target_xyz, k=k_eff, workers=-1)
    if k_eff == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    if sigma is None:
        kth = dist[:, -1]
        finite = kth[np.isfinite(kth) & (kth > 0)]
        sigma = float(np.median(finite)) if finite.size else 1.0
    weights = np.exp(-(dist * dist) / (2.0 * max(float(sigma), 1e-12) ** 2))
    weight_sum = weights.sum(axis=1, keepdims=True)
    nearest = source_rgb[idx[:, 0]]
    output = (source_rgb[idx] * weights[:, :, None]).sum(axis=1) / np.maximum(weight_sum, 1e-12)
    output[weight_sum[:, 0] <= 1e-12] = nearest[weight_sum[:, 0] <= 1e-12]
    output[dist[:, 0] <= copy_distance] = nearest[dist[:, 0] <= copy_distance]
    return output


class DummyGQE(torch.nn.Module):
    def forward(self, data):
        return (data[:, 3, :] + 0.1 * data[:, 0, :]).unsqueeze(-1)


class PerformanceEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_gaussian_chunking_preserves_v3_formula(self):
        source_xyz = self.rng.normal(size=(91, 3))
        source_rgb = self.rng.integers(0, 256, size=(91, 3)).astype(np.float32)
        target_xyz = self.rng.normal(size=(137, 3))
        expected = gaussian_reference(source_xyz, source_rgb, target_xyz, 8, None, 0.05)
        actual = gaussian_recolor(
            source_xyz,
            source_rgb,
            target_xyz,
            8,
            None,
            copy_distance=0.05,
            query_chunk_size=17,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_temporal_chunking_preserves_mapping(self):
        ref_xyz = self.rng.normal(size=(83, 3))
        ref_rgb = self.rng.integers(0, 256, size=(83, 3)).astype(np.float32)
        cur_xyz = self.rng.normal(size=(109, 3))
        cur_rgb = self.rng.integers(0, 256, size=(109, 3)).astype(np.float32)
        tree = cKDTree(ref_xyz)
        dist, idx = tree.query(cur_xyz, k=1, workers=-1)
        expected_mapped = ref_rgb[idx]
        expected_valid = (dist <= 0.8) & (
            np.linalg.norm(expected_mapped.astype(np.float64) - cur_rgb.astype(np.float64), axis=1) <= 90
        )
        mapped, valid = map_colors(
            ref_xyz,
            ref_rgb,
            cur_xyz,
            cur_rgb,
            0.8,
            90,
            query_chunk_size=13,
            ref_tree=tree,
        )
        np.testing.assert_array_equal(mapped, expected_mapped)
        np.testing.assert_array_equal(valid, expected_valid)

    def test_gqe_v3_centers_and_shared_geometry(self):
        xyz = self.rng.normal(size=(64, 3))
        yuv = self.rng.normal(size=(64, 3)).astype(np.float32)
        args = SimpleNamespace(batch_size=3, patch_size=16, coord_scale=1.0)
        centers = select_centers(xyz, args.patch_size, 8, "v3_stride")
        np.testing.assert_array_equal(centers, xyz[::8])
        output, covered = run_models(
            [DummyGQE(), DummyGQE(), DummyGQE()],
            xyz,
            yuv,
            centers,
            args,
            torch.device("cpu"),
        )

        tree = cKDTree(xyz)
        expected_accum = np.zeros((3, xyz.shape[0]), dtype=np.float64)
        expected_count = np.zeros(xyz.shape[0], dtype=np.float64)
        for center in centers:
            _, idx = tree.query(center[None, :], k=args.patch_size, workers=-1)
            idx = np.asarray(idx).reshape(-1)
            patch = xyz[idx]
            normalized = (patch - patch.mean(axis=0, keepdims=True)).astype(np.float32)
            expected_count[idx] += 1
            for channel in range(3):
                expected_accum[channel, idx] += yuv[idx, channel] + 0.1 * normalized[:, 0]
        expected = np.zeros_like(output)
        expected[expected_count > 0] = (
            expected_accum[:, expected_count > 0] / expected_count[expected_count > 0]
        ).T
        np.testing.assert_allclose(output, expected, rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(covered, expected_count > 0)

    def test_runtime_mean_sums_stage_means_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage_a = root / "valid_geometry.jsonl"
            stage_b = root / "valid_recolor.jsonl"
            stage_a.write_text(
                "\n".join(
                    [
                        json.dumps({"output": "a.ply", "seconds": 1.0}),
                        json.dumps({"output": "b.ply", "seconds": 3.0}),
                        json.dumps({"output": "a.ply", "seconds": 2.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            stage_b.write_text(
                "\n".join(
                    [
                        json.dumps({"output": "a.ply", "seconds": 4.0}),
                        json.dumps({"output": "b.ply", "seconds": 6.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            result = collect_runtime(str(root / "*.jsonl"))
            self.assertEqual(result["runtime_record_count"], 4)
            self.assertEqual(result["runtime_stage_count"], 2)
            self.assertAlmostEqual(result["runtime_seconds_mean"], 7.5)


if __name__ == "__main__":
    unittest.main()
