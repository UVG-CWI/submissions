#!/usr/bin/env python3
"""
Extract timestamp-filtered synchronized RGB frames from 8 RealSense bag files,
then generate SAM3 foreground masks for each image.

Output layout:
  output/
    001/
      <serial>.png
      mask_<serial>.png
    002/
      ...
    cameras.json
    alignment_metadata.json

Optional stage 3:
  After writing frames and masks, run DA3 masked foreground point-cloud
  reconstruction directly from the same output folder. It can optionally use
  RealSense depth from the bags to correct DA3 scale/shift and filter points by
  multi-view reprojection consistency before writing the fused point cloud.

The extraction follows the simple rule requested here:
  1. Read timestamp_start/timestamp_end from a sequence JSON.
  2. For every bag, keep color frames inside that range.
  3. Verify all cameras have the same number of frames.
  4. Align by frame-list index, not by frame_number.

The metadata is structured so later DA3/GS stages can consume the same frame
folders together with camera serials, timestamps, and c2w/w2c extrinsics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import torch
except ImportError:
    torch = None

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    DepthAnything3 = None


DEFAULT_PRIMARY_PROMPTS = "person in center,human in center,performer in center,actor in center,player in center,speaker in center"
DEFAULT_RELATED_PROMPTS = (
    "held object,object held by a person,object touched by a person,"
    "object in a person's hands,interactive object,performance prop,"
    "sports ball,ball,painting,picture,poster,canvas,artwork,board,sign,"
    "table,pedestal,platform,game table,toy"
)


def parse_prompt_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def read_timestamp_range(sequence_json: Path, seq_name: str | None):
    data = load_json(sequence_json)
    if seq_name is not None:
        if seq_name not in data:
            raise KeyError(f"Sequence {seq_name!r} was not found in {sequence_json}")
        entry = data[seq_name]
    else:
        entry = data

    cwi = entry.get("CWI", entry)
    start = cwi.get("timestamp_start", cwi.get("start"))
    end = cwi.get("timestamp_end", cwi.get("end"))
    if start is None or end is None:
        raise KeyError(
            "Could not find timestamp_start/timestamp_end. Expected either "
            "dataset_sequence style {seq: {CWI: {...}}} or a flat JSON."
        )
    return int(start), int(end)


def get_bag_serial(bag_path: Path) -> str:
    if rs is None:
        raise RuntimeError("pyrealsense2 is required to read .bag files.")
    config = rs.config()
    config.enable_device_from_file(str(bag_path))
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    try:
        return profile.get_device().get_info(rs.camera_info.serial_number)
    finally:
        pipeline.stop()


def extract_color_intrinsics_from_bag(bag_path: Path) -> dict:
    if rs is None:
        raise RuntimeError("pyrealsense2 is required to read .bag files.")
    config = rs.config()
    config.enable_device_from_file(str(bag_path))
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    try:
        stream = profile.get_stream(rs.stream.color)
        intr = stream.as_video_stream_profile().get_intrinsics()
        return {
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "width": int(intr.width),
            "height": int(intr.height),
        }
    finally:
        pipeline.stop()


def discover_bags(bag_dir: Path) -> dict[str, Path]:
    bag_paths = sorted(bag_dir.rglob("*.bag"))
    if not bag_paths:
        raise RuntimeError(f"No .bag files found under {bag_dir}")

    serial_to_bag: dict[str, Path] = {}
    for bag_path in bag_paths:
        serial = get_bag_serial(bag_path)
        if serial in serial_to_bag:
            raise RuntimeError(f"Duplicate bag serial {serial}: {serial_to_bag[serial]} and {bag_path}")
        serial_to_bag[serial] = bag_path
        print(f"Found bag serial {serial}: {bag_path}")
    return serial_to_bag


def build_intrinsics_cache(serials: list[str], serial_to_bag: dict[str, Path], output_path: Path) -> dict:
    intrinsics = {}
    for serial in serials:
        bag_path = serial_to_bag[serial]
        intr = extract_color_intrinsics_from_bag(bag_path)
        intrinsics[serial] = intr
        print(
            f"Intrinsics {serial}: fx={intr['fx']:.2f} fy={intr['fy']:.2f} "
            f"cx={intr['cx']:.2f} cy={intr['cy']:.2f} "
            f"{intr['width']}x{intr['height']}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(intrinsics, f, indent=2)
    print(f"Saved intrinsics: {output_path}")
    return intrinsics


def load_camera_config(camera_config_path: Path):
    config = load_json(camera_config_path)
    cameras = config.get("camera", [])
    if not cameras:
        raise RuntimeError(f"No camera entries found in {camera_config_path}")
    serials = [camera["serial"] for camera in cameras]
    c2w_by_serial = {
        camera["serial"]: np.array(camera["trafo"], dtype=np.float64)
        for camera in cameras
    }
    master_serial = config.get("sync", {}).get("sync_master_serial")
    return config, serials, c2w_by_serial, master_serial


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(matrix, dtype=np.float64))


def get_torch_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.astype(np.float32)
    points_h = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
    return (matrix @ points_h.T).T[:, :3].astype(np.float32)


def back_project_masked(
    depth: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    mask_threshold: int,
    min_depth: float,
    max_depth: float,
    stride: int,
):
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    valid = mask >= mask_threshold
    valid &= np.isfinite(depth)
    valid &= depth > min_depth
    valid &= depth < max_depth

    if stride > 1:
        grid = np.zeros_like(valid, dtype=bool)
        grid[::stride, ::stride] = True
        valid &= grid

    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    ys, xs = np.where(valid)
    z = depth[ys, xs].astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = rgb[ys, xs].astype(np.uint8)
    return points, colors


def write_pointcloud(path: Path, points: np.ndarray, colors: np.ndarray, voxel_size: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    if o3d is None:
        npz_path = path.with_suffix(".npz")
        np.savez_compressed(npz_path, points=points, colors=colors)
        return npz_path

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    if voxel_size > 0 and len(points) > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        if len(pcd.points) >= 32:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud(str(path), pcd)
    return path


def resize_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    if depth.shape[:2] == (height, width):
        return depth.astype(np.float32)
    return cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)


def robust_align_depth_to_sensor(
    da3_depth: np.ndarray,
    sensor_depth: np.ndarray,
    mask: np.ndarray,
    args,
):
    h, w = da3_depth.shape
    sensor_depth = resize_depth(sensor_depth, h, w)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    valid_mask = mask >= args.depth_mask_threshold
    if args.rgbd_fit_erode > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.rgbd_fit_erode * 2 + 1, args.rgbd_fit_erode * 2 + 1),
        )
        valid_mask = cv2.erode(valid_mask.astype(np.uint8), k) > 0

    valid = valid_mask
    valid &= np.isfinite(da3_depth) & np.isfinite(sensor_depth)
    valid &= da3_depth > args.min_depth
    valid &= sensor_depth > args.rgbd_min_sensor_depth
    valid &= sensor_depth < args.rgbd_max_sensor_depth

    da3_values = da3_depth[valid].astype(np.float64)
    sensor_values = sensor_depth[valid].astype(np.float64)
    stats = {
        "used": False,
        "num_candidates": int(len(da3_values)),
        "scale": 1.0,
        "shift": 0.0,
        "num_inliers": 0,
        "inlier_ratio": 0.0,
    }
    if len(da3_values) < args.rgbd_min_fit_points:
        return da3_depth.astype(np.float32), stats

    rng = np.random.default_rng(args.rgbd_ransac_seed)
    if len(da3_values) > args.rgbd_max_fit_points:
        pick = rng.choice(len(da3_values), size=args.rgbd_max_fit_points, replace=False)
        fit_da3 = da3_values[pick]
        fit_sensor = sensor_values[pick]
    else:
        fit_da3 = da3_values
        fit_sensor = sensor_values

    best_inliers = None
    best_count = 0
    for _ in range(args.rgbd_ransac_iters):
        sample_idx = rng.choice(len(fit_da3), size=2, replace=False)
        x = fit_da3[sample_idx]
        y = fit_sensor[sample_idx]
        if abs(x[1] - x[0]) < 1e-6:
            continue
        scale = (y[1] - y[0]) / (x[1] - x[0])
        shift = y[0] - scale * x[0]
        if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0:
            continue
        residual = np.abs(fit_sensor - (scale * fit_da3 + shift))
        inliers = residual < args.rgbd_ransac_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < args.rgbd_min_fit_points:
        return da3_depth.astype(np.float32), stats

    A = np.stack([fit_da3[best_inliers], np.ones(best_count)], axis=1)
    scale, shift = np.linalg.lstsq(A, fit_sensor[best_inliers], rcond=None)[0]
    if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0:
        return da3_depth.astype(np.float32), stats

    aligned_da3 = (scale * da3_depth.astype(np.float32) + shift).astype(np.float32)
    reliable_sensor = valid & (np.abs(sensor_depth - aligned_da3) < args.rgbd_blend_thresh)
    refined = aligned_da3.copy()
    if args.rgbd_sensor_weight > 0:
        refined[reliable_sensor] = (
            args.rgbd_sensor_weight * sensor_depth[reliable_sensor]
            + (1.0 - args.rgbd_sensor_weight) * aligned_da3[reliable_sensor]
        )

    stats.update(
        {
            "used": True,
            "scale": float(scale),
            "shift": float(shift),
            "num_inliers": int(best_count),
            "inlier_ratio": float(best_count / max(1, len(fit_da3))),
            "num_blended_sensor_pixels": int(reliable_sensor.sum()),
        }
    )
    return refined.astype(np.float32), stats


def project_world_to_view(points_world: np.ndarray, w2c: np.ndarray, K: np.ndarray):
    if len(points_world) == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    points_h = np.hstack([points_world, np.ones((len(points_world), 1), dtype=np.float32)])
    points_cam = (w2c @ points_h.T).T[:, :3]
    z = points_cam[:, 2]
    z_safe = np.where(np.abs(z) > 1e-8, z, np.nan)
    u = K[0, 0] * points_cam[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / z_safe + K[1, 2]
    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32)


def filter_points_by_multiview_consistency(
    points_world: np.ndarray,
    source_index: int,
    depths: list[np.ndarray],
    masks: list[np.ndarray],
    Ks: list[np.ndarray],
    w2cs: np.ndarray,
    args,
) -> np.ndarray:
    if not args.rgbd_multiview_filter or len(points_world) == 0:
        return np.ones(len(points_world), dtype=bool)

    support = np.zeros(len(points_world), dtype=np.int32)
    for view_idx, depth in enumerate(depths):
        if view_idx == source_index:
            continue
        h, w = depth.shape
        u, v, z = project_world_to_view(points_world, w2cs[view_idx], Ks[view_idx])
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        inside = (z > args.min_depth) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if not np.any(inside):
            continue
        idx = np.where(inside)[0]
        target_depth = depth[vi[idx], ui[idx]]
        target_mask = masks[view_idx][vi[idx], ui[idx]] >= args.depth_mask_threshold
        tol = np.maximum(args.rgbd_reproj_abs_thresh, args.rgbd_reproj_rel_thresh * target_depth)
        ok = target_mask & np.isfinite(target_depth) & (target_depth > args.min_depth)
        ok &= np.abs(target_depth - z[idx]) <= tol
        support[idx[ok]] += 1

    return support >= args.rgbd_min_consistent_views


def skew_symmetric(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector.reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def fundamental_from_known_poses(K1: np.ndarray, w2c1: np.ndarray, K2: np.ndarray, w2c2: np.ndarray):
    R1, t1 = w2c1[:3, :3], w2c1[:3, 3]
    R2, t2 = w2c2[:3, :3], w2c2[:3, 3]
    R21 = R2 @ R1.T
    t21 = t2 - R21 @ t1
    E = skew_symmetric(t21) @ R21
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    norm = np.linalg.norm(F)
    return F / norm if norm > 0 else F


def sampson_errors(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    pts1_h = np.column_stack([pts1, np.ones(len(pts1), dtype=np.float64)])
    pts2_h = np.column_stack([pts2, np.ones(len(pts2), dtype=np.float64)])
    Fx1 = (F @ pts1_h.T).T
    Ftx2 = (F.T @ pts2_h.T).T
    x2tFx1 = np.sum(pts2_h * Fx1, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.abs(x2tFx1) / np.sqrt(np.maximum(denom, 1e-12))


def projection_matrix(K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    return K @ w2c[:3, :]


def camera_center_from_w2c(w2c: np.ndarray) -> np.ndarray:
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return (-R.T @ t).astype(np.float64)


def triangulate_points_world(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K1: np.ndarray,
    w2c1: np.ndarray,
    K2: np.ndarray,
    w2c2: np.ndarray,
) -> np.ndarray:
    P1 = projection_matrix(K1, w2c1)
    P2 = projection_matrix(K2, w2c2)
    points_h = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64)).T
    denom = points_h[:, 3:4]
    return (points_h[:, :3] / np.where(np.abs(denom) > 1e-12, denom, np.nan)).astype(np.float64)


def reproject_world(points_world: np.ndarray, K: np.ndarray, w2c: np.ndarray):
    u, v, z = project_world_to_view(points_world, w2c, K)
    return np.column_stack([u, v]).astype(np.float64), z.astype(np.float64)


def triangulation_angles(points_world: np.ndarray, w2c1: np.ndarray, w2c2: np.ndarray) -> np.ndarray:
    c1 = camera_center_from_w2c(w2c1)
    c2 = camera_center_from_w2c(w2c2)
    ray1 = points_world - c1[None, :]
    ray2 = points_world - c2[None, :]
    ray1 /= np.maximum(np.linalg.norm(ray1, axis=1, keepdims=True), 1e-12)
    ray2 /= np.maximum(np.linalg.norm(ray2, axis=1, keepdims=True), 1e-12)
    cosang = np.clip(np.sum(ray1 * ray2, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def erode_feature_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    out = (mask >= 128).astype(np.uint8) * 255
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        out = cv2.erode(out, kernel)
    return out


def detect_sift_features(rgb: np.ndarray, mask: np.ndarray, args):
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT is unavailable. Install opencv-contrib-python or use a build with SIFT.")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    feature_mask = erode_feature_mask(mask, args.corr_mask_erode)
    sift = cv2.SIFT_create(nfeatures=args.corr_max_features, contrastThreshold=args.corr_sift_contrast)
    keypoints, descriptors = sift.detectAndCompute(gray, feature_mask)
    if descriptors is None or not keypoints:
        return np.empty((0, 2), dtype=np.float32), None
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return points, descriptors


def match_descriptors(desc1, desc2, args):
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return np.empty((0, 2), dtype=np.int32)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(desc1, desc2, k=2)
    matches = []
    for pair in knn:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < args.corr_ratio * second.distance:
            matches.append((first.queryIdx, first.trainIdx))
    return np.asarray(matches, dtype=np.int32)


def robust_affine_fit(x: np.ndarray, y: np.ndarray, min_points: int, threshold: float, iters: int, seed: int):
    stats = {"used": False, "scale": 1.0, "shift": 0.0, "num_inliers": 0, "inlier_ratio": 0.0}
    if len(x) < min_points:
        return 1.0, 0.0, np.zeros(len(x), dtype=bool), stats
    rng = np.random.default_rng(seed)
    best = np.zeros(len(x), dtype=bool)
    for _ in range(iters):
        sample = rng.choice(len(x), size=2, replace=False)
        if abs(x[sample[1]] - x[sample[0]]) < 1e-6:
            continue
        scale = (y[sample[1]] - y[sample[0]]) / (x[sample[1]] - x[sample[0]])
        shift = y[sample[0]] - scale * x[sample[0]]
        if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0:
            continue
        inliers = np.abs(y - (scale * x + shift)) <= threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < min_points:
        return 1.0, 0.0, best, stats
    A = np.stack([x[best], np.ones(best.sum(), dtype=np.float64)], axis=1)
    scale, shift = np.linalg.lstsq(A, y[best], rcond=None)[0]
    stats.update(
        {
            "used": True,
            "scale": float(scale),
            "shift": float(shift),
            "num_inliers": int(best.sum()),
            "inlier_ratio": float(best.sum() / len(x)),
        }
    )
    return float(scale), float(shift), best, stats


def idw_residual_correction(
    shape: tuple[int, int],
    anchor_uv: np.ndarray,
    residuals: np.ndarray,
    args,
) -> np.ndarray:
    height, width = shape
    if len(anchor_uv) == 0:
        return np.zeros((height, width), dtype=np.float32)

    grid_w = max(8, int(args.corr_grid_width))
    grid_h = max(8, int(round(grid_w * height / max(1, width))))
    xs = np.linspace(0, width - 1, grid_w, dtype=np.float64)
    ys = np.linspace(0, height - 1, grid_h, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.reshape(-1), gy.reshape(-1)])
    correction = np.zeros(len(grid), dtype=np.float64)

    anchors = anchor_uv.astype(np.float64)
    residuals = residuals.astype(np.float64)
    k = min(max(1, int(args.corr_idw_k)), len(anchors))
    for start in range(0, len(grid), 512):
        chunk = grid[start : start + 512]
        diff = chunk[:, None, :] - anchors[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        if k < len(anchors):
            idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
            local_dist2 = np.take_along_axis(dist2, idx, axis=1)
            local_res = residuals[idx]
        else:
            local_dist2 = dist2
            local_res = residuals[None, :]
        weights = 1.0 / np.maximum(local_dist2, 1.0) ** (args.corr_idw_power * 0.5)
        nearest = np.sqrt(np.min(local_dist2, axis=1))
        confidence = np.exp(-((nearest / max(args.corr_max_anchor_distance_px, 1.0)) ** 2))
        correction[start : start + len(chunk)] = confidence * np.sum(weights * local_res, axis=1) / np.sum(weights, axis=1)

    low = correction.reshape(grid_h, grid_w).astype(np.float32)
    return cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)


class MaskedDA3Reconstructor:
    """Inline DA3 foreground point-cloud reconstruction for stage 3."""

    def __init__(
        self,
        frames_root: Path,
        intrinsics_path: Path,
        model_name: str,
        model_path: str | None,
        device: str | None,
        process_res: int,
        process_res_method: str,
        ref_view_strategy: str,
    ):
        if DepthAnything3 is None:
            raise RuntimeError("Depth Anything 3 is not importable.")

        self.frames_root = Path(frames_root)
        self.cameras_meta = load_json(self.frames_root / "cameras.json")
        self.alignment_meta = load_json(self.frames_root / "alignment_metadata.json")
        self.intrinsics = load_json(intrinsics_path)
        self.serials = self.cameras_meta["serials"]
        self.model_name = model_name
        self.model_path = model_path
        self.device = get_torch_device(device)
        self.process_res = process_res
        self.process_res_method = process_res_method
        self.ref_view_strategy = ref_view_strategy
        self._model = None
        self._feature_cache = {}

    def load_model(self):
        if self._model is not None:
            return self._model
        print(f"Loading DA3 on {self.device}...")
        if self.model_path:
            self._model = DepthAnything3.from_pretrained(self.model_path).to(self.device).eval()
        else:
            hf_model_id = {
                "da3-large": "depth-anything/DA3-LARGE-1.1",
                "da3-giant": "depth-anything/DA3-GIANT-1.1",
                "da3nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                "da3metric-large": "depth-anything/DA3METRIC-LARGE",
            }.get(self.model_name, self.model_name)
            try:
                self._model = DepthAnything3.from_pretrained(hf_model_id).to(self.device).eval()
            except Exception as exc:
                print(f"from_pretrained failed ({exc}); falling back to model_name={self.model_name}")
                self._model = DepthAnything3(model_name=self.model_name).to(self.device).eval()
        return self._model

    def load_rgb_mask_sensor_depth(self, image_path: str, mask_path: str, sensor_depth_path: str | None):
        rgb_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Could not read RGB image: {image_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask image: {mask_path}")
        sensor_depth = None
        if sensor_depth_path is not None:
            sensor_depth = np.load(sensor_depth_path).astype(np.float32)
        return rgb, mask, sensor_depth

    def build_frame_inputs(self, frame_meta: dict):
        image_paths = []
        mask_paths = []
        sensor_depth_paths = []
        extrinsics = []
        intrinsics = []
        c2ws = []
        serials_used = []

        for serial in self.serials:
            camera_frame = frame_meta["cameras"].get(serial)
            if camera_frame is None:
                continue
            image_path = self.frames_root / camera_frame["rgb"]
            mask_path = self.frames_root / camera_frame["mask"]
            sensor_depth_path = None
            if camera_frame.get("depth") is not None:
                sensor_depth_path = self.frames_root / camera_frame["depth"]
            if not image_path.exists() or not mask_path.exists():
                print(f"  SKIP {serial}: missing image or mask")
                continue
            if sensor_depth_path is not None and not sensor_depth_path.exists():
                print(f"  WARN {serial}: missing sensor depth {sensor_depth_path}")
                sensor_depth_path = None
            if serial not in self.intrinsics:
                print(f"  SKIP {serial}: missing intrinsics")
                continue

            camera_meta = self.cameras_meta["cameras"][serial]
            w2c = np.array(camera_meta["w2c"], dtype=np.float64)
            c2w = np.array(camera_meta["c2w"], dtype=np.float64)
            intr = self.intrinsics[serial]
            K = np.array(
                [
                    [float(intr["fx"]), 0.0, float(intr["cx"])],
                    [0.0, float(intr["fy"]), float(intr["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            image_paths.append(str(image_path))
            mask_paths.append(str(mask_path))
            sensor_depth_paths.append(str(sensor_depth_path) if sensor_depth_path is not None else None)
            extrinsics.append(w2c)
            intrinsics.append(K)
            c2ws.append(c2w)
            serials_used.append(serial)

        if len(image_paths) < 2:
            raise RuntimeError(f"Frame {frame_meta['frame_name']} has only {len(image_paths)} valid views.")

        return (
            image_paths,
            mask_paths,
            sensor_depth_paths,
            np.stack(extrinsics, axis=0),
            np.stack(intrinsics, axis=0),
            np.stack(c2ws, axis=0),
            serials_used,
        )

    def get_features(self, image_path: str, mask_path: str, args):
        key = (image_path, mask_path, args.corr_mask_erode, args.corr_max_features, args.corr_sift_contrast)
        if key in self._feature_cache:
            return self._feature_cache[key]
        rgb_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Could not read RGB image: {image_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask image: {mask_path}")
        result = detect_sift_features(rgb, mask, args)
        self._feature_cache[key] = result
        return result

    def triangulate_pair_anchors(self, target_idx: int, other_idx: int, view_payloads: list[dict], args):
        target = view_payloads[target_idx]
        other = view_payloads[other_idx]
        pts_t, desc_t = self.get_features(target["image_path"], target["mask_path"], args)
        pts_o, desc_o = self.get_features(other["image_path"], other["mask_path"], args)
        matches = match_descriptors(desc_t, desc_o, args)
        stats = {
            "target": target["serial"],
            "other": other["serial"],
            "num_keypoints_target": int(len(pts_t)),
            "num_keypoints_other": int(len(pts_o)),
            "num_matches": int(len(matches)),
            "num_epipolar": 0,
            "num_anchors": 0,
        }
        if len(matches) < args.corr_min_pair_matches:
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32), stats

        matched_t = pts_t[matches[:, 0]]
        matched_o = pts_o[matches[:, 1]]
        F = fundamental_from_known_poses(target["K_original"], target["w2c"], other["K_original"], other["w2c"])
        epi = sampson_errors(F, matched_t, matched_o)
        keep = epi <= args.corr_epipolar_thresh
        matched_t = matched_t[keep]
        matched_o = matched_o[keep]
        stats["num_epipolar"] = int(len(matched_t))
        if len(matched_t) < args.corr_min_pair_matches:
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32), stats

        points_world = triangulate_points_world(
            matched_t,
            matched_o,
            target["K_original"],
            target["w2c"],
            other["K_original"],
            other["w2c"],
        )
        reproj_t, z_t_original = reproject_world(points_world, target["K_original"], target["w2c"])
        reproj_o, z_o_original = reproject_world(points_world, other["K_original"], other["w2c"])
        err_t = np.linalg.norm(reproj_t - matched_t, axis=1)
        err_o = np.linalg.norm(reproj_o - matched_o, axis=1)
        angles = triangulation_angles(points_world, target["w2c"], other["w2c"])

        depth_uv, depth_z = reproject_world(points_world, target["K"], target["w2c"])
        h, w = target["depth"].shape
        ui = np.rint(depth_uv[:, 0]).astype(np.int32)
        vi = np.rint(depth_uv[:, 1]).astype(np.int32)
        inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        valid = inside
        valid &= np.isfinite(points_world).all(axis=1)
        valid &= z_t_original > args.min_depth
        valid &= z_o_original > args.min_depth
        valid &= depth_z > args.min_depth
        valid &= depth_z < args.max_depth
        valid &= err_t <= args.corr_reproj_thresh
        valid &= err_o <= args.corr_reproj_thresh
        valid &= angles >= args.corr_min_triangulation_angle
        if np.any(valid):
            valid_idx = np.where(valid)[0]
            valid_mask = target["mask"][vi[valid_idx], ui[valid_idx]] >= args.depth_mask_threshold
            da3_values = target["depth"][vi[valid_idx], ui[valid_idx]]
            valid_depth = np.isfinite(da3_values) & (da3_values > args.min_depth) & (da3_values < args.max_depth)
            valid[valid_idx] &= valid_mask & valid_depth

        anchor_uv = depth_uv[valid].astype(np.float32)
        anchor_z = depth_z[valid].astype(np.float32)
        stats["num_anchors"] = int(len(anchor_z))
        return anchor_uv, anchor_z, stats

    def refine_depths_with_correspondence(self, view_payloads: list[dict], output_dir: Path, frame_name: str, args):
        if not args.stage3_correspondence_refine or args.stage3_depth_source == "sensor":
            return view_payloads

        if args.corr_refine_scope == "master":
            master_serial = args.corr_master_serial or self.cameras_meta.get("master_serial") or self.serials[0]
            target_indices = [i for i, payload in enumerate(view_payloads) if payload["serial"] == master_serial]
        else:
            target_indices = list(range(len(view_payloads)))

        debug_dir = output_dir / "correspondence_debug" / frame_name
        if args.corr_save_debug:
            debug_dir.mkdir(parents=True, exist_ok=True)

        for target_idx in target_indices:
            target = view_payloads[target_idx]
            all_uv = []
            all_z = []
            pair_stats = []
            for other_idx, other in enumerate(view_payloads):
                if other_idx == target_idx:
                    continue
                uv, z, stats = self.triangulate_pair_anchors(target_idx, other_idx, view_payloads, args)
                pair_stats.append(stats)
                if len(z) > 0:
                    all_uv.append(uv)
                    all_z.append(z)

            if not all_z:
                target["correspondence_refine"] = {
                    "used": False,
                    "reason": "no_valid_anchors",
                    "pairs": pair_stats,
                }
                continue

            anchor_uv = np.concatenate(all_uv, axis=0)
            anchor_z = np.concatenate(all_z, axis=0)
            h, w = target["depth"].shape
            ui = np.rint(anchor_uv[:, 0]).astype(np.int32)
            vi = np.rint(anchor_uv[:, 1]).astype(np.int32)
            inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            anchor_uv = anchor_uv[inside]
            anchor_z = anchor_z[inside]
            ui = ui[inside]
            vi = vi[inside]
            da3_z = target["depth"][vi, ui].astype(np.float64)
            valid = np.isfinite(da3_z) & np.isfinite(anchor_z)
            valid &= da3_z > args.min_depth
            valid &= anchor_z > args.min_depth
            valid &= anchor_z < args.max_depth
            anchor_uv = anchor_uv[valid]
            anchor_z = anchor_z[valid]
            da3_z = da3_z[valid]

            if len(anchor_z) < args.corr_min_anchors:
                target["correspondence_refine"] = {
                    "used": False,
                    "reason": "too_few_anchors",
                    "num_anchors": int(len(anchor_z)),
                    "pairs": pair_stats,
                }
                continue

            scale, shift, inliers, affine_stats = robust_affine_fit(
                da3_z.astype(np.float64),
                anchor_z.astype(np.float64),
                min_points=args.corr_min_anchors,
                threshold=args.corr_depth_ransac_thresh,
                iters=args.corr_depth_ransac_iters,
                seed=args.corr_seed,
            )
            if not affine_stats["used"]:
                target["correspondence_refine"] = {
                    "used": False,
                    "reason": "affine_fit_failed",
                    "num_anchors": int(len(anchor_z)),
                    "pairs": pair_stats,
                }
                continue

            anchor_uv_in = anchor_uv[inliers]
            anchor_z_in = anchor_z[inliers]
            da3_z_in = da3_z[inliers]
            corrected = (target["depth"].astype(np.float32) * scale + shift).astype(np.float32)
            method = args.corr_correction_method
            if method == "residual":
                residuals = anchor_z_in.astype(np.float32) - corrected[
                    np.rint(anchor_uv_in[:, 1]).astype(np.int32),
                    np.rint(anchor_uv_in[:, 0]).astype(np.int32),
                ]
                correction = idw_residual_correction((h, w), anchor_uv_in, residuals, args)
                corrected = corrected + correction
            elif method == "none":
                corrected = target["depth"]

            corrected = np.where(np.isfinite(corrected), corrected, target["depth"]).astype(np.float32)
            corrected = np.clip(corrected, args.min_depth, args.max_depth)
            target["depth"] = corrected
            target["correspondence_refine"] = {
                "used": method != "none",
                "method": method,
                "num_anchors": int(len(anchor_z)),
                "num_inlier_anchors": int(inliers.sum()),
                "affine": affine_stats,
                "pairs": pair_stats,
            }

            if args.corr_save_debug:
                np.savez_compressed(
                    debug_dir / f"anchors_{target['serial']}.npz",
                    uv=anchor_uv_in.astype(np.float32),
                    z_anchor=anchor_z_in.astype(np.float32),
                    z_da3=da3_z_in.astype(np.float32),
                    scale=np.array([scale], dtype=np.float32),
                    shift=np.array([shift], dtype=np.float32),
                )

        return view_payloads

    def reconstruct_frame(self, frame_meta: dict, output_dir: Path, args):
        image_paths, mask_paths, sensor_depth_paths, ext_w2c, input_K, c2ws, serials = self.build_frame_inputs(frame_meta)

        prediction = None
        depths = None
        rgbs = None
        da3_intrinsics = None
        if args.stage3_depth_source != "sensor":
            model = self.load_model()
            prediction = model.inference(
                image=image_paths,
                extrinsics=ext_w2c,
                intrinsics=input_K,
                align_to_input_ext_scale=True,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                ref_view_strategy=args.ref_view_strategy,
            )
            depths = prediction.depth
            rgbs = prediction.processed_images
            da3_intrinsics = prediction.intrinsics
            if depths is None or len(depths) == 0:
                raise RuntimeError(f"DA3 returned no depth for frame {frame_meta['frame_name']}")

        view_payloads = []
        view_stats = {}
        for i, serial in enumerate(serials):
            if args.stage3_depth_source != "sensor" and i >= len(depths):
                break

            rgb_orig, mask_orig, sensor_depth = self.load_rgb_mask_sensor_depth(
                image_paths[i],
                mask_paths[i],
                sensor_depth_paths[i],
            )

            if args.stage3_depth_source == "sensor":
                if sensor_depth is None:
                    print(f"  SKIP {serial}: sensor depth is required for --stage3-depth-source sensor")
                    continue
                depth = sensor_depth
                rgb = rgb_orig
                mask_for_depth = mask_orig
                K = input_K[i]
                depth_refine_stats = {"used": True, "mode": "sensor"}
            else:
                depth = np.asarray(depths[i], dtype=np.float32)
                if rgbs is not None and i < len(rgbs):
                    rgb = rgbs[i]
                else:
                    rgb = rgb_orig
                K = da3_intrinsics[i] if da3_intrinsics is not None else input_K[i]
                K = np.asarray(K, dtype=np.float64)
                if mask_orig.shape[:2] != depth.shape:
                    mask_for_depth = cv2.resize(
                        mask_orig,
                        (depth.shape[1], depth.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    mask_for_depth = mask_orig

                depth_refine_stats = {"used": False, "mode": "da3"}
                if args.stage3_depth_source == "rgbd_refined":
                    if sensor_depth is None:
                        print(f"  WARN {serial}: no bag depth saved; using DA3 depth")
                    else:
                        depth, depth_refine_stats = robust_align_depth_to_sensor(
                            da3_depth=depth,
                            sensor_depth=sensor_depth,
                            mask=mask_orig,
                            args=args,
                        )
                        depth_refine_stats["mode"] = "rgbd_refined"

            K = np.asarray(K, dtype=np.float64)
            if depth.shape[:2] != mask_for_depth.shape[:2]:
                mask_for_depth = cv2.resize(
                    mask_for_depth,
                    (depth.shape[1], depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            if depth.shape[:2] != rgb.shape[:2]:
                rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

            if not np.any(mask_for_depth >= args.depth_mask_threshold):
                print(f"  SKIP {serial}: empty foreground mask")
                continue

            view_payloads.append(
                {
                    "serial": serial,
                    "depth": depth,
                    "rgb": rgb,
                    "mask": mask_for_depth,
                    "K": K,
                    "K_original": input_K[i],
                    "c2w": c2ws[i],
                    "w2c": ext_w2c[i],
                    "image_path": image_paths[i],
                    "mask_path": mask_paths[i],
                    "depth_refine": depth_refine_stats,
                    "correspondence_refine": {"used": False},
                }
            )

        if not view_payloads:
            raise RuntimeError(f"No valid views for frame {frame_meta['frame_name']}")

        view_payloads = self.refine_depths_with_correspondence(view_payloads, output_dir, frame_meta["frame_name"], args)

        all_points = []
        all_colors = []
        depths_for_consistency = [payload["depth"] for payload in view_payloads]
        masks_for_consistency = [payload["mask"] for payload in view_payloads]
        Ks_for_consistency = [payload["K"] for payload in view_payloads]
        w2cs_for_consistency = np.stack([payload["w2c"] for payload in view_payloads], axis=0)

        for i, payload in enumerate(view_payloads):
            pts_cam, colors = back_project_masked(
                depth=payload["depth"],
                rgb=payload["rgb"],
                mask=payload["mask"],
                K=payload["K"],
                mask_threshold=args.depth_mask_threshold,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                stride=args.pixel_stride,
            )
            pts_world = transform_points(pts_cam, payload["c2w"])
            num_points_before_multiview = len(pts_world)
            keep = filter_points_by_multiview_consistency(
                points_world=pts_world,
                source_index=i,
                depths=depths_for_consistency,
                masks=masks_for_consistency,
                Ks=Ks_for_consistency,
                w2cs=w2cs_for_consistency,
                args=args,
            )
            pts_world = pts_world[keep]
            colors = colors[keep]
            all_points.append(pts_world)
            all_colors.append(colors)
            view_stats[payload["serial"]] = {
                "num_points": int(len(pts_world)),
                "num_points_before_multiview": int(num_points_before_multiview),
                "depth_shape": list(payload["depth"].shape),
                "mask_path": payload["mask_path"],
                "bag_depth_refine": payload["depth_refine"],
                "correspondence_refine": payload["correspondence_refine"],
            }

        if not all_points:
            raise RuntimeError(f"No foreground points for frame {frame_meta['frame_name']}")

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        if len(points) == 0:
            raise RuntimeError(f"No foreground points survived filtering for frame {frame_meta['frame_name']}")
        out_path = output_dir / f"{frame_meta['frame_name']}.ply"
        written = write_pointcloud(out_path, points, colors, args.voxel_size)

        return {
            "frame_name": frame_meta["frame_name"],
            "pointcloud": str(written),
            "num_points_before_downsample": int(len(points)),
            "views": view_stats,
        }

    def run(self, output_dir: Path, args):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = self.alignment_meta["frames"]
        if args.stage3_start_idx:
            frames = frames[args.stage3_start_idx :]
        if args.stage3_max_frames is not None:
            frames = frames[: args.stage3_max_frames]

        results = []
        if args.stage3_depth_source != "sensor":
            self.load_model()
        for i, frame_meta in enumerate(frames, start=1):
            print(f"[{i}/{len(frames)}] reconstruct frame {frame_meta['frame_name']}")
            result = self.reconstruct_frame(frame_meta, output_dir, args)
            results.append(result)
            print(f"  wrote {result['pointcloud']} ({result['num_points_before_downsample']:,} raw pts)")

        with open(output_dir / "reconstruction_metadata.json", "w") as f:
            json.dump(
                {
                    "frames_root": str(self.frames_root),
                    "intrinsics_path": str(args.intrinsics or args.intrinsics_output),
                    "model_name": self.model_name,
                    "process_res": self.process_res,
                    "process_res_method": self.process_res_method,
                    "depth_source": args.stage3_depth_source,
                    "bag_depth_refine": args.stage3_depth_source == "rgbd_refined",
                    "multiview_filter": args.rgbd_multiview_filter,
                    "correspondence_refine": args.stage3_correspondence_refine,
                    "results": results,
                },
                f,
                indent=2,
            )


def extract_frames_from_bag(
    bag_path: Path,
    timestamp_start: int,
    timestamp_end: int,
    align_depth_to_color: bool = True,
    save_depth: bool = False,
):
    """
    Return a list of frames inside [timestamp_start, timestamp_end].

    Each item is:
      {timestamp, frame_number, rgb, depth_m}
    where rgb is RGB uint8 and depth_m is aligned metric depth when requested.
    """
    if rs is None:
        raise RuntimeError("pyrealsense2 is required to read .bag files.")
    config = rs.config()
    config.enable_device_from_file(str(bag_path))
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    device = profile.get_device()
    try:
        depth_scale = float(device.first_depth_sensor().get_depth_scale())
    except RuntimeError:
        depth_scale = 0.001
    aligner = rs.align(rs.stream.color) if align_depth_to_color else None

    frames_out = []
    try:
        while True:
            frames = pipeline.wait_for_frames(5000)
            if aligner is not None:
                frames = aligner.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            depth_m = None
            if save_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_raw = np.asanyarray(depth_frame.get_data()).copy()
                    depth_m = depth_raw.astype(np.float32) * depth_scale
            timestamp = int(color_frame.get_timestamp())
            if timestamp < timestamp_start:
                continue
            if timestamp > timestamp_end:
                break
            rgb = np.asanyarray(color_frame.get_data()).copy()
            frames_out.append(
                {
                    "timestamp": timestamp,
                    "frame_number": int(color_frame.get_frame_number()),
                    "rgb": rgb,
                    "depth_m": depth_m,
                }
            )
    except RuntimeError:
        pass
    finally:
        pipeline.stop()
    return frames_out[::2]


def ensure_uint8_mask(mask, height: int, width: int) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    if mask.dtype != np.bool_:
        mask = mask > 0.5
    return mask.astype(np.uint8) * 255


def union_masks(masks: list[np.ndarray], height: int, width: int) -> np.ndarray:
    if not masks:
        return np.zeros((height, width), dtype=np.uint8)
    union = np.logical_or.reduce([(ensure_uint8_mask(mask, height, width) > 0) for mask in masks])
    return union.astype(np.uint8) * 255


def postprocess_mask(mask: np.ndarray, close: int, open_: int, dilate: int, erode: int, blur: int) -> np.ndarray:
    out = mask.copy()
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close * 2 + 1, close * 2 + 1))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if open_ > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ * 2 + 1, open_ * 2 + 1))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode * 2 + 1, erode * 2 + 1))
        out = cv2.erode(out, k)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        out = cv2.dilate(out, k)
    if blur > 0:
        k = blur * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


class NullMasker:
    def segment_foreground(self, image: Image.Image, args):
        width, height = image.size
        mask = np.full((height, width), 255, dtype=np.uint8)
        return mask, {
            "sam3_disabled": True,
            "foreground_ratio": 1.0,
            "num_primary_instances": 0,
            "num_kept_related_instances": 0,
        }


class Sam3TransformersMasker:
    def __init__(self, args):
        import torch

        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "Transformers SAM3 is unavailable. Install a newer transformers build, "
                "or run with --skip-sam3."
            ) from exc

        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map[args.dtype]

        model_kwargs = {"local_files_only": args.local_files_only}
        if args.device_map is not None:
            model_kwargs["device_map"] = args.device_map
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = Sam3Model.from_pretrained(args.sam3_model, **model_kwargs)
        self.processor = Sam3Processor.from_pretrained(
            args.sam3_model,
            local_files_only=args.local_files_only,
        )

        self.device = None
        if args.device_map is None:
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.device(device)
            self.model.to(self.device)
        self.model.eval()

        self.threshold = args.threshold
        self.mask_threshold = args.mask_threshold
        self.min_area_ratio = args.min_area_ratio

    def _to_device(self, inputs):
        if self.device is not None:
            return inputs.to(self.device)
        device = next(self.model.parameters()).device
        return inputs.to(device)

    def segment_prompt(self, image: Image.Image, prompt: str):
        width, height = image.size
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = self._to_device(inputs)

        import torch

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = inputs.get("original_sizes")
        if hasattr(target_sizes, "tolist"):
            target_sizes = target_sizes.tolist()
        else:
            target_sizes = [(height, width)]

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )[0]

        masks = results.get("masks", [])
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        selected_masks = []
        selected_meta = []
        min_area = self.min_area_ratio * float(height * width)

        for i, mask in enumerate(masks):
            mask_u8 = ensure_uint8_mask(mask, height, width)
            area = int(mask_u8.sum() // 255)
            if area < min_area:
                continue
            if hasattr(scores, "__len__") and i < len(scores):
                score = float(scores[i].detach().cpu()) if hasattr(scores[i], "detach") else float(scores[i])
            else:
                score = 1.0
            meta = {"prompt": prompt, "score": score, "area": area}
            if hasattr(boxes, "__len__") and i < len(boxes):
                box = boxes[i].detach().cpu().numpy() if hasattr(boxes[i], "detach") else np.asarray(boxes[i])
                meta["box"] = box.tolist()
            selected_masks.append(mask_u8)
            selected_meta.append(meta)
        return selected_masks, selected_meta

    def segment_foreground(self, image: Image.Image, args):
        width, height = image.size
        primary_prompts = parse_prompt_list(args.primary_prompts)
        related_prompts = parse_prompt_list(args.related_prompts)

        primary_masks = []
        primary_meta = []
        for prompt in primary_prompts:
            masks, metas = self.segment_prompt(image, prompt)
            primary_masks.extend(masks)
            primary_meta.extend(metas)
        primary_mask = union_masks(primary_masks, height, width)

        related_masks = []
        related_meta = []
        for prompt in related_prompts:
            masks, metas = self.segment_prompt(image, prompt)
            related_masks.extend(masks)
            related_meta.extend(metas)

        kept_related, kept_related_meta = filter_related_masks(
            primary_mask,
            related_masks,
            related_meta,
            dilate_px=args.related_dilate,
            min_touch_ratio=args.related_min_touch_ratio,
        )

        raw_mask = union_masks([primary_mask], height, width)
        final_mask = postprocess_mask(
            raw_mask,
            close=args.close,
            open_=args.open,
            dilate=args.dilate,
            erode=args.erode,
            blur=args.feather,
        )
        return final_mask, {
            "primary_prompts": primary_prompts,
            "related_prompts": related_prompts,
            "primary_instances": primary_meta,
            "kept_related_instances": kept_related_meta,
            "num_primary_instances": len(primary_meta),
            "num_kept_related_instances": len(kept_related_meta),
            "foreground_ratio": float(final_mask.mean() / 255.0),
        }


def filter_related_masks(
    primary_mask: np.ndarray,
    related_masks: list[np.ndarray],
    related_meta: list[dict],
    dilate_px: int,
    min_touch_ratio: float,
):
    if not related_masks:
        return [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    expanded_primary = cv2.dilate((primary_mask > 0).astype(np.uint8) * 255, kernel) > 0

    kept_masks = []
    kept_meta = []
    for mask, meta in zip(related_masks, related_meta):
        mask_bool = mask > 0
        area = int(mask_bool.sum())
        if area == 0:
            continue
        touch_ratio = float(np.logical_and(mask_bool, expanded_primary).sum()) / float(area)
        if touch_ratio >= min_touch_ratio:
            meta = dict(meta)
            meta["touch_ratio"] = touch_ratio
            kept_masks.append(mask)
            kept_meta.append(meta)
    return kept_masks, kept_meta


def write_rgb_and_mask(
    frame_dir: Path,
    serial: str,
    rgb: np.ndarray,
    masker,
    args,
):
    rgb_path = frame_dir / f"{serial}.png"
    mask_path = frame_dir / f"mask_{serial}.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    image = Image.fromarray(rgb)
    mask, mask_meta = masker.segment_foreground(image, args)
    cv2.imwrite(str(mask_path), mask)
    return rgb_path, mask_path, mask_meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 8-camera timestamp-filtered bag frames and SAM3 foreground masks."
    )
    parser.add_argument("--bag-dir", required=True, help="Directory containing 8 .bag files.")
    parser.add_argument("--camera-config", required=True, help="Camera config JSON with serials and extrinsics.")
    parser.add_argument("--sequence-json", required=True, help="File A, e.g. dataset_sequence.json.")
    parser.add_argument("--seq-name", default=None, help="Sequence key in dataset_sequence.json, e.g. BlueSpeech.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--master-serial", default=None, help="Optional override for master camera serial.")
    parser.add_argument("--index-width", type=int, default=3, help="Frame folder width, e.g. 3 -> 001.")
    parser.add_argument("--allow-truncate", action="store_true", help="If frame counts differ, truncate to minimum count.")
    parser.add_argument(
        "--intrinsics-output",
        default=None,
        help="Where to write extracted color intrinsics. Defaults to <output>/camera_intrinsics.json.",
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="Save RealSense depth aligned to color as depth_<serial>.npy for each frame.",
    )
    parser.add_argument("--skip-sam3", action="store_true", help="Save full-white masks without running SAM3.")

    # SAM3 Transformers options.
    parser.add_argument("--sam3-model", default=None, help="Transformers SAM3 model id or local ModelScope directory.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    parser.add_argument("--primary-prompts", default=DEFAULT_PRIMARY_PROMPTS)
    parser.add_argument("--related-prompts", default=DEFAULT_RELATED_PROMPTS)
    parser.add_argument("--related-dilate", type=int, default=120)
    parser.add_argument("--related-min-touch-ratio", type=float, default=0.01)
    parser.add_argument("--close", type=int, default=5)
    parser.add_argument("--open", type=int, default=1)
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--erode", type=int, default=0)
    parser.add_argument("--feather", type=int, default=2)

    # Stage 3: DA3 masked foreground point-cloud reconstruction.
    parser.add_argument(
        "--stage3-reconstruct",
        action="store_true",
        help="After extraction/masking, run DA3 masked foreground point-cloud reconstruction.",
    )
    parser.add_argument(
        "--intrinsics",
        default=None,
        help="Stage 3 input: camera_intrinsics.json. If omitted, uses --intrinsics-output.",
    )
    parser.add_argument(
        "--pointcloud-output",
        default=None,
        help="Stage 3 output directory. Defaults to <output>/masked_pointclouds.",
    )
    parser.add_argument("--da3-model", default="da3-large")
    parser.add_argument("--da3-model-path", default=None)
    parser.add_argument("--da3-device", default=None)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--ref-view-strategy", default="saddle_balanced")
    parser.add_argument("--depth-mask-threshold", type=int, default=128)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--stage3-start-idx", type=int, default=0)
    parser.add_argument("--stage3-max-frames", type=int, default=None)
    parser.add_argument(
        "--stage3-depth-source",
        default="da3",
        choices=["da3", "rgbd_refined", "sensor"],
        help="Depth source for point fusion: DA3, DA3 corrected by bag depth, or raw aligned sensor depth.",
    )
    parser.add_argument(
        "--stage3-use-bag-depth-refine",
        action="store_true",
        help="Legacy alias for --stage3-depth-source rgbd_refined.",
    )
    parser.add_argument("--rgbd-min-sensor-depth", type=float, default=0.15)
    parser.add_argument("--rgbd-max-sensor-depth", type=float, default=8.0)
    parser.add_argument("--rgbd-min-fit-points", type=int, default=500)
    parser.add_argument("--rgbd-max-fit-points", type=int, default=30000)
    parser.add_argument("--rgbd-ransac-iters", type=int, default=96)
    parser.add_argument("--rgbd-ransac-thresh", type=float, default=0.035)
    parser.add_argument("--rgbd-blend-thresh", type=float, default=0.06)
    parser.add_argument("--rgbd-sensor-weight", type=float, default=0.35)
    parser.add_argument("--rgbd-fit-erode", type=int, default=3)
    parser.add_argument("--rgbd-ransac-seed", type=int, default=1234)
    parser.add_argument(
        "--rgbd-multiview-filter",
        action="store_true",
        help="Filter fused points by reprojection consistency against other refined depth views.",
    )
    parser.add_argument("--rgbd-min-consistent-views", type=int, default=1)
    parser.add_argument("--rgbd-reproj-abs-thresh", type=float, default=0.05)
    parser.add_argument("--rgbd-reproj-rel-thresh", type=float, default=0.035)
    parser.add_argument(
        "--stage3-correspondence-refine",
        action="store_true",
        help="Calibrate DA3 depth using SIFT matches triangulated with known intrinsics/extrinsics.",
    )
    parser.add_argument(
        "--corr-refine-scope",
        default="master",
        choices=["master", "all"],
        help="Apply correspondence depth correction only to the master view or to all views.",
    )
    parser.add_argument("--corr-master-serial", default=None)
    parser.add_argument(
        "--corr-correction-method",
        default="residual",
        choices=["affine", "residual", "none"],
        help="Use global affine correction, affine plus sparse residual field, or debug-only no correction.",
    )
    parser.add_argument("--corr-max-features", type=int, default=8000)
    parser.add_argument("--corr-sift-contrast", type=float, default=0.01)
    parser.add_argument("--corr-ratio", type=float, default=0.75)
    parser.add_argument("--corr-mask-erode", type=int, default=7)
    parser.add_argument("--corr-min-pair-matches", type=int, default=24)
    parser.add_argument("--corr-min-anchors", type=int, default=80)
    parser.add_argument("--corr-epipolar-thresh", type=float, default=2.0)
    parser.add_argument("--corr-reproj-thresh", type=float, default=3.0)
    parser.add_argument("--corr-min-triangulation-angle", type=float, default=1.0)
    parser.add_argument("--corr-depth-ransac-thresh", type=float, default=0.08)
    parser.add_argument("--corr-depth-ransac-iters", type=int, default=128)
    parser.add_argument("--corr-grid-width", type=int, default=72)
    parser.add_argument("--corr-idw-k", type=int, default=24)
    parser.add_argument("--corr-idw-power", type=float, default=2.0)
    parser.add_argument("--corr-max-anchor-distance-px", type=float, default=160.0)
    parser.add_argument("--corr-seed", type=int, default=2026)
    parser.add_argument("--corr-save-debug", action="store_true")
    return parser.parse_args()


def run_stage3_reconstruction(args, frames_root: Path):
    """Run DA3 masked point-cloud reconstruction from the extracted frames/masks."""
    intrinsics_path = args.intrinsics or args.intrinsics_output
    if not intrinsics_path:
        raise RuntimeError("--stage3-reconstruct requires --intrinsics or --intrinsics-output")

    pointcloud_output = Path(args.pointcloud_output) if args.pointcloud_output else frames_root / "masked_pointclouds"
    rec = MaskedDA3Reconstructor(
        frames_root=frames_root,
        intrinsics_path=Path(intrinsics_path),
        model_name=args.da3_model,
        model_path=args.da3_model_path,
        device=args.da3_device or args.device,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        ref_view_strategy=args.ref_view_strategy,
    )
    rec.run(pointcloud_output, args)


def main():
    args = parse_args()
    if args.stage3_use_bag_depth_refine and args.stage3_depth_source == "da3":
        args.stage3_depth_source = "rgbd_refined"
    bag_dir = Path(args.bag_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.intrinsics_output is None:
        args.intrinsics_output = str(output_dir / "camera_intrinsics.json")

    timestamp_start, timestamp_end = read_timestamp_range(Path(args.sequence_json), args.seq_name)
    print(f"Timestamp range: {timestamp_start} -> {timestamp_end}")

    camera_config, serials, c2w_by_serial, config_master = load_camera_config(Path(args.camera_config))
    master_serial = args.master_serial or config_master or serials[0]
    if master_serial not in serials:
        raise RuntimeError(f"Master serial {master_serial} is not in camera config.")
    print(f"Master serial: {master_serial}")

    serial_to_bag = discover_bags(bag_dir)
    missing = [serial for serial in serials if serial not in serial_to_bag]
    if missing:
        raise RuntimeError(f"Missing bag files for camera serials: {missing}")

    intrinsics_cache = build_intrinsics_cache(
        serials=serials,
        serial_to_bag=serial_to_bag,
        output_path=Path(args.intrinsics_output),
    )

    save_depth = bool(args.save_depth or args.stage3_depth_source in {"rgbd_refined", "sensor"})
    frames_by_serial = {}
    for serial in serials:
        print(f"Extracting timestamps from {serial}...")
        frames = extract_frames_from_bag(
            serial_to_bag[serial],
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            save_depth=save_depth,
        )
        frames_by_serial[serial] = frames
        print(f"  {serial}: {len(frames)} frames")

    counts = {serial: len(frames_by_serial[serial]) for serial in serials}
    unique_counts = sorted(set(counts.values()))
    if len(unique_counts) != 1:
        print(f"Frame count mismatch: {counts}")
        if not args.allow_truncate:
            raise RuntimeError("Frame counts differ. Re-run with --allow-truncate to use the minimum count.")
        num_frames = min(unique_counts)
        print(f"Truncating all cameras to {num_frames} frames.")
    else:
        num_frames = unique_counts[0]

    if args.skip_sam3:
        masker = NullMasker()
    else:
        if not args.sam3_model:
            raise RuntimeError("--sam3-model is required unless --skip-sam3 is set.")
        masker = Sam3TransformersMasker(args)

    cameras_meta = {
        "master_serial": master_serial,
        "serials": serials,
        "camera_config": str(Path(args.camera_config)),
        "cameras": {},
    }
    for serial in serials:
        c2w = c2w_by_serial[serial]
        w2c = invert_transform(c2w)
        cameras_meta["cameras"][serial] = {
            "bag_path": str(serial_to_bag[serial]),
            "c2w": c2w.tolist(),
            "w2c": w2c.tolist(),
            "intrinsics": intrinsics_cache.get(serial),
        }

    frames_meta = []
    for frame_idx in range(num_frames):
        frame_name = f"{(frame_idx + 1):0{args.index_width}d}"
        frame_dir = output_dir / frame_name
        frame_dir.mkdir(parents=True, exist_ok=True)

        master_ts = frames_by_serial[master_serial][frame_idx]["timestamp"]
        frame_meta = {
            "frame_index": frame_idx + 1,
            "frame_name": frame_name,
            "master_serial": master_serial,
            "master_timestamp": master_ts,
            "cameras": {},
        }

        for serial in serials:
            item = frames_by_serial[serial][frame_idx]
            rgb_path, mask_path, mask_meta = write_rgb_and_mask(
                frame_dir,
                serial,
                item["rgb"],
                masker,
                args,
            )
            depth_rel = None
            if save_depth and item.get("depth_m") is not None:
                depth_path = frame_dir / f"depth_{serial}.npy"
                np.save(depth_path, item["depth_m"].astype(np.float32))
                depth_rel = str(depth_path.relative_to(output_dir))
            timestamp = item["timestamp"]
            frame_meta["cameras"][serial] = {
                "timestamp": timestamp,
                "frame_number": item["frame_number"],
                "delta_from_master_ms": int(timestamp - master_ts),
                "rgb": str(rgb_path.relative_to(output_dir)),
                "mask": str(mask_path.relative_to(output_dir)),
                "depth": depth_rel,
                "mask_meta": mask_meta,
            }

        frames_meta.append(frame_meta)
        print(f"[{frame_idx + 1}/{num_frames}] wrote {frame_dir}")
        
        if frame_idx > 10:
            break

    with open(output_dir / "cameras.json", "w") as f:
        json.dump(cameras_meta, f, indent=2)

    metadata = {
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "num_frames": num_frames,
        "frame_counts": counts,
        "output_layout": "<frame_index>/<serial>.png, mask_<serial>.png, and optional depth_<serial>.npy",
        "saved_aligned_depth": save_depth,
        "frames": frames_meta,
    }
    with open(output_dir / "alignment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Output: {output_dir}")
    print(f"Cameras: {output_dir / 'cameras.json'}")
    print(f"Metadata: {output_dir / 'alignment_metadata.json'}")

    if args.stage3_reconstruct:
        print("\n[Stage 3] Reconstructing masked foreground point clouds with DA3")
        run_stage3_reconstruction(args, output_dir)


if __name__ == "__main__":
    main()
