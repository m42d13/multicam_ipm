"""Pure NumPy/OpenCV implementation of equidistant ground-plane projection."""

from __future__ import annotations

import cv2
import numpy as np

from .models import CameraModel, GridSpec, ProjectionMap


class BirdseyeProjector:
    """Precompute image-remap tables and render a flat-ground BEV image."""

    def __init__(self, cameras: list[CameraModel], grid: GridSpec, camera_aligned: bool):
        self.cameras = cameras
        self.grid = grid
        self.camera_aligned = camera_aligned
        self.maps = {camera.index: self._build_map(camera) for camera in cameras}
        weights = np.stack([self.maps[camera.index].weight for camera in cameras])
        self.owner_map = np.argmax(weights, axis=0)
        self.owner_valid = np.max(weights, axis=0) > 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    @property
    def coverage(self) -> float:
        coverage = np.zeros(self.shape, dtype=bool)
        for projection in self.maps.values():
            coverage |= projection.weight > 0.0
        return float(np.count_nonzero(coverage)) / coverage.size

    def render(self, frames: dict[int, np.ndarray], blend_mode: str) -> np.ndarray:
        """Warp cached camera frames and compose them using the requested blend."""
        blend_mode = blend_mode.lower()
        if blend_mode not in ('winner', 'feather'):
            raise ValueError("blend_mode must be 'winner' or 'feather'")
        height, width = self.shape
        if blend_mode == 'winner':
            output = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            output = np.zeros((height, width, 3), dtype=np.float32)
            total_weight = np.zeros((height, width), dtype=np.float32)

        for owner, camera in enumerate(self.cameras):
            projection = self.maps[camera.index]
            warped = cv2.remap(
                frames[camera.index], projection.source_x, projection.source_y,
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )
            if blend_mode == 'winner':
                selected = self.owner_valid & (self.owner_map == owner)
                output[selected] = warped[selected]
            else:
                output += warped.astype(np.float32) * projection.weight[..., None]
                total_weight += projection.weight.astype(np.float32)

        if blend_mode == 'winner':
            return output
        valid = total_weight > 0.0
        result = np.zeros_like(output, dtype=np.uint8)
        result[valid] = np.clip(output[valid] / total_weight[valid, None], 0, 255).astype(np.uint8)
        return result

    def vehicle_mask(self, vehicle_length: float, vehicle_width: float, center_x: float) -> np.ndarray:
        """Return an ego-frame rectangular vehicle mask for display-only hiding."""
        if self.camera_aligned:
            raise ValueError('mask_vehicle requires camera_aligned_output:=false')
        rows, cols = np.indices(self.shape)
        x = self.grid.x_max - (rows + 0.5) * self.grid.meters_per_pixel
        y = self.grid.y_max - (cols + 0.5) * self.grid.meters_per_pixel
        return ((np.abs(x - center_x) <= vehicle_length / 2.0) & (np.abs(y) <= vehicle_width / 2.0))

    def _build_map(self, camera: CameraModel) -> ProjectionMap:
        longitudinal, lateral = self._ground_grid_for(camera)
        ego_points = np.stack((
            longitudinal, lateral, np.zeros_like(longitudinal), np.ones_like(longitudinal),
        ), axis=0).reshape(4, -1)
        x, y, z = (camera.camera_from_ego @ ego_points)[:3]
        radius = np.hypot(x, y)
        theta = np.arctan2(radius, z)
        k1, k2, k3, k4 = camera.distortion
        theta2 = theta * theta
        distorted_theta = theta * (1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))
        factor = np.divide(distorted_theta, radius, out=np.zeros_like(theta), where=radius > 1e-9)
        fx, fy, cx, cy = camera.intrinsics
        height, width = self.shape
        source_x = (fx * x * factor + cx).reshape(height, width)
        source_y = (fy * y * factor + cy).reshape(height, width)
        source_width, source_height = camera.resolution
        valid = ((z > 0.0) & (source_x.ravel() >= 0.0) & (source_x.ravel() < source_width - 1)
                 & (source_y.ravel() >= 0.0) & (source_y.ravel() < source_height - 1)).reshape(height, width)
        incidence = np.divide(z, np.linalg.norm(np.stack((x, y, z)), axis=0), out=np.zeros_like(z), where=z > 0.0)
        weight = np.where(valid, np.maximum(incidence.reshape(height, width), 0.05), 0.0).astype(np.float32)
        return ProjectionMap(source_x.astype(np.float32), source_y.astype(np.float32), weight)

    def _ground_grid_for(self, camera: CameraModel) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = np.indices(self.shape, dtype=np.float64)
        longitudinal = self.grid.x_max - (rows + 0.5) * self.grid.meters_per_pixel
        lateral = self.grid.y_max - (cols + 0.5) * self.grid.meters_per_pixel
        if not self.camera_aligned:
            return longitudinal, lateral

        ego_from_camera = np.linalg.inv(camera.camera_from_ego)
        forward = ego_from_camera[:2, 2]
        left = -ego_from_camera[:2, 0]
        forward_norm = np.linalg.norm(forward)
        left_norm = np.linalg.norm(left)
        if forward_norm < 1e-9 or left_norm < 1e-9:
            raise ValueError(
                f'camera {camera.name} has no horizontal pose direction; '
                'provide a rig calibration or use camera_aligned_output:=false'
            )
        forward /= forward_norm
        left /= left_norm
        position = ego_from_camera[:3, 3]
        return (
            position[0] + forward[0] * longitudinal + left[0] * lateral,
            position[1] + forward[1] * longitudinal + left[1] * lateral,
        )
