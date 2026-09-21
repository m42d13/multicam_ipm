"""Small data models shared by the calibration and projection layers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """One equidistant camera and its pose relative to the vehicle frame."""

    index: int
    name: str
    topic: str
    intrinsics: np.ndarray
    distortion: np.ndarray
    resolution: tuple[int, int]
    camera_from_ego: np.ndarray


@dataclass(frozen=True)
class GridSpec:
    """Metric BEV raster bounds in an ego or camera-aligned ground frame."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    meters_per_pixel: float

    def __post_init__(self):
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError('IPM bounds must have positive width and height')
        if self.meters_per_pixel <= 0.0:
            raise ValueError('meters_per_pixel must be positive')

    @property
    def shape(self) -> tuple[int, int]:
        height = int(np.ceil((self.x_max - self.x_min) / self.meters_per_pixel))
        width = int(np.ceil((self.y_max - self.y_min) / self.meters_per_pixel))
        return height, width


@dataclass(frozen=True)
class ProjectionMap:
    """OpenCV remap coordinates and a confidence weight for a BEV grid."""

    source_x: np.ndarray
    source_y: np.ndarray
    weight: np.ndarray
