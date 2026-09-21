"""Load Kalibr and rig calibration files into validated camera models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from .models import CameraModel


def load_cameras(
    camchain_file: str,
    rig_calibration_file: str = '',
    topic_suffix_override: str = '',
    fallback_cam0_from_ego: np.ndarray | None = None,
) -> list[CameraModel]:
    """Load a Kalibr camchain and optional absolute camera poses.

    Kalibr stores ``T_cn_cnm1`` as the transform from camera ``n - 1`` to
    camera ``n``. A rig JSON, when supplied, takes precedence because it gives
    an absolute ego pose for every camera.
    """
    path = Path(camchain_file).expanduser()
    if not camchain_file:
        raise RuntimeError('camchain_file must be set')
    with path.open(encoding='utf-8') as stream:
        camchain = yaml.safe_load(stream)
    if not isinstance(camchain, dict):
        raise RuntimeError(f'{path} is not a Kalibr camchain mapping')

    rig_sensors = _load_rig_sensors(rig_calibration_file)
    keys = _camera_keys(camchain)
    cam0_from_ego = (
        np.eye(4, dtype=np.float64)
        if fallback_cam0_from_ego is None
        else np.asarray(fallback_cam0_from_ego, dtype=np.float64)
    )
    if cam0_from_ego.shape != (4, 4):
        raise RuntimeError('fallback cam0 pose must be a 4x4 transform')

    camera_from_cam0 = np.eye(4, dtype=np.float64)
    cameras = []
    for index, key in enumerate(keys):
        config = camchain[key]
        if index:
            step = np.asarray(config.get('T_cn_cnm1'), dtype=np.float64)
            if step.shape != (4, 4):
                raise RuntimeError(f'{key}.T_cn_cnm1 must be a 4x4 transform')
            camera_from_cam0 = step @ camera_from_cam0

        intrinsics = np.asarray(config.get('intrinsics'), dtype=np.float64)
        distortion = np.asarray(config.get('distortion_coeffs'), dtype=np.float64)
        resolution = tuple(config.get('resolution', ()))
        if intrinsics.shape != (4,) or distortion.shape != (4,):
            raise RuntimeError(f'{key} must contain four intrinsics and four distortion coefficients')
        if len(resolution) != 2:
            raise RuntimeError(f'{key}.resolution must be [width, height]')

        topic = _override_topic_suffix(str(config.get('rostopic', '')), topic_suffix_override)
        name = _camera_name_from_topic(topic)
        sensor_name = f'UDP_GMSL_{name}'
        if rig_sensors and sensor_name not in rig_sensors:
            raise RuntimeError(f'{sensor_name} is missing from rig_calibration_file')
        sensor = rig_sensors.get(sensor_name) if rig_sensors else None
        camera_from_ego = (
            _camera_from_ego(sensor, sensor_name)
            if sensor is not None
            else camera_from_cam0 @ cam0_from_ego
        )
        cameras.append(CameraModel(
            index=index,
            name=name,
            topic=topic,
            intrinsics=intrinsics,
            distortion=distortion,
            resolution=(int(resolution[0]), int(resolution[1])),
            camera_from_ego=camera_from_ego,
        ))
    return cameras


def select_cameras(cameras: list[CameraModel], selector: str) -> list[CameraModel]:
    """Select one named camera or every camera with ``ALL``."""
    selector = selector.upper()
    selected = cameras if selector == 'ALL' else [c for c in cameras if c.name == selector]
    if selected:
        return selected
    available = ', '.join(camera.name for camera in cameras)
    raise RuntimeError(f'unknown camera_name {selector!r}; choose one of: {available}, ALL')


def _camera_keys(camchain: dict) -> list[str]:
    keys = sorted((key for key in camchain if key.startswith('cam')), key=lambda key: int(key[3:]))
    if not keys or keys[0] != 'cam0':
        raise RuntimeError('camchain must contain cam0')
    return keys


def _load_rig_sensors(rig_calibration_file: str) -> dict:
    if not rig_calibration_file:
        return {}
    path = Path(rig_calibration_file).expanduser()
    with path.open(encoding='utf-8-sig') as stream:
        data = json.load(stream)
    sensors = data.get('sensor')
    if not isinstance(sensors, dict):
        raise RuntimeError(f'{path} does not contain a sensor mapping')
    return sensors


def _camera_from_ego(sensor: dict, sensor_name: str) -> np.ndarray:
    try:
        extrinsic = sensor['extrinsic']
        ego_from_camera_rotation = np.asarray(extrinsic['rotation'], dtype=np.float64)
        ego_from_camera_translation = np.asarray(extrinsic['translation'], dtype=np.float64)
    except KeyError as error:
        raise RuntimeError(f'{sensor_name} has no complete extrinsic') from error
    if ego_from_camera_rotation.shape != (3, 3) or ego_from_camera_translation.shape != (3,):
        raise RuntimeError(f'{sensor_name} extrinsic must contain a 3x3 rotation and three-vector translation')
    camera_from_ego = np.eye(4, dtype=np.float64)
    camera_from_ego[:3, :3] = ego_from_camera_rotation.T
    camera_from_ego[:3, 3] = -ego_from_camera_rotation.T @ ego_from_camera_translation
    return camera_from_ego


def _override_topic_suffix(topic: str, suffix: str) -> str:
    if not suffix:
        return topic
    if not suffix.startswith('/image_'):
        raise RuntimeError('topic_suffix_override must start with /image_, for example /image_raw/compressed')
    if '/image_' not in topic:
        raise RuntimeError(f'cannot replace image suffix in topic {topic!r}')
    return topic.split('/image_', maxsplit=1)[0] + suffix


def _camera_name_from_topic(topic: str) -> str:
    marker = '/UDP_GMSL_'
    if marker not in topic:
        raise RuntimeError(f'cannot determine camera name from topic {topic!r}')
    return topic.split(marker, maxsplit=1)[1].split('/', maxsplit=1)[0]
