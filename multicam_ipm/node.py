"""ROS 2 wrapper for the calibration and projection modules."""

from threading import Lock
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from .calibration import load_cameras, select_cameras
from .models import GridSpec
from .projection import BirdseyeProjector


def stamp_to_nanoseconds(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class MulticamIpmNode(Node):
    """Subscribe to calibrated images and publish a flat-ground BEV image."""

    def __init__(self):
        super().__init__('multicam_ipm')
        self._declare_parameters()
        camchain_file = self.get_parameter('camchain_file').value
        if not camchain_file:
            camchain_file = str(
                Path(get_package_share_directory('multicam_ipm'))
                / 'config' / 'mduc_yaza_hybrid_camchain.yaml'
            )
            self.get_logger().info(f'using bundled calibration: {camchain_file}')
        cameras = load_cameras(
            camchain_file,
            self.get_parameter('rig_calibration_file').value,
            self.get_parameter('topic_suffix_override').value,
            self._fallback_cam0_from_ego(),
        )
        self.cameras = select_cameras(cameras, self.get_parameter('camera_name').value)
        self.projector = BirdseyeProjector(
            self.cameras, self._grid_spec(), bool(self.get_parameter('camera_aligned_output').value)
        )
        self.vehicle_mask = self._build_vehicle_mask()
        self.bridge = CvBridge()
        self.lock = Lock()
        self.latest_frames: dict[int, tuple[int, np.ndarray]] = {}
        self.last_published_stamp = -1

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.camera_subscriptions = []
        for camera in self.cameras:
            callback = lambda message, index=camera.index: self._image_callback(index, message)
            self.camera_subscriptions.append(self.create_subscription(CompressedImage, camera.topic, callback, qos))
            self.get_logger().info(f'cam{camera.index} ({camera.name}): {camera.topic}')

        self.publisher = self.create_publisher(Image, self.get_parameter('output_topic').value, 2)
        rate = float(self.get_parameter('publish_rate').value)
        if rate <= 0.0:
            raise RuntimeError('publish_rate must be positive')
        self.timer = self.create_timer(1.0 / rate, self._publish_ipm)
        height, width = self.projector.shape
        self.get_logger().info(
            f'IPM output: {width}x{height} pixels, {self._grid_spec().meters_per_pixel:.3f} m/pixel, '
            f'coverage: {self.projector.coverage:.1%}'
        )

    def _declare_parameters(self) -> None:
        defaults = {
            'camchain_file': '', 'rig_calibration_file': '', 'camera_name': 'FM',
            'output_topic': '/multicam/ipm', 'output_frame': 'base_link',
            'max_frame_span': 0.15, 'publish_rate': 5.0,
            'x_min': -3.0, 'x_max': 12.0, 'y_min': -6.0, 'y_max': 6.0,
            'meters_per_pixel': 0.02, 'camera_aligned_output': True,
            'blend_mode': 'feather', 'camera_height': 1.2,
            'camera_forward_offset': 0.0, 'camera_left_offset': 0.0,
            'camera_pitch_deg': 0.0, 'camera_roll_deg': 0.0,
            'camera_yaw_deg': 0.0, 'mask_vehicle': False,
            'vehicle_length': 4.8, 'vehicle_width': 2.0,
            'vehicle_center_x': 0.0, 'topic_suffix_override': '',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _grid_spec(self) -> GridSpec:
        return GridSpec(*(
            float(self.get_parameter(name).value)
            for name in ('x_min', 'x_max', 'y_min', 'y_max', 'meters_per_pixel')
        ))

    def _fallback_cam0_from_ego(self) -> np.ndarray:
        """Create cam0 <- ego from the documented manual fallback parameters."""
        degrees = np.pi / 180.0
        roll = float(self.get_parameter('camera_roll_deg').value) * degrees
        pitch = float(self.get_parameter('camera_pitch_deg').value) * degrees
        yaw = float(self.get_parameter('camera_yaw_deg').value) * degrees
        ego_from_camera = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        ego_from_camera = _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll) @ ego_from_camera
        position = np.array([
            float(self.get_parameter('camera_forward_offset').value),
            float(self.get_parameter('camera_left_offset').value),
            float(self.get_parameter('camera_height').value),
        ])
        camera_from_ego = np.eye(4, dtype=np.float64)
        camera_from_ego[:3, :3] = ego_from_camera.T
        camera_from_ego[:3, 3] = -ego_from_camera.T @ position
        return camera_from_ego

    def _build_vehicle_mask(self) -> np.ndarray | None:
        if not self.get_parameter('mask_vehicle').value:
            return None
        try:
            return self.projector.vehicle_mask(
                float(self.get_parameter('vehicle_length').value),
                float(self.get_parameter('vehicle_width').value),
                float(self.get_parameter('vehicle_center_x').value),
            )
        except ValueError as error:
            raise RuntimeError(str(error)) from error

    def _image_callback(self, camera_index: int, message: CompressedImage) -> None:
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(message, 'bgr8')
        except Exception as error:
            self.get_logger().error(f'cam{camera_index} decode failed: {error}')
            return
        with self.lock:
            self.latest_frames[camera_index] = (stamp_to_nanoseconds(message.header.stamp), image)

    def _publish_ipm(self) -> None:
        with self.lock:
            if len(self.latest_frames) != len(self.cameras):
                return
            frames = dict(self.latest_frames)
        timestamps = [frame[0] for frame in frames.values()]
        if len(timestamps) > 1:
            allowed_span = round(float(self.get_parameter('max_frame_span').value) * 1_000_000_000)
            if max(timestamps) - min(timestamps) > allowed_span:
                return
        output_stamp = max(timestamps)
        if output_stamp <= self.last_published_stamp:
            return
        try:
            output = self.projector.render(
                {index: frame[1] for index, frame in frames.items()}, self.get_parameter('blend_mode').value
            )
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        if self.vehicle_mask is not None:
            output[self.vehicle_mask] = 0
        message = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
        message.header.frame_id = self.get_parameter('output_frame').value
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(output_stamp, 1_000_000_000)
        self.publisher.publish(message)
        self.last_published_stamp = output_stamp


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = MulticamIpmNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
