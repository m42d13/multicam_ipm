"""Fast ROS-free checks for IPM geometry helpers."""

import unittest

import numpy as np

from multicam_ipm.models import CameraModel, GridSpec
from multicam_ipm.projection import BirdseyeProjector


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        # Optical camera at (0, 0, 1), looking horizontally along ego +X.
        camera_from_ego = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.camera = CameraModel(
            0, 'FM', '/UDP_GMSL_FM/image_raw/compressed',
            np.array([20.0, 20.0, 50.0, 50.0]), np.zeros(4),
            (100, 100), camera_from_ego,
        )

    def test_grid_shape_and_valid_forward_ground(self):
        projector = BirdseyeProjector([self.camera], GridSpec(1.0, 3.0, -1.0, 1.0, 1.0), False)
        self.assertEqual(projector.shape, (2, 2))
        self.assertTrue(np.all(projector.maps[0].weight > 0.0))

    def test_vehicle_mask_requires_ego_aligned_grid(self):
        projector = BirdseyeProjector([self.camera], GridSpec(1.0, 3.0, -1.0, 1.0, 1.0), True)
        with self.assertRaises(ValueError):
            projector.vehicle_mask(4.0, 2.0, 0.0)


if __name__ == '__main__':
    unittest.main()
