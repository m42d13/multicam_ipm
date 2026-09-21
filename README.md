# multicam_ipm

ROS 2 inverse-perspective mapping for a calibrated six-camera surround view.

## Build

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select multicam_ipm --symlink-install
source install/setup.bash
```

## Run Surround View

```bash
ros2 launch multicam_ipm surround_ipm.launch.py \
  config_file:=$HOME/ros2_ws/src/multicam_ipm/config/surround.yaml
```

The package includes this rig's Kalibr intrinsics, relative transforms, and
embedded camera-to-ego poses, so it does not need an external JSON file. For a
different rig, set `camchain_file` to its calibration YAML. If that YAML has
no `T_ego_camera` matrices, provide its rig JSON through
`rig_calibration_file`.
