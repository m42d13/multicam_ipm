# multicam_ipm

ROS 2 inverse-perspective mapping for calibrated camera images.

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
  config_file:=$HOME/ros2_ws/src/multicam_ipm/config/surround.local.yaml
```

The local configuration must contain paths to your Kalibr camchain and rig
calibration JSON. It is ignored by Git.
