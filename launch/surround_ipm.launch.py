from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description():
    config = Path(get_package_share_directory('multicam_ipm')) / 'config' / 'surround.yaml'
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=str(config),
            description='ROS parameter YAML; use a local copy with calibration paths',
        ),
        Node(
            package='multicam_ipm',
            executable='surround_ipm_node',
            name='surround_ipm',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
        )
    ])
