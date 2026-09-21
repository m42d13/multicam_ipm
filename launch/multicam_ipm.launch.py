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
        DeclareLaunchArgument(
            'camera_name', default_value='FM',
            description='One physical camera to project, normally FM or BM',
        ),
        DeclareLaunchArgument(
            'topic_suffix', default_value='/image_throttled/compressed',
            description='Image topic suffix after /UDP_GMSL_<camera>',
        ),
        Node(
            package='multicam_ipm',
            executable='ipm_node',
            name='multicam_ipm',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'camera_name': LaunchConfiguration('camera_name'),
                    'topic_suffix_override': LaunchConfiguration('topic_suffix'),
                },
            ],
        )
    ])
