import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('rio_cpp'),
        'config',
        'parameters.yaml'
    )

    rio_node = Node(
        package='rio_cpp',
        executable='rio_node',
        name='state_estimator_cpp',
        output='screen',
        parameters=[config],
    )

    return LaunchDescription([
        rio_node,
    ])
