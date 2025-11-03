from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_name = 'go1_mujoco'
    pkg_share = get_package_share_directory(pkg_name)
    config = os.path.join(pkg_share, 'config', 'ros2_control.yaml')

    return LaunchDescription([
        # Start the ROS 2 control node (this is the controller manager)
        Node(
            package='controller_manager',
            executable='ros2_control_node',       
            parameters=[config],
            output='screen'
        ),

        # Wait a moment before spawning controllers
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
            output='screen'
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['go2_position_controller', '--controller-manager', '/controller_manager'],
            output='screen'
        ),
    ])
