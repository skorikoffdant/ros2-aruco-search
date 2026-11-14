from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='aruco_search', executable='aruco_detector',
             output='screen', parameters=[{'use_sim_time': True}]),
        Node(package='aruco_search', executable='search_manager',
             output='screen', parameters=[{'use_sim_time': True}]),
    ])
