# src/my_bot/launch/tb3_localization.launch.py
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    map_file_path = '/home/dev/map.yaml' 

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    map_yaml_file = LaunchConfiguration('map', default=map_file_path)


    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            # Navigation2 module for localization 
            os.path.join(nav2_bringup_dir, 'launch', 'localization_launch.py')
        ),
        launch_arguments={
            'map': map_yaml_file,
            'use_sim_time': use_sim_time,
            'params_file': os.path.join(nav2_bringup_dir, 'params', 'nav2_params.yaml') # 기본 파라미터 사용
        }.items()
    )

    navigation_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        # localization 대신 navigation_launch 또는 bringup_launch 사용
        os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map': map_yaml_file,
            'use_sim_time': use_sim_time,
            'params_file': os.path.join(nav2_bringup_dir, 'params', 'nav2_params.yaml')
        }.items()
    )

    rviz_config_dir = os.path.join(
        get_package_share_directory('nav2_bringup'), 
        'rviz', 'nav2_default_view.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    a_star_node = Node(
        package='img_pkg',
        executable='heapq',
        name='cho',
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=map_file_path,
            description='Full path to map yaml file to load'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),
        localization_launch,
        rviz_node,
        a_star_node
    ])