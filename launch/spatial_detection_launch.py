from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command, LaunchConfiguration

def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(
        '/ros2_ws/',
        'phntm_oak_ros2_params.yaml'
        )
    
    spatial_detection_node = Node(
        package='phntm_oak_ros2',
        executable='spatial_detection',
        output='screen',
        emulate_tty=True,
        parameters=[config]
        )
    
    xacro_path = os.path.join(
        '/ros2_ws',
        'src/phntm_oak_ros2/description',
        'depthai_descr.urdf.xacro'
        )

    camera_model = LaunchConfiguration("camera_model", default="OAK-D-LITE")
    tf_prefix = LaunchConfiguration("tf_prefix", default="oak")
    base_frame = LaunchConfiguration("base_frame", default="oak-d_frame")
    parent_frame = LaunchConfiguration("parent_frame", default="oak-d-base-frame")
    cam_pos_x = LaunchConfiguration("cam_pos_x", default="0.0")
    cam_pos_y = LaunchConfiguration("cam_pos_y", default="0.0")
    cam_pos_z = LaunchConfiguration("cam_pos_z", default="1.15")
    cam_roll = LaunchConfiguration("cam_roll", default="0.0")
    cam_pitch = LaunchConfiguration("cam_pitch", default="0.0")
    cam_yaw = LaunchConfiguration("cam_yaw", default="0.0")
    rs_compat = LaunchConfiguration("rs_compat", default="false")

    robot_description = {
        "robot_description": Command(
            [
                "xacro",
                " ",
                xacro_path,
                " ",
                "camera_name:=",
                tf_prefix,
                " ",
                "camera_model:=",
                camera_model,
                " ",
                "base_frame:=",
                base_frame,
                " ",
                "parent_frame:=",
                parent_frame,
                " ",
                "cam_pos_x:=",
                cam_pos_x,
                " ",
                "cam_pos_y:=",
                cam_pos_y,
                " ",
                "cam_pos_z:=",
                cam_pos_z,
                " ",
                "cam_roll:=",
                cam_roll,
                " ",
                "cam_pitch:=",
                cam_pitch,
                " ",
                "cam_yaw:=",
                cam_yaw,
                " ",
                "rs_compat:=",
                rs_compat,
            ]
        )
    }
    
    state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="oak_state_publisher",
        parameters=[robot_description]
        )
    
    return LaunchDescription([
        state_publisher_node,
        spatial_detection_node
    ])
