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
    
    # declared_arguments = [
    #     DeclareLaunchArgument(
    #         "namespace",
    #         default_value="",
    #         description='Specifies the namespace of the robot state publisher node. Default value will be ""',
    #     ),
    #     DeclareLaunchArgument(
    #         "camera_model",
    #         default_value="OAK-D",
    #         description="The model of the camera. Using a wrong camera model can disable camera features. Valid models: `OAK-D, OAK-D-LITE`.",
    #     ),
    #     DeclareLaunchArgument(
    #         "tf_prefix",
    #         default_value="oak",
    #         description="The name of the camera. It can be different from the camera model and it will be used as node `namespace`.",
    #     ),
    #     DeclareLaunchArgument(
    #         "base_frame",
    #         default_value="oak-d_frame",
    #         description="Name of the base link.",
    #     ),
    #     DeclareLaunchArgument(
    #         "parent_frame",
    #         default_value="oak-d-base-frame",
    #         description="Name of the parent link from other a robot TF for example that can be connected to the base of the OAK.",
    #     ),
    #     DeclareLaunchArgument(
    #         "cam_pos_x",
    #         default_value="0.0",
    #         description="Position X of the camera with respect to the base frame.",
    #     ),
    #     DeclareLaunchArgument(
    #         "cam_pos_y",
    #         default_value="0.0",
    #         description="Position Y of the camera with respect to the base frame.",
    #     ),
    #     DeclareLaunchArgument(
    #         "cam_pos_z",
    #         default_value="0.0",
    #         description="Position Z of the camera with respect to the base frame.",
    #     ),
    #     DeclareLaunchArgument(
    #         "cam_roll",
    #         default_value="0.0",
    #         description="Roll orientation of the camera with respect to the base frame.",
    #     ),
    #     DeclareLaunchArgument(
    #         "cam_pitch",
    #         default_value="0.0",
    #         description="Pitch orientation of the camera with respect to the base frame.",
    #     ),
    #     DeclareLaunchArgument(
    #         "cam_yaw",
    #         default_value="0.0",
    #         description="Yaw orientation of the camera with respect to the base frame.",
    #     ),
    #     DeclareLaunchArgument(
    #         "use_composition",
    #         default_value="false",
    #         description="Use composition to start the robot_state_publisher node. Default value will be false",
    #     ),
    #     DeclareLaunchArgument(
    #         "use_base_descr",
    #         default_value="false",
    #         description="Launch base description. Default value will be false",
    #     ),
    #     DeclareLaunchArgument(
    #         "rs_compat",
    #         default_value="false",
    #         description="Enable RealSense compatibility mode. Default value will be false",
    #     ),
    # ]
    
    # bringup_dir = get_package_share_directory("depthai_descriptions")
    #use_base_descr = LaunchConfiguration("use_base_descr", default="false")

    bringup_dir = get_package_share_directory("phntm_oak_ros2")
    xacro_path = os.path.join(
        '/ros2_ws',
        'src/phntm_oak_ros2/description',
        'depthai_descr.urdf.xacro'
        )
    #os.path.join(bringup_dir, "urdf", "depthai_descr.urdf.xacro")

    camera_model = LaunchConfiguration("camera_model", default="OAK-D-LITE")
    tf_prefix = LaunchConfiguration("tf_prefix", default="oak")
    base_frame = LaunchConfiguration("base_frame", default="oak-d_frame")
    parent_frame = LaunchConfiguration("parent_frame", default="oak-d-base-frame")
    cam_pos_x = LaunchConfiguration("cam_pos_x", default="0.0")
    cam_pos_y = LaunchConfiguration("cam_pos_y", default="0.0")
    cam_pos_z = LaunchConfiguration("cam_pos_z", default="1.15")
    cam_roll = LaunchConfiguration("cam_roll", default="0.0")
    cam_pitch = LaunchConfiguration("cam_pitch", default="0.0")
    cam_yaw = LaunchConfiguration("cam_yaw", default="1.5708")
    # namespace = LaunchConfiguration("namespace", default="")
    rs_compat = LaunchConfiguration("rs_compat", default="false")
    # use_composition = LaunchConfiguration("use_composition", default="false")

    # name = LaunchConfiguration("tf_prefix").perform(context)
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
