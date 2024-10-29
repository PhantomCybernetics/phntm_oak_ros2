from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'phntm_oak_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # include all launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mirek Burkon',
    maintainer_email='mirek@phntm.io',
    description='Custom Python ROS2 driver for the OAK-D Lite camera',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spatial_detection = phntm_oak_ros2.spatial_detection:main'
        ],
    },
)