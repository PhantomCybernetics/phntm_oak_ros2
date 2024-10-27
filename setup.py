from setuptools import find_packages, setup

package_name = 'phntm_oak_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'tracker_test = phntm_oak_ros2.oak_tracker_test:main',
            'detector_test = phntm_oak_ros2.oak_detector_test:main'
        ],
    },
)