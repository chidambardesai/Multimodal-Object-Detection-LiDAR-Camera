from setuptools import setup
import os
from glob import glob

package_name = 'kitti_fusion'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lenovo',
    maintainer_email='your_email@example.com',
    description='Multi-modal object detection using LiDAR and Camera',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'multimodal_node = kitti_fusion.multimodal_fusion_node:main',
            'realtime_node = kitti_fusion.realtime_detection_node:main',
            'webcam_publisher = kitti_fusion.webcam_publisher:main',
            'kitti_simulator = kitti_fusion.kitti_sensor_simulator:main',
            'fusion_node = kitti_fusion.realtime_lidar_camera_fusion:main',
            'realtime_velodyne = kitti_fusion.realtime_velodyne_fusion:main',
            'calibration_tool = kitti_fusion.calibration_tool:main'
        ],
    },
)
