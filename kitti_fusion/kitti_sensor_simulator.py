#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import cv2
import os
import numpy as np
import time

class KITTISensorSimulator(Node):
    """
    Simulates real LiDAR + Camera sensors by publishing KITTI data
    at real-time rate (10 Hz) on ROS2 topics
    """
    def __init__(self):
        super().__init__('kitti_sensor_simulator')
        
        # Publishers - Like real sensors!
        self.image_pub = self.create_publisher(Image, '/sensor/camera/image_raw', 10)
        self.lidar_pub = self.create_publisher(PointCloud2, '/sensor/lidar/points', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Dataset paths
        self.image_dir = '/home/lenovo/ros2_ws/datasets/kitti/training/image_2/'
        self.velodyne_dir = '/home/lenovo/ros2_ws/datasets/kitti/data_object_velodyne.1/training/velodyne/'
        
        # Verify paths
        if not os.path.exists(self.image_dir) or not os.path.exists(self.velodyne_dir):
            self.get_logger().error('Dataset directories not found!')
            raise RuntimeError('KITTI dataset not found')
        
        # Get files
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.velodyne_files = sorted([f for f in os.listdir(self.velodyne_dir) if f.endswith('.bin')])
        self.num_frames = min(len(self.image_files), len(self.velodyne_files))
        
        self.frame_idx = 0
        
        self.get_logger().info('=' * 70)
        self.get_logger().info('🚗 KITTI SENSOR SIMULATOR')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'📊 Loaded {self.num_frames} synchronized frames')
        self.get_logger().info(f'📷 Camera topic: /sensor/camera/image_raw')
        self.get_logger().info(f'📡 LiDAR topic: /sensor/lidar/points')
        self.get_logger().info(f'⚡ Publishing at 10 Hz (like real sensors)')
        self.get_logger().info('=' * 70)
        
        # Timer at 10 Hz (like real KITTI data rate)
        self.timer = self.create_timer(0.1, self.publish_sensor_data)
        
    def publish_sensor_data(self):
        """Publish synchronized camera and LiDAR data"""
        try:
            # Load camera image
            image_path = os.path.join(self.image_dir, self.image_files[self.frame_idx])
            image = cv2.imread(image_path)
            
            if image is None:
                self.get_logger().error(f'Failed to load image {image_path}')
                return
            
            # Load LiDAR point cloud
            velodyne_path = os.path.join(self.velodyne_dir, self.velodyne_files[self.frame_idx])
            points_3d = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
            
            # Create synchronized timestamp
            timestamp = self.get_clock().now().to_msg()
            
            # Publish Camera Image
            img_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            img_msg.header.stamp = timestamp
            img_msg.header.frame_id = 'camera'
            self.image_pub.publish(img_msg)
            
            # Publish LiDAR Point Cloud
            header = Header()
            header.stamp = timestamp
            header.frame_id = 'velodyne'
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            
            pc_msg = pc2.create_cloud(header, fields, points_3d)
            self.lidar_pub.publish(pc_msg)
            
            # Log periodically
            if self.frame_idx % 10 == 0:
                self.get_logger().info(
                    f'📡 Frame {self.frame_idx}/{self.num_frames} | '
                    f'Image: {image.shape[1]}x{image.shape[0]} | '
                    f'LiDAR: {points_3d.shape[0]} points'
                )
            
            # Advance to next frame
            self.frame_idx += 1
            if self.frame_idx >= self.num_frames:
                self.frame_idx = 0  # Loop
                self.get_logger().info('🔄 Looping back to frame 0')
            
        except Exception as e:
            self.get_logger().error(f'Error publishing sensor data: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = KITTISensorSimulator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
