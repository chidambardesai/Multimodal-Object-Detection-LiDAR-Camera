#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import os

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')
        
        # Publisher
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/kitti/lidar/points', 10)
        
        # Dataset path
        self.velodyne_dir = '/home/lenovo/ros2_ws/datasets/kitti/data_object_velodyne.1/training/velodyne/'
        
        # Verify path exists
        if not os.path.exists(self.velodyne_dir):
            self.get_logger().error(f'Velodyne directory not found: {self.velodyne_dir}')
            return
        
        # Get sorted list of velodyne files
        self.velodyne_files = sorted([f for f in os.listdir(self.velodyne_dir) if f.endswith('.bin')])
        self.get_logger().info(f'Found {len(self.velodyne_files)} velodyne files')
        
        if len(self.velodyne_files) == 0:
            self.get_logger().error('No velodyne files found!')
            return
        
        # Current frame index
        self.frame_idx = 0
        
        # Manual control - slower update rate, synchronized with camera
        self.get_logger().info('MANUAL MODE: Waiting for camera node to control playback')
        
        # Timer to check and republish current frame
        self.timer = self.create_timer(0.05, self.timer_callback)
        
        # Load first frame
        self.load_frame()
        
    def load_frame(self):
        """Load current frame"""
        if self.frame_idx >= len(self.velodyne_files):
            self.frame_idx = 0
        
        # Load velodyne binary file
        velodyne_path = os.path.join(self.velodyne_dir, self.velodyne_files[self.frame_idx])
        self.current_points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        
        self.get_logger().info(f'Loaded frame {self.frame_idx}: {self.velodyne_files[self.frame_idx]} - {self.current_points.shape[0]} points')
        
    def timer_callback(self):
        """Continuously publish current frame"""
        if not hasattr(self, 'current_points'):
            return
            
        # Create PointCloud2 message
        header = self.create_header()
        pointcloud_msg = self.create_pointcloud2(self.current_points, header)
        
        self.pointcloud_pub.publish(pointcloud_msg)
    
    def create_header(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'velodyne'
        return header
    
    def create_pointcloud2(self, points, header):
        # Create PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Convert numpy array to PointCloud2
        pointcloud_msg = pc2.create_cloud(header, fields, points)
        
        return pointcloud_msg
    
    def next_frame(self):
        """Move to next frame"""
        self.frame_idx += 1
        self.load_frame()
    
    def prev_frame(self):
        """Move to previous frame"""
        self.frame_idx -= 1
        if self.frame_idx < 0:
            self.frame_idx = len(self.velodyne_files) - 1
        self.load_frame()

def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
