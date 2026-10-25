#!/usr/bin/env python3
"""
Interactive LiDAR-Camera Calibration Tool
Use keyboard to adjust transformation matrix in real-time
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import cv2
import numpy as np
import message_filters

class CalibrationTool(Node):
    def __init__(self):
        super().__init__('calibration_tool')
        
        self.bridge = CvBridge()
        
        # Camera intrinsics (adjust if you have camera_info)
        self.fx = 700.0
        self.fy = 700.0
        self.cx = 640.0
        self.cy = 360.0
        
        # Rotation angles (degrees) - ADJUSTABLE
        self.roll = 0.0   # Rotation around X axis
        self.pitch = 0.0  # Rotation around Y axis
        self.yaw = 0.0    # Rotation around Z axis
        
        # Translation (meters) - ADJUSTABLE
        self.tx = 0.0
        self.ty = 0.0
        self.tz = 0.0
        
        # Adjustment step sizes
        self.angle_step = 5.0  # degrees
        self.trans_step = 0.05  # meters
        
        self.update_transformation()
        
        # Subscribers
        self.image_sub = message_filters.Subscriber(self, Image, '/camera')
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, '/velodyne_points')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.callback)
        
        self.get_logger().info('=' * 70)
        self.get_logger().info('🎯 LIDAR-CAMERA CALIBRATION TOOL')
        self.get_logger().info('=' * 70)
        self.get_logger().info('KEYBOARD CONTROLS:')
        self.get_logger().info('  Rotation (Roll):  A/Z  (around X-axis)')
        self.get_logger().info('  Rotation (Pitch): S/X  (around Y-axis)')
        self.get_logger().info('  Rotation (Yaw):   D/C  (around Z-axis)')
        self.get_logger().info('  Translation X:    I/K  (forward/backward)')
        self.get_logger().info('  Translation Y:    J/L  (left/right)')
        self.get_logger().info('  Translation Z:    U/O  (up/down)')
        self.get_logger().info('  Speed: +/- (increase/decrease step size)')
        self.get_logger().info('  Print values: P')
        self.get_logger().info('  Reset: R')
        self.get_logger().info('  Quit: Q')
        self.get_logger().info('=' * 70)
    
    def update_transformation(self):
        """Update transformation matrix from Euler angles"""
        # Convert degrees to radians
        roll_rad = np.radians(self.roll)
        pitch_rad = np.radians(self.pitch)
        yaw_rad = np.radians(self.yaw)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])
        
        Ry = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        
        Rz = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: Rz * Ry * Rx
        self.R = Rz @ Ry @ Rx
        
        # Translation vector
        self.t = np.array([self.tx, self.ty, self.tz])
        
        # 4x4 transformation matrix
        self.T = np.eye(4, dtype=np.float32)
        self.T[0:3, 0:3] = self.R
        self.T[0:3, 3] = self.t
    
    def callback(self, image_msg, lidar_msg):
        """Process and visualize"""
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            points = self.parse_pointcloud(lidar_msg)
            
            # Filter points
            points = points[(points[:, 0] > 0.5) & (points[:, 0] < 30)]
            
            # Project to image
            if len(points) > 0:
                vis = self.project_and_visualize(image, points)
            else:
                vis = image.copy()
            
            # Add info overlay
            self.draw_info(vis)
            
            cv2.imshow('Calibration Tool', vis)
            key = cv2.waitKey(1) & 0xFF
            self.handle_key(key)
            
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
    
    def parse_pointcloud(self, msg):
        """Parse point cloud"""
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])
        return np.array(points, dtype=np.float32) if points else np.array([]).reshape(0, 3)
    
    def project_and_visualize(self, image, points_3d):
        """Project LiDAR points onto image"""
        vis = image.copy()
        
        # Transform to camera frame
        points_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        points_cam = (self.T @ points_hom.T).T[:, :3]
        
        # Filter points in front of camera
        mask = points_cam[:, 2] > 0.1
        points_cam = points_cam[mask]
        
        if len(points_cam) == 0:
            return vis
        
        # Project to image
        points_2d = np.zeros((points_cam.shape[0], 2))
        points_2d[:, 0] = (points_cam[:, 0] * self.fx / points_cam[:, 2]) + self.cx
        points_2d[:, 1] = (points_cam[:, 1] * self.fy / points_cam[:, 2]) + self.cy
        depths = points_cam[:, 2]
        
        # Draw points
        h, w = image.shape[:2]
        for pt, depth in zip(points_2d, depths):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                # Color by depth
                color = self.depth_to_color(depth)
                cv2.circle(vis, (x, y), 2, color, -1)
        
        return vis
    
    def depth_to_color(self, depth):
        """Map depth to color"""
        normalized = np.clip(depth / 20.0, 0, 1)
        if normalized < 0.33:
            return (0, 255, 0)  # Green - close
        elif normalized < 0.66:
            return (0, 255, 255)  # Yellow - medium
        else:
            return (0, 0, 255)  # Red - far
    
    def draw_info(self, image):
        """Draw calibration info on image"""
        y = 30
        dy = 25
        
        cv2.rectangle(image, (10, 10), (400, 250), (0, 0, 0), -1)
        
        cv2.putText(image, 'CALIBRATION MODE', (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += dy
        
        cv2.putText(image, f'Roll:  {self.roll:+7.1f} deg  (A/Z)', (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += dy
        
        cv2.putText(image, f'Pitch: {self.pitch:+7.1f} deg  (S/X)', (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += dy
        
        cv2.putText(image, f'Yaw:   {self.yaw:+7.1f} deg  (D/C)', (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += dy
        
        cv2.putText(image, f'TX: {self.tx:+7.2f} m  (I/K)', (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += dy
        
        cv2.putText(image, f'TY: {self.ty:+7.2f} m  (J/L)', (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += dy
        
        cv2.putText(image, f'TZ: {self.tz:+7.2f} m  (U/O)', (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += dy
        
        cv2.putText(image, f'Step: {self.angle_step:.1f}deg / {self.trans_step:.2f}m', (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def handle_key(self, key):
        """Handle keyboard input"""
        changed = False
        
        # Rotation controls
        if key == ord('a'):
            self.roll += self.angle_step
            changed = True
        elif key == ord('z'):
            self.roll -= self.angle_step
            changed = True
        elif key == ord('s'):
            self.pitch += self.angle_step
            changed = True
        elif key == ord('x'):
            self.pitch -= self.angle_step
            changed = True
        elif key == ord('d'):
            self.yaw += self.angle_step
            changed = True
        elif key == ord('c'):
            self.yaw -= self.angle_step
            changed = True
        
        # Translation controls
        elif key == ord('i'):
            self.tx += self.trans_step
            changed = True
        elif key == ord('k'):
            self.tx -= self.trans_step
            changed = True
        elif key == ord('j'):
            self.ty += self.trans_step
            changed = True
        elif key == ord('l'):
            self.ty -= self.trans_step
            changed = True
        elif key == ord('u'):
            self.tz += self.trans_step
            changed = True
        elif key == ord('o'):
            self.tz -= self.trans_step
            changed = True
        
        # Step size controls
        elif key == ord('+') or key == ord('='):
            self.angle_step = min(self.angle_step * 1.5, 45.0)
            self.trans_step = min(self.trans_step * 1.5, 1.0)
            self.get_logger().info(f'Step size: {self.angle_step:.1f}° / {self.trans_step:.2f}m')
        elif key == ord('-'):
            self.angle_step = max(self.angle_step / 1.5, 0.5)
            self.trans_step = max(self.trans_step / 1.5, 0.01)
            self.get_logger().info(f'Step size: {self.angle_step:.1f}° / {self.trans_step:.2f}m')
        
        # Print current values
        elif key == ord('p'):
            self.print_calibration()
        
        # Reset
        elif key == ord('r'):
            self.roll = self.pitch = self.yaw = 0.0
            self.tx = self.ty = self.tz = 0.0
            changed = True
            self.get_logger().info('Reset to zero')
        
        # Quit
        elif key == ord('q'):
            self.print_calibration()
            cv2.destroyAllWindows()
            rclpy.shutdown()
        
        if changed:
            self.update_transformation()
    
    def print_calibration(self):
        """Print current calibration values"""
        print('\n' + '='*70)
        print('📋 CURRENT CALIBRATION VALUES')
        print('='*70)
        print(f'Roll:  {self.roll:+8.2f}°')
        print(f'Pitch: {self.pitch:+8.2f}°')
        print(f'Yaw:   {self.yaw:+8.2f}°')
        print(f'TX:    {self.tx:+8.3f} m')
        print(f'TY:    {self.ty:+8.3f} m')
        print(f'TZ:    {self.tz:+8.3f} m')
        print('='*70)
        print('\nCopy this to your fusion code:')
        print('\n# Euler angles (degrees)')
        print(f'roll, pitch, yaw = {self.roll}, {self.pitch}, {self.yaw}')
        print(f't = np.array([{self.tx}, {self.ty}, {self.tz}])')
        print('\n# OR use the rotation matrix directly:')
        print('R = np.array([')
        for row in self.R:
            print(f'    [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}],')
        print('])')
        print(f't = np.array([{self.tx}, {self.ty}, {self.tz}])')
        print('='*70 + '\n')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = CalibrationTool()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
