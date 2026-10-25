import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import cv2
import time

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/kitti/camera/image_raw', self.image_callback, 10)
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/kitti/camera/detections', self.detection_callback, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/kitti/lidar/points', self.pointcloud_callback, 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Storage for latest messages with timestamps
        self.latest_image = None
        self.latest_image_time = 0
        self.latest_detections = None
        self.latest_detections_time = 0
        self.latest_pointcloud = None
        self.latest_pointcloud_time = 0
        
        # Track last processed time to avoid reprocessing
        self.last_processed_time = 0
        
        # KITTI Camera calibration (P2 - left color camera)
        self.P2 = np.array([
            [7.188560e+02, 0.000000e+00, 6.071928e+02, 4.538225e+01],
            [0.000000e+00, 7.188560e+02, 1.852157e+02, -1.130887e-01],
            [0.000000e+00, 0.000000e+00, 1.000000e+00, 3.779761e-03]
        ])
        
        # Velodyne to Camera transformation
        self.R0_rect = np.array([
            [0.9999239, 0.00983776, -0.00744505, 0],
            [-0.0098698, 0.9999421, -0.00427846, 0],
            [0.00740253, 0.00435161, 0.9999631, 0],
            [0, 0, 0, 1]
        ])
        
        self.Tr_velo_to_cam = np.array([
            [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
            [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
            [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
            [0, 0, 0, 1]
        ])
        
        # Timer to check and process fusion regularly
        self.timer = self.create_timer(0.05, self.check_and_process)
        
        self.get_logger().info('Fusion node initialized and waiting for data...')
        
    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_image_time = time.time()
        self.get_logger().info('Received new image', throttle_duration_sec=1.0)
        
    def detection_callback(self, msg):
        self.latest_detections = msg
        self.latest_detections_time = time.time()
        self.get_logger().info(f'Received {len(msg.detections)} detections', throttle_duration_sec=1.0)
        
    def pointcloud_callback(self, msg):
        # Convert PointCloud2 to numpy array
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            points.append([point[0], point[1], point[2], point[3]])
        self.latest_pointcloud = np.array(points)
        self.latest_pointcloud_time = time.time()
        self.get_logger().info(f'Received point cloud with {len(points)} points', throttle_duration_sec=1.0)
        
    def check_and_process(self):
        """Check if we have fresh data and process fusion"""
        # Check if we have all data
        if self.latest_image is None or self.latest_pointcloud is None:
            return
        
        # Check if data is fresh (within last 1 second)
        current_time = time.time()
        max_age = 1.0  # seconds
        
        image_age = current_time - self.latest_image_time
        pc_age = current_time - self.latest_pointcloud_time
        det_age = current_time - self.latest_detections_time if self.latest_detections else float('inf')
        
        # Only process if we have relatively fresh data
        if image_age < max_age and pc_age < max_age:
            # Get the most recent timestamp
            newest_time = max(self.latest_image_time, self.latest_pointcloud_time, 
                            self.latest_detections_time if self.latest_detections else 0)
            
            # Only process if this is new data
            if newest_time > self.last_processed_time:
                self.process_fusion()
                self.last_processed_time = newest_time
        
    def project_lidar_to_image(self, points_3d):
        """Project 3D LiDAR points to 2D image plane"""
        if points_3d.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert to homogeneous coordinates
        points_3d_hom = np.hstack([points_3d[:, :3], np.ones((points_3d.shape[0], 1))])
        
        # Transform from velodyne to camera coordinates
        points_cam = (self.Tr_velo_to_cam @ points_3d_hom.T).T
        points_cam = (self.R0_rect @ points_cam.T).T
        
        # Filter points behind camera
        mask = points_cam[:, 2] > 0
        points_cam = points_cam[mask]
        
        if points_cam.shape[0] == 0:
            return np.array([]), np.array([]), mask
        
        # Project to image plane
        points_2d_hom = (self.P2 @ points_cam.T).T
        points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]
        
        return points_2d, points_cam[:, 2], mask
        
    def process_fusion(self):
        """Fuse camera detections with LiDAR point cloud"""
        if self.latest_image is None or self.latest_pointcloud is None:
            return
        
        # Project LiDAR points to image
        points_2d, depths, mask = self.project_lidar_to_image(self.latest_pointcloud)
        
        # Create visualization
        fusion_img = self.latest_image.copy()
        
        if len(points_2d) > 0:
            # Draw LiDAR points
            for pt, depth in zip(points_2d, depths):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < fusion_img.shape[1] and 0 <= y < fusion_img.shape[0]:
                    # Color based on depth (green=close, red=far)
                    color = int(255 * (1 - min(depth / 50.0, 1.0)))
                    cv2.circle(fusion_img, (x, y), 1, (0, color, 255-color), -1)
        
        # Draw detections if available
        if self.latest_detections is not None and len(self.latest_detections.detections) > 0:
            for detection in self.latest_detections.detections:
                cx = int(detection.bbox.center.position.x)
                cy = int(detection.bbox.center.position.y)
                w = int(detection.bbox.size_x)
                h = int(detection.bbox.size_y)
                
                x1, y1 = cx - w//2, cy - h//2
                x2, y2 = cx + w//2, cy + h//2
                
                # Draw bounding box
                cv2.rectangle(fusion_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Find LiDAR points inside bounding box
                points_in_box = []
                if len(points_2d) > 0:
                    for pt, depth in zip(points_2d, depths):
                        if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
                            points_in_box.append(depth)
                
                # Calculate average depth
                if points_in_box:
                    avg_depth = np.mean(points_in_box)
                    min_depth = np.min(points_in_box)
                    label = f'Depth: {avg_depth:.2f}m (min: {min_depth:.2f}m)'
                    cv2.putText(fusion_img, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add frame info
        cv2.putText(fusion_img, 'Camera-LiDAR Fusion Result', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(fusion_img, f'Points: {len(points_2d)} | Detections: {len(self.latest_detections.detections) if self.latest_detections else 0}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Fusion Result', fusion_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
