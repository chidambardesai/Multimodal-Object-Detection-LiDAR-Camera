#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import message_filters

class RealtimeLidarCameraFusion(Node):
    """
    Real-time multi-modal fusion processing LIVE sensor data
    Subscribes to camera and LiDAR topics (like real sensors)
    """
    def __init__(self):
        super().__init__('realtime_fusion_node')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.get_logger().info('Loading YOLOv8 model...')
        self.model = YOLO('yolov8n.pt')
        self.class_names = self.model.names
        self.get_logger().info('YOLOv8 loaded!')
        
        # KITTI Calibration matrices
        self.P2 = np.array([
            [7.188560e+02, 0.000000e+00, 6.071928e+02, 4.538225e+01],
            [0.000000e+00, 7.188560e+02, 1.852157e+02, -1.130887e-01],
            [0.000000e+00, 0.000000e+00, 1.000000e+00, 3.779761e-03]
        ])
        
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
        
        # Synchronized subscribers using message_filters
        self.image_sub = message_filters.Subscriber(self, Image, '/sensor/camera/image_raw')
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, '/sensor/lidar/points')
        
        # Synchronize camera and LiDAR data (max 0.05s time difference)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.fusion_callback)
        
        self.frame_count = 0
        
        self.get_logger().info('=' * 70)
        self.get_logger().info('🚀 REAL-TIME LIDAR-CAMERA FUSION NODE')
        self.get_logger().info('=' * 70)
        self.get_logger().info('📷 Subscribed to: /sensor/camera/image_raw')
        self.get_logger().info('📡 Subscribed to: /sensor/lidar/points')
        self.get_logger().info('⚡ Processing multi-modal data in real-time!')
        self.get_logger().info('=' * 70)
        self.get_logger().info('Press Q in window to quit')
        self.get_logger().info('=' * 70)
    
    def fusion_callback(self, image_msg, lidar_msg):
        """Process synchronized camera and LiDAR data"""
        try:
            self.frame_count += 1
            
            # Convert ROS messages to usable data
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            points_3d = self.parse_pointcloud(lidar_msg)
            
            # Log
            if self.frame_count % 10 == 0:
                self.get_logger().info(
                    f'🔄 Processing frame {self.frame_count} | '
                    f'Image: {image.shape} | LiDAR: {len(points_3d)} points'
                )
            
            # STEP 1: 2D Object Detection
            results = self.model(image, verbose=False)
            detections_2d = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    detections_2d.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class_id': cls_id,
                        'class_name': self.class_names[cls_id],
                        'confidence': confidence
                    })
            
            # STEP 2: Project LiDAR to Image
            points_2d, depths, mask = self.project_lidar_to_image(points_3d)
            
            # STEP 3: Paint Scores on LiDAR
            enriched_points = self.paint_scores_on_lidar(points_3d, points_2d, depths, mask, detections_2d)
            
            # STEP 4: 3D Object Detection
            detections_3d = self.detect_3d_objects(enriched_points, points_3d)
            
            # STEP 5: Visualize
            self.visualize_fusion(image, points_3d, detections_2d, enriched_points, detections_3d)
            
        except Exception as e:
            self.get_logger().error(f'Error in fusion callback: {e}')
    
    def parse_pointcloud(self, msg):
        """Convert PointCloud2 message to numpy array"""
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            points.append([point[0], point[1], point[2], point[3]])
        return np.array(points, dtype=np.float32)
    
    def project_lidar_to_image(self, points_3d):
        """Project 3D LiDAR points to 2D image plane"""
        if points_3d.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])
        
        points_3d_hom = np.hstack([points_3d[:, :3], np.ones((points_3d.shape[0], 1))])
        points_cam = (self.Tr_velo_to_cam @ points_3d_hom.T).T
        points_cam = (self.R0_rect @ points_cam.T).T
        
        mask = points_cam[:, 2] > 0
        points_cam = points_cam[mask]
        
        if points_cam.shape[0] == 0:
            return np.array([]), np.array([]), mask
        
        points_2d_hom = (self.P2 @ points_cam.T).T
        points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]
        
        return points_2d, points_cam[:, 2], mask
    
    def paint_scores_on_lidar(self, points_3d, points_2d, depths, mask, detections_2d):
        """Paint 2D detection scores onto LiDAR points"""
        valid_points = points_3d[mask]
        enriched = np.zeros((valid_points.shape[0], 6))
        enriched[:, :4] = valid_points
        enriched[:, 4] = -1
        enriched[:, 5] = 0.0
        
        for det in detections_2d:
            x1, y1, x2, y2 = det['bbox']
            inside_mask = (
                (points_2d[:, 0] >= x1) & (points_2d[:, 0] <= x2) &
                (points_2d[:, 1] >= y1) & (points_2d[:, 1] <= y2)
            )
            enriched[inside_mask, 4] = det['class_id']
            enriched[inside_mask, 5] = det['confidence']
        
        labeled_mask = enriched[:, 4] >= 0
        return enriched[labeled_mask]
    
    def detect_3d_objects(self, enriched_points, all_points):
        """3D object detection using clustered enriched points"""
        if len(enriched_points) == 0:
            return []
        
        detections_3d = []
        unique_classes = np.unique(enriched_points[:, 4])
        
        for class_id in unique_classes:
            if class_id < 0:
                continue
            
            class_mask = enriched_points[:, 4] == class_id
            class_points = enriched_points[class_mask]
            
            if len(class_points) < 10:
                continue
            
            clustering = DBSCAN(eps=1.5, min_samples=10).fit(class_points[:, :3])
            labels = clustering.labels_
            
            for label in set(labels):
                if label == -1:
                    continue
                
                cluster_mask = labels == label
                cluster_points = class_points[cluster_mask]
                
                if len(cluster_points) < 10:
                    continue
                
                min_coords = np.min(cluster_points[:, :3], axis=0)
                max_coords = np.max(cluster_points[:, :3], axis=0)
                
                center = (min_coords + max_coords) / 2
                size = max_coords - min_coords
                avg_confidence = np.mean(cluster_points[:, 5])
                
                detections_3d.append({
                    'center': center,
                    'size': size,
                    'class_id': int(class_id),
                    'class_name': self.class_names[int(class_id)],
                    'confidence': avg_confidence,
                    'num_points': len(cluster_points)
                })
        
        return detections_3d
    
    def visualize_fusion(self, image, points_3d, detections_2d, enriched_points, detections_3d):
        """Create 2x2 visualization grid"""
        h, w = image.shape[:2]
        
        # View 1: Raw camera
        view1 = image.copy()
        cv2.putText(view1, f'Camera Feed (Frame {self.frame_count})', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # View 2: LiDAR point cloud (colored by depth)
        view2 = self.create_lidar_view(image, points_3d)
        cv2.putText(view2, f'LiDAR ({len(points_3d)} points)', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # View 3: 2D detections
        view3 = image.copy()
        for det in detections_2d:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(view3, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(view3, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(view3, f'2D Detection ({len(detections_2d)} objects)', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # View 4: 3D detections
        view4 = self.create_3d_view(image, enriched_points, detections_3d)
        cv2.putText(view4, f'3D Detection ({len(detections_3d)} 3D boxes)', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Create grid
        scale = 0.75
        new_w, new_h = int(w * scale), int(h * scale)
        
        view1 = cv2.resize(view1, (new_w, new_h))
        view2 = cv2.resize(view2, (new_w, new_h))
        view3 = cv2.resize(view3, (new_w, new_h))
        view4 = cv2.resize(view4, (new_w, new_h))
        
        top_row = np.hstack([view1, view2])
        bottom_row = np.hstack([view3, view4])
        combined = np.vstack([top_row, bottom_row])
        
        combined = cv2.copyMakeBorder(combined, 60, 40, 10, 10,
                                      cv2.BORDER_CONSTANT, value=(40, 40, 40))
        
        title_x = combined.shape[1] // 2 - 350
        cv2.putText(combined, 'Real-Time LiDAR-Camera Multi-Modal Fusion',
                   (title_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.putText(combined, 'Press Q to Quit',
                   (combined.shape[1] - 200, combined.shape[0]-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Real-Time Multi-Modal Fusion', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quitting...')
            cv2.destroyAllWindows()
            rclpy.shutdown()
    
    def create_lidar_view(self, image, points_3d):
        """Visualize LiDAR points colored by depth"""
        h, w = image.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(points_3d) == 0:
            return vis
        
        points_2d, depths, mask = self.project_lidar_to_image(points_3d)
        
        if len(points_2d) == 0:
            return vis
        
        depths_valid = depths[depths > 0]
        if len(depths_valid) == 0:
            return vis
        
        depth_min, depth_max = np.min(depths_valid), np.max(depths_valid)
        
        for pt, depth in zip(points_2d, depths):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h and depth > 0:
                normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)
                
                if normalized < 0.25:
                    color = (0, int(255 * normalized * 4), 255)
                elif normalized < 0.5:
                    color = (0, 255, int(255 * (1 - (normalized - 0.25) * 4)))
                elif normalized < 0.75:
                    color = (int(255 * (normalized - 0.5) * 4), 255, 0)
                else:
                    color = (255, int(255 * (1 - (normalized - 0.75) * 4)), 0)
                
                cv2.circle(vis, (x, y), 2, color, -1)
        
        return vis
    
    def create_3d_view(self, image, enriched_points, detections_3d):
        """Visualize 3D detections"""
        vis = image.copy()
        
        # Draw enriched points
        if len(enriched_points) > 0:
            points_2d, _, _ = self.project_lidar_to_image(enriched_points)
            
            for pt, cls_id in zip(points_2d, enriched_points[:, 4]):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                    if cls_id == 2:  # car
                        color = (0, 255, 0)
                    elif cls_id == 0:  # person
                        color = (255, 0, 0)
                    else:
                        color = (255, 255, 0)
                    cv2.circle(vis, (x, y), 2, color, -1)
        
        # Draw 3D boxes
        for det_3d in detections_3d:
            center = det_3d['center']
            size = det_3d['size']
            
            corners = self.get_3d_box_corners(center, size)
            corners_2d = self.project_3d_box_to_image(corners)
            
            if corners_2d is not None:
                self.draw_3d_box(vis, corners_2d, (0, 0, 255))
                
                label = f"{det_3d['class_name']}: {det_3d['confidence']:.2f}"
                cx, cy = int(np.mean(corners_2d[:, 0])), int(np.min(corners_2d[:, 1]))
                if cy > 0:
                    cv2.putText(vis, label, (cx, cy-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return vis
    
    def get_3d_box_corners(self, center, size):
        """Get 8 corners of 3D box"""
        x, y, z = center
        l, w, h = size
        
        corners = np.array([
            [x-l/2, y-w/2, z-h/2], [x+l/2, y-w/2, z-h/2],
            [x+l/2, y+w/2, z-h/2], [x-l/2, y+w/2, z-h/2],
            [x-l/2, y-w/2, z+h/2], [x+l/2, y-w/2, z+h/2],
            [x+l/2, y+w/2, z+h/2], [x-l/2, y+w/2, z+h/2],
        ])
        return corners
    
    def project_3d_box_to_image(self, corners_3d):
        """Project 3D box to image"""
        corners_hom = np.hstack([corners_3d, np.ones((8, 1))])
        corners_cam = (self.Tr_velo_to_cam @ corners_hom.T).T
        corners_cam = (self.R0_rect @ corners_cam.T).T
        
        if np.any(corners_cam[:, 2] <= 0):
            return None
        
        corners_2d_hom = (self.P2 @ corners_cam.T).T
        corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]
        
        return corners_2d
    
    def draw_3d_box(self, img, corners_2d, color):
        """Draw 3D bounding box"""
        corners = corners_2d.astype(int)
        
        for i in range(4):
            cv2.line(img, tuple(corners[i]), tuple(corners[(i+1)%4]), color, 2)
        
        for i in range(4, 8):
            cv2.line(img, tuple(corners[i]), tuple(corners[4+(i+1)%4]), color, 2)
        
        for i in range(4):
            cv2.line(img, tuple(corners[i]), tuple(corners[i+4]), color, 2)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RealtimeLidarCameraFusion()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
