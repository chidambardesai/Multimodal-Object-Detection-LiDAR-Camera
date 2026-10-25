#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from vision_msgs.msg import Detection2DArray, Detection3DArray, Detection3D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import cv2
import os
from ultralytics import YOLO
import numpy as np
from sklearn.cluster import DBSCAN
import time

class MultiModalFusionNode(Node):
    def __init__(self):
        super().__init__('multimodal_fusion_node')
        
        # Publishers
        self.image_pub = self.create_publisher(Image, '/kitti/camera/image_raw', 10)
        self.detection_2d_pub = self.create_publisher(Detection2DArray, '/kitti/camera/detections_2d', 10)
        self.detection_3d_pub = self.create_publisher(Detection3DArray, '/kitti/detections_3d', 10)
        self.enriched_pc_pub = self.create_publisher(PointCloud2, '/kitti/lidar/enriched_points', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.get_logger().info('Loading YOLOv8 model...')
        self.model = YOLO('yolov8n.pt')
        self.get_logger().info('YOLOv8 model loaded!')
        
        # YOLO class names for reference
        self.class_names = self.model.names
        
        # Dataset paths
        self.image_dir = '/home/lenovo/ros2_ws/datasets/kitti/training/image_2/'
        self.velodyne_dir = '/home/lenovo/ros2_ws/datasets/kitti/training/velodyne/'
        
        # Verify paths
        if not os.path.exists(self.image_dir) or not os.path.exists(self.velodyne_dir):
            self.get_logger().error('Dataset directories not found!')
            return
        
        # Get files
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.velodyne_files = sorted([f for f in os.listdir(self.velodyne_dir) if f.endswith('.bin')])
        self.num_frames = min(len(self.image_files), len(self.velodyne_files))
        
        self.get_logger().info(f'Found {self.num_frames} synchronized frames')
        
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
        
        self.frame_idx = 0
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('MULTI-MODAL FUSION PIPELINE')
        self.get_logger().info('RGB → 2D Detector → Paint Scores → 3D Detector')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Controls: SPACE=Next | B=Back | Q=Quit')
        self.get_logger().info('=' * 60)
        
        # Storage for current frame data to republish
        self.current_image = None
        self.current_enriched_points = None
        self.current_detections_2d = None
        
        # Process first frame
        self.process_multimodal_frame()
        
        # Timer for keyboard input
        self.timer = self.create_timer(0.05, self.check_input)
        
    def check_input(self):
        """Handle keyboard input and continuously republish current frame"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            self.frame_idx += 1
            if self.frame_idx >= self.num_frames:
                self.frame_idx = 0
            self.process_multimodal_frame()
        elif key == ord('b'):
            self.frame_idx -= 1
            if self.frame_idx < 0:
                self.frame_idx = self.num_frames - 1
            self.process_multimodal_frame()
        elif key == ord('q'):
            self.get_logger().info('Quitting...')
            cv2.destroyAllWindows()
            rclpy.shutdown()
        else:
            # Continuously republish current frame data for RViz
            self.republish_current_frame()
    
    def process_multimodal_frame(self):
        """
        Main pipeline: RGB → 2D Detection → Paint LiDAR → 3D Detection
        """
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'PROCESSING FRAME {self.frame_idx + 1}/{self.num_frames}')
        self.get_logger().info('=' * 60)
        
        # STEP 1: Load RGB image
        image_path = os.path.join(self.image_dir, self.image_files[self.frame_idx])
        image = cv2.imread(image_path)
        if image is None:
            self.get_logger().error(f'Failed to load image')
            return
        
        # STEP 2: Run 2D detector on RGB image
        self.get_logger().info('STEP 1: Running 2D detector on RGB image...')
        results = self.model(image, verbose=False)
        detections_2d = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.class_names[cls_id]
                
                detections_2d.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': confidence
                })
        
        self.get_logger().info(f'  → Found {len(detections_2d)} 2D detections')
        for det in detections_2d:
            self.get_logger().info(f'    - {det["class_name"]}: {det["confidence"]:.2f}')
        
        # STEP 3: Load LiDAR point cloud
        velodyne_path = os.path.join(self.velodyne_dir, self.velodyne_files[self.frame_idx])
        points_3d = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        self.get_logger().info(f'STEP 2: Loaded {points_3d.shape[0]} LiDAR points')
        
        # STEP 4: Project LiDAR to image and paint scores
        self.get_logger().info('STEP 3: Projecting LiDAR to image plane...')
        points_2d, depths, mask = self.project_lidar_to_image(points_3d)
        
        # STEP 5: Paint detection scores onto LiDAR points
        self.get_logger().info('STEP 4: Painting 2D detection scores onto LiDAR points...')
        enriched_points = self.paint_scores_on_lidar(points_3d, points_2d, depths, mask, detections_2d)
        self.get_logger().info(f'  → Enriched {len(enriched_points)} points with semantic info')
        
        # STEP 6: Perform 3D object detection using enriched LiDAR
        self.get_logger().info('STEP 5: Running 3D object detection on enriched LiDAR...')
        detections_3d = self.detect_3d_objects(enriched_points, points_3d)
        self.get_logger().info(f'  → Generated {len(detections_3d)} 3D bounding boxes')
        
        # STEP 7: Visualize results
        self.visualize_pipeline(image, detections_2d, points_3d, enriched_points, detections_3d)
        
        # STEP 8: Publish ROS messages
        self.publish_messages(image, detections_2d, detections_3d, enriched_points)
        
        self.get_logger().info('=' * 60)
    
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
        """
        Paint 2D detection scores onto LiDAR points
        Returns: array of [x, y, z, intensity, class_id, confidence]
        """
        # Start with original points (only those projected to image)
        valid_points = points_3d[mask]
        enriched = np.zeros((valid_points.shape[0], 6))  # x,y,z,intensity,class,conf
        enriched[:, :4] = valid_points
        enriched[:, 4] = -1  # No class by default
        enriched[:, 5] = 0.0  # No confidence by default
        
        # For each 2D detection, find LiDAR points inside and paint scores
        for det in detections_2d:
            x1, y1, x2, y2 = det['bbox']
            
            # Find points inside this bounding box
            inside_mask = (
                (points_2d[:, 0] >= x1) & (points_2d[:, 0] <= x2) &
                (points_2d[:, 1] >= y1) & (points_2d[:, 1] <= y2)
            )
            
            # Paint class ID and confidence
            enriched[inside_mask, 4] = det['class_id']
            enriched[inside_mask, 5] = det['confidence']
        
        # Return only points with semantic labels
        labeled_mask = enriched[:, 4] >= 0
        return enriched[labeled_mask]
    
    def detect_3d_objects(self, enriched_points, all_points):
        """
        Perform 3D object detection using enriched LiDAR points
        Uses clustering to generate 3D bounding boxes
        """
        if len(enriched_points) == 0:
            return []
        
        detections_3d = []
        
        # Group points by class
        unique_classes = np.unique(enriched_points[:, 4])
        
        for class_id in unique_classes:
            if class_id < 0:
                continue
            
            class_mask = enriched_points[:, 4] == class_id
            class_points = enriched_points[class_mask]
            
            # Cluster points of same class
            if len(class_points) < 10:
                continue
            
            # DBSCAN clustering in 3D space
            clustering = DBSCAN(eps=1.5, min_samples=10).fit(class_points[:, :3])
            labels = clustering.labels_
            
            # For each cluster, create a 3D bounding box
            for label in set(labels):
                if label == -1:  # Noise
                    continue
                
                cluster_mask = labels == label
                cluster_points = class_points[cluster_mask]
                
                if len(cluster_points) < 10:
                    continue
                
                # Calculate 3D bounding box
                min_coords = np.min(cluster_points[:, :3], axis=0)
                max_coords = np.max(cluster_points[:, :3], axis=0)
                
                center = (min_coords + max_coords) / 2
                size = max_coords - min_coords
                
                # Average confidence
                avg_confidence = np.mean(cluster_points[:, 5])
                
                # Get class name
                class_name = self.class_names[int(class_id)]
                
                detections_3d.append({
                    'center': center,
                    'size': size,
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'confidence': avg_confidence,
                    'num_points': len(cluster_points)
                })
        
        return detections_3d
    
    def visualize_pipeline(self, image, detections_2d, points_3d, enriched_points, detections_3d):
        """Visualize the complete pipeline in 2x2 grid maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        # Create 4 views for 2x2 grid
        # View 1: Camera and LiDAR data (raw camera with basic info)
        view1 = image.copy()
        cv2.putText(view1, 'Camera and LiDAR data', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(view1, f'Frame: {self.frame_idx+1}/{self.num_frames}', (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # View 2: LiDAR Point Cloud (colored by depth)
        view2 = self.create_lidar_visualization(image, points_3d)
        cv2.putText(view2, 'LiDAR Point Cloud', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # View 3: Data fusion and Depth estimation (detections with colored overlay)
        view3 = self.create_fusion_overlay(image, detections_2d, enriched_points)
        cv2.putText(view3, 'Data fusion and Depth estimation', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # View 4: 3D Detection Result with depth
        view4 = self.create_3d_detection_view(image, detections_3d, enriched_points)
        cv2.putText(view4, '3D Object Detection', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Scale down by 75% to fit screen better while maintaining aspect ratio
        scale = 0.75
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        view1 = cv2.resize(view1, (new_w, new_h))
        view2 = cv2.resize(view2, (new_w, new_h))
        view3 = cv2.resize(view3, (new_w, new_h))
        view4 = cv2.resize(view4, (new_w, new_h))
        
        # Create 2x2 grid
        top_row = np.hstack([view1, view2])
        bottom_row = np.hstack([view3, view4])
        combined = np.vstack([top_row, bottom_row])
        
        # Add border and title
        combined = cv2.copyMakeBorder(combined, 60, 40, 10, 10, 
                                      cv2.BORDER_CONSTANT, value=(40, 40, 40))
        
        # Title
        title_x = combined.shape[1] // 2 - 300
        cv2.putText(combined, 'Multi-Modal Object Detection Pipeline', 
                   (title_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Controls at bottom
        controls_x = combined.shape[1] - 350
        cv2.putText(combined, 'SPACE=Next | B=Back | Q=Quit', 
                   (controls_x, combined.shape[0]-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display single window with all 4 views
        cv2.imshow('Multi-Modal Detection', combined)
        cv2.waitKey(1)
    
    def create_lidar_visualization(self, image, points_3d):
        """Create colored LiDAR point cloud visualization"""
        h, w = image.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(points_3d) == 0:
            return vis
        
        points_2d, depths, mask = self.project_lidar_to_image(points_3d)
        
        if len(points_2d) == 0:
            return vis
        
        # Normalize depths for color mapping
        depths_valid = depths[depths > 0]
        if len(depths_valid) == 0:
            return vis
        
        depth_min, depth_max = np.min(depths_valid), np.max(depths_valid)
        
        # Draw points colored by depth
        for pt, depth in zip(points_2d, depths):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h and depth > 0:
                # Color map: close = red/yellow, far = blue/cyan
                normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)
                
                # Create color gradient
                if normalized < 0.25:
                    color = (0, int(255 * normalized * 4), 255)  # Red to yellow
                elif normalized < 0.5:
                    color = (0, 255, int(255 * (1 - (normalized - 0.25) * 4)))  # Yellow to green
                elif normalized < 0.75:
                    color = (int(255 * (normalized - 0.5) * 4), 255, 0)  # Green to cyan
                else:
                    color = (255, int(255 * (1 - (normalized - 0.75) * 4)), 0)  # Cyan to blue
                
                cv2.circle(vis, (x, y), 2, color, -1)
        
        return vis
    
    def create_fusion_overlay(self, image, detections_2d, enriched_points):
        """Create fusion visualization with colored depth overlay"""
        vis = image.copy()
        
        # Create semi-transparent colored overlay for depth
        overlay = np.zeros_like(image)
        
        if len(enriched_points) > 0:
            points_2d, depths, mask = self.project_lidar_to_image(enriched_points)
            
            # Get depth range
            if len(depths) > 0:
                depth_min, depth_max = np.min(depths), np.max(depths)
                
                # Draw colored depth points
                for pt, depth, cls_id in zip(points_2d, depths, enriched_points[:, 4]):
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                        # Color based on depth
                        normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)
                        
                        # Warm colors for close objects, cool for far
                        if normalized < 0.5:
                            color = (int(100 + 155 * (1 - normalized * 2)), 
                                    int(50 + 205 * (1 - normalized * 2)), 
                                    255)  # Pink/magenta for close
                        else:
                            color = (255, 
                                    int(255 * (1 - (normalized - 0.5) * 2)), 
                                    int(100 + 155 * (normalized - 0.5) * 2))  # Red to orange for far
                        
                        cv2.circle(overlay, (x, y), 3, color, -1)
        
        # Blend overlay with original image
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
        # Draw 2D detection boxes
        for det in detections_2d:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(vis, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis
    
    def create_3d_detection_view(self, image, detections_3d, enriched_points):
        """Create 3D detection visualization"""
        vis = image.copy()
        
        # Add semi-transparent depth coloring in background
        if len(enriched_points) > 0:
            points_2d, depths, mask = self.project_lidar_to_image(enriched_points)
            
            if len(depths) > 0:
                depth_min, depth_max = np.min(depths), np.max(depths)
                
                for pt, depth in zip(points_2d, depths):
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                        normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)
                        # Light depth coloring
                        color = (int(200 * normalized), int(150 * (1-normalized)), int(100 + 100 * normalized))
                        cv2.circle(vis, (x, y), 1, color, -1)
        
        # Draw 3D bounding boxes
        for det_3d in detections_3d:
            center = det_3d['center']
            size = det_3d['size']
            
            # Create 8 corners of 3D box
            corners = self.get_3d_box_corners(center, size)
            corners_2d = self.project_3d_box_to_image(corners)
            
            if corners_2d is not None:
                # Draw 3D box
                self.draw_3d_box(vis, corners_2d, (0, 0, 255))
                
                # Add label with confidence
                label = f"{det_3d['class_name']}: {det_3d['confidence']:.2f}"
                cx, cy = int(np.mean(corners_2d[:, 0])), int(np.min(corners_2d[:, 1]))
                if cy > 0:
                    cv2.putText(vis, label, (cx, cy-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return vis
    
    def get_3d_box_corners(self, center, size):
        """Get 8 corners of 3D bounding box"""
        x, y, z = center
        l, w, h = size
        
        corners = np.array([
            [x-l/2, y-w/2, z-h/2],
            [x+l/2, y-w/2, z-h/2],
            [x+l/2, y+w/2, z-h/2],
            [x-l/2, y+w/2, z-h/2],
            [x-l/2, y-w/2, z+h/2],
            [x+l/2, y-w/2, z+h/2],
            [x+l/2, y+w/2, z+h/2],
            [x-l/2, y+w/2, z+h/2],
        ])
        return corners
    
    def project_3d_box_to_image(self, corners_3d):
        """Project 3D box corners to image"""
        corners_hom = np.hstack([corners_3d, np.ones((8, 1))])
        corners_cam = (self.Tr_velo_to_cam @ corners_hom.T).T
        corners_cam = (self.R0_rect @ corners_cam.T).T
        
        if np.any(corners_cam[:, 2] <= 0):
            return None
        
        corners_2d_hom = (self.P2 @ corners_cam.T).T
        corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]
        
        return corners_2d
    
    def draw_3d_box(self, img, corners_2d, color):
        """Draw 3D bounding box on image"""
        corners = corners_2d.astype(int)
        
        # Draw bottom face
        for i in range(4):
            cv2.line(img, tuple(corners[i]), tuple(corners[(i+1)%4]), color, 2)
        
        # Draw top face
        for i in range(4, 8):
            cv2.line(img, tuple(corners[i]), tuple(corners[4+(i+1)%4]), color, 2)
        
        # Draw vertical lines
        for i in range(4):
            cv2.line(img, tuple(corners[i]), tuple(corners[i+4]), color, 2)
    
    def publish_messages(self, image, detections_2d, detections_3d, enriched_points):
        """Publish ROS messages"""
        # Store for continuous republishing
        self.current_image = image
        self.current_enriched_points = enriched_points
        
        # Publish image
        img_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera'
        self.image_pub.publish(img_msg)
        
        # Publish enriched point cloud
        if len(enriched_points) > 0:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'velodyne'
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='class_id', offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name='confidence', offset=20, datatype=PointField.FLOAT32, count=1),
            ]
            
            pc_msg = pc2.create_cloud(header, fields, enriched_points)
            self.enriched_pc_pub.publish(pc_msg)
        
        # Publish 2D detections
        detection_2d_array = Detection2DArray()
        detection_2d_array.header = img_msg.header
        self.detection_2d_pub.publish(detection_2d_array)
        
        # Publish 3D detections  
        detection_3d_array = Detection3DArray()
        detection_3d_array.header.stamp = self.get_clock().now().to_msg()
        detection_3d_array.header.frame_id = 'velodyne'
        self.detection_3d_pub.publish(detection_3d_array)
    
    def republish_current_frame(self):
        """Continuously republish current frame for RViz"""
        if self.current_image is None:
            return
        
        # Republish image
        img_msg = self.bridge.cv2_to_imgmsg(self.current_image, encoding='bgr8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera'
        self.image_pub.publish(img_msg)
        
        # Republish enriched points
        if self.current_enriched_points is not None and len(self.current_enriched_points) > 0:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'velodyne'
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='class_id', offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name='confidence', offset=20, datatype=PointField.FLOAT32, count=1),
            ]
            
            pc_msg = pc2.create_cloud(header, fields, self.current_enriched_points)
            self.enriched_pc_pub.publish(pc_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MultiModalFusionNode()
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
