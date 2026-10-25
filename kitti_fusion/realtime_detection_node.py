#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class RealtimeDetectionNode(Node):
    def __init__(self):
        super().__init__('realtime_detection_node')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.get_logger().info('Loading YOLOv8 model...')
        self.yolo_model = YOLO('yolov8n.pt')
        self.get_logger().info('YOLOv8 loaded!')
        
        # Load MiDaS depth estimation model
        self.get_logger().info('Loading MiDaS depth model...')
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = self.midas_transforms.small_transform
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.midas.to(self.device)
        self.midas.eval()
        
        self.get_logger().info(f'MiDaS loaded on {self.device}!')
        
        # YOLO class names
        self.class_names = self.yolo_model.names
        
        # Subscribe to webcam
        self.subscription = self.create_subscription(
            Image,
            'camera',
            self.image_callback,
            10
        )
        
        # Camera intrinsics (approximate for typical webcam)
        self.fx = 800.0  # focal length x
        self.fy = 800.0  # focal length y
        self.cx = 640.0  # principal point x
        self.cy = 360.0  # principal point y
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('REAL-TIME MULTI-MODAL DETECTION')
        self.get_logger().info('Webcam → 2D Detection → Depth Estimation → 3D Estimation')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Press Q to quit')
        self.get_logger().info('=' * 60)
        
        self.frame_count = 0
        
    def estimate_depth(self, image):
        """Estimate depth map using MiDaS"""
        # Prepare image for MiDaS
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-255 for visualization
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return depth_map, depth_map_normalized
    
    def estimate_3d_bbox(self, bbox_2d, depth_map, avg_depth):
        """Estimate 3D bounding box from 2D bbox and depth"""
        x1, y1, x2, y2 = bbox_2d
        
        # Get depth values in bounding box region
        depth_roi = depth_map[y1:y2, x1:x2]
        
        if depth_roi.size == 0:
            return None
        
        # Use median depth as object depth
        object_depth = np.median(depth_roi)
        
        # Normalize depth (MiDaS outputs inverse depth)
        # Convert to approximate meters (calibration needed for accuracy)
        depth_meters = 10.0 / (object_depth + 1e-6)  # Approximate conversion
        
        # Calculate 3D center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Back-project to 3D using camera intrinsics
        X = (center_x - self.cx) * depth_meters / self.fx
        Y = (center_y - self.cy) * depth_meters / self.fy
        Z = depth_meters
        
        # Estimate size from 2D bbox and depth
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # Approximate 3D size (assumes object fills bbox)
        width_3d = width_2d * depth_meters / self.fx
        height_3d = height_2d * depth_meters / self.fy
        length_3d = width_3d  # Assume cubic-ish object
        
        return {
            'center': np.array([X, Y, Z]),
            'size': np.array([length_3d, width_3d, height_3d]),
            'depth': depth_meters
        }
    
    def image_callback(self, msg):
        """Process incoming webcam frames"""
        try:
            # Convert ROS Image to OpenCV
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = image.shape[:2]
            
            self.frame_count += 1
            
            # STEP 1: 2D Object Detection
            results = self.yolo_model(image, verbose=False)
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
            
            # STEP 2: Depth Estimation
            depth_map, depth_vis = self.estimate_depth(image)
            
            # STEP 3: 3D Bounding Box Estimation
            detections_3d = []
            avg_depth = np.mean(depth_map)
            
            for det_2d in detections_2d:
                bbox_3d = self.estimate_3d_bbox(det_2d['bbox'], depth_map, avg_depth)
                if bbox_3d is not None:
                    detections_3d.append({
                        'bbox_2d': det_2d,
                        'bbox_3d': bbox_3d
                    })
            
            # STEP 4: Visualize
            self.visualize_realtime(image, depth_vis, detections_2d, detections_3d)
            
            # Log periodically
            if self.frame_count % 30 == 0:
                self.get_logger().info(f'Frame {self.frame_count}: Detected {len(detections_2d)} objects')
            
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')
    
    def visualize_realtime(self, image, depth_vis, detections_2d, detections_3d):
        """Create 2x2 grid visualization"""
        h, w = image.shape[:2]
        
        # View 1: Raw camera feed
        view1 = image.copy()
        cv2.putText(view1, 'Camera Feed', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(view1, f'Frame: {self.frame_count}', (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # View 2: Depth estimation (colored)
        view2 = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        cv2.putText(view2, 'Depth Estimation', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(view2, '(Closer = Bright)', (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # View 3: 2D Object Detection
        view3 = image.copy()
        for det in detections_2d:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(view3, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(view3, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(view3, '2D Object Detection', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(view3, f'{len(detections_2d)} objects', (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # View 4: 3D Estimation with depth overlay
        view4 = image.copy()
        
        # Overlay depth with transparency
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        view4 = cv2.addWeighted(view4, 0.6, depth_colored, 0.4, 0)
        
        # Draw 2D boxes with depth info
        for det_3d in detections_3d:
            det_2d = det_3d['bbox_2d']
            bbox_3d = det_3d['bbox_3d']
            
            x1, y1, x2, y2 = det_2d['bbox']
            cv2.rectangle(view4, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Show depth in label
            depth_m = bbox_3d['depth']
            label = f"{det_2d['class_name']}: {depth_m:.2f}m"
            cv2.putText(view4, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.putText(view4, '3D Estimation (Depth)', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(view4, f'{len(detections_3d)} 3D boxes', (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Scale down to fit screen
        scale = 0.65
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
        
        title_x = combined.shape[1] // 2 - 350
        cv2.putText(combined, 'Real-Time Multi-Modal Object Detection', 
                   (title_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.putText(combined, 'Press Q to Quit', 
                   (combined.shape[1] - 200, combined.shape[0]-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display
        cv2.imshow('Real-Time Detection', combined)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quitting...')
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RealtimeDetectionNode()
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
