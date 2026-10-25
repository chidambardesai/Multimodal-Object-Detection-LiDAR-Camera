#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import os
from ultralytics import YOLO
import numpy as np

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # Publishers
        self.image_pub = self.create_publisher(Image, '/kitti/camera/image_raw', 10)
        self.detection_pub = self.create_publisher(Detection2DArray, '/kitti/camera/detections', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.get_logger().info('Loading YOLOv8 model...')
        self.model = YOLO('yolov8n.pt')
        self.get_logger().info('YOLOv8 model loaded!')
        
        # Dataset path
        self.image_dir = '/home/lenovo/ros2_ws/datasets/kitti/training/image_2/'
        
        # Verify path exists
        if not os.path.exists(self.image_dir):
            self.get_logger().error(f'Image directory not found: {self.image_dir}')
            return
            
        # Get sorted list of images
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.get_logger().info(f'Found {len(self.image_files)} images')
        
        if len(self.image_files) == 0:
            self.get_logger().error('No images found!')
            return
        
        # Current frame index
        self.frame_idx = 0
        
        # Manual control - process frame only when called
        self.get_logger().info('===================================')
        self.get_logger().info('MANUAL CONTROL MODE')
        self.get_logger().info('===================================')
        self.get_logger().info('Press SPACEBAR to go to next image')
        self.get_logger().info('Press "q" to quit')
        self.get_logger().info('===================================')
        
        # Process first frame
        self.process_frame()
        
        # Timer to check for keyboard input
        self.timer = self.create_timer(0.05, self.check_input)
        
    def check_input(self):
        """Check for keyboard input"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Spacebar - next frame
            self.frame_idx += 1
            if self.frame_idx >= len(self.image_files):
                self.get_logger().info('Reached end of dataset, looping back to start...')
                self.frame_idx = 0
            self.process_frame()
            
        elif key == ord('q'):  # Q - quit
            self.get_logger().info('Quitting...')
            rclpy.shutdown()
            
        elif key == ord('b'):  # B - previous frame
            self.frame_idx -= 1
            if self.frame_idx < 0:
                self.frame_idx = len(self.image_files) - 1
            self.process_frame()
        
    def process_frame(self):
        """Process and publish current frame"""
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[self.frame_idx])
        image = cv2.imread(image_path)
        
        if image is None:
            self.get_logger().error(f'Failed to load image: {image_path}')
            return
        
        self.get_logger().info(f'=== Frame {self.frame_idx}/{len(self.image_files)} === {self.image_files[self.frame_idx]}')
        
        # Run YOLOv8 detection
        results = self.model(image, verbose=False)
        
        # Publish original image
        img_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera'
        self.image_pub.publish(img_msg)
        
        # Publish detections
        detection_array = Detection2DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = 'camera'
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = Detection2D()
                
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detection.bbox.center.position.x = float((x1 + x2) / 2)
                detection.bbox.center.position.y = float((y1 + y2) / 2)
                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)
                
                # Class and confidence
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(int(box.cls[0]))
                hypothesis.hypothesis.score = float(box.conf[0])
                detection.results.append(hypothesis)
                
                detection_array.detections.append(detection)
        
        self.detection_pub.publish(detection_array)
        self.get_logger().info(f'Detected {len(detection_array.detections)} objects')
        
        # Draw detections for visualization
        annotated_image = results[0].plot()
        
        # Add instructions on image
        cv2.putText(annotated_image, f'Frame: {self.frame_idx}/{len(self.image_files)}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_image, 'SPACE: Next | B: Back | Q: Quit', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('KITTI Camera Detections', annotated_image)

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
