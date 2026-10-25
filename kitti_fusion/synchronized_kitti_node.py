import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import cv2
import os
from ultralytics import YOLO
import numpy as np

class SynchronizedKittiNode(Node):
    def __init__(self):
        super().__init__('synchronized_kitti_node')
        
        # Publishers
        self.image_pub = self.create_publisher(Image, '/kitti/camera/image_raw', 10)
        self.detection_pub = self.create_publisher(Detection2DArray, '/kitti/camera/detections', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/kitti/lidar/points', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.get_logger().info('Loading YOLOv8 model...')
        self.model = YOLO('yolov8n.pt')
        self.get_logger().info('YOLOv8 model loaded!')
        
        # Dataset paths
        self.image_dir = '/home/lenovo/ros2_ws/datasets/kitti/training/image_2/'
        self.velodyne_dir = '/home/lenovo/ros2_ws/datasets/kitti/data_object_velodyne.1/training/velodyne/'
        
        # Verify paths exist
        if not os.path.exists(self.image_dir):
            self.get_logger().error(f'Image directory not found: {self.image_dir}')
            return
        if not os.path.exists(self.velodyne_dir):
            self.get_logger().error(f'Velodyne directory not found: {self.velodyne_dir}')
            return
            
        # Get sorted list of files
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.velodyne_files = sorted([f for f in os.listdir(self.velodyne_dir) if f.endswith('.bin')])
        
        self.get_logger().info(f'Found {len(self.image_files)} images')
        self.get_logger().info(f'Found {len(self.velodyne_files)} velodyne files')
        
        if len(self.image_files) == 0 or len(self.velodyne_files) == 0:
            self.get_logger().error('No data files found!')
            return
        
        # Use minimum of both datasets
        self.num_frames = min(len(self.image_files), len(self.velodyne_files))
        self.get_logger().info(f'Total synchronized frames: {self.num_frames}')
        
        # Current frame index
        self.frame_idx = 0
        
        # Storage for current data
        self.current_image = None
        self.current_detections = None
        self.current_points = None
        
        # Manual control
        self.get_logger().info('===================================')
        self.get_logger().info('SYNCHRONIZED MANUAL CONTROL')
        self.get_logger().info('===================================')
        self.get_logger().info('SPACEBAR: Next frame')
        self.get_logger().info('B: Previous frame')
        self.get_logger().info('Q: Quit')
        self.get_logger().info('===================================')
        
        # Load and display first frame
        self.load_and_publish_frame()
        
        # Timer to check for keyboard input and republish
        self.timer = self.create_timer(0.05, self.timer_callback)
        
    def load_and_publish_frame(self):
        """Load and publish both camera and LiDAR data for current frame"""
        
        # Load camera image
        image_path = os.path.join(self.image_dir, self.image_files[self.frame_idx])
        self.current_image = cv2.imread(image_path)
        
        if self.current_image is None:
            self.get_logger().error(f'Failed to load image: {image_path}')
            return
        
        # Load LiDAR point cloud
        velodyne_path = os.path.join(self.velodyne_dir, self.velodyne_files[self.frame_idx])
        self.current_points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        
        self.get_logger().info(f'')
        self.get_logger().info(f'====== FRAME {self.frame_idx + 1}/{self.num_frames} ======')
        self.get_logger().info(f'Image: {self.image_files[self.frame_idx]}')
        self.get_logger().info(f'LiDAR: {self.velodyne_files[self.frame_idx]} ({self.current_points.shape[0]} points)')
        
        # Run YOLO detection
        results = self.model(self.current_image, verbose=False)
        
        # Publish camera image
        img_msg = self.bridge.cv2_to_imgmsg(self.current_image, encoding='bgr8')
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
        self.current_detections = detection_array
        self.get_logger().info(f'Detected {len(detection_array.detections)} objects')
        
        # Publish point cloud
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'velodyne'
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        pointcloud_msg = pc2.create_cloud(header, fields, self.current_points)
        self.pointcloud_pub.publish(pointcloud_msg)
        
        # Display camera detection window
        annotated_image = results[0].plot()
        cv2.putText(annotated_image, f'Frame: {self.frame_idx + 1}/{self.num_frames}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_image, 'SPACE: Next | B: Back | Q: Quit', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('KITTI Camera Detections', annotated_image)
        
    def timer_callback(self):
        """Check for keyboard input and republish current frame"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Spacebar - next frame
            self.frame_idx += 1
            if self.frame_idx >= self.num_frames:
                self.get_logger().info('Reached end of dataset, looping back...')
                self.frame_idx = 0
            self.load_and_publish_frame()
            
        elif key == ord('b'):  # B - previous frame
            self.frame_idx -= 1
            if self.frame_idx < 0:
                self.frame_idx = self.num_frames - 1
            self.load_and_publish_frame()
            
        elif key == ord('q'):  # Q - quit
            self.get_logger().info('Quitting...')
            cv2.destroyAllWindows()
            rclpy.shutdown()
        
        # Republish current frame data to keep fusion updated
        if self.current_image is not None:
            img_msg = self.bridge.cv2_to_imgmsg(self.current_image, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera'
            self.image_pub.publish(img_msg)
            
        if self.current_detections is not None:
            self.current_detections.header.stamp = self.get_clock().now().to_msg()
            self.detection_pub.publish(self.current_detections)
            
        if self.current_points is not None:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'velodyne'
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            
            pointcloud_msg = pc2.create_cloud(header, fields, self.current_points)
            self.pointcloud_pub.publish(pointcloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SynchronizedKittiNode()
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
