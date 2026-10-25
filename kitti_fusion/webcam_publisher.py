import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')

        # Declare parameters for configurability
        self.declare_parameter('camera_index', 2)
        self.declare_parameter('fps', 30)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 720)

        # Get parameters
        camera_index = self.get_parameter('camera_index').value
        fps = self.get_parameter('fps').value
        frame_width = self.get_parameter('frame_width').value
        frame_height = self.get_parameter('frame_height').value

        # Publisher for raw image frames
        self.publisher = self.create_publisher(Image, 'camera', 10)
        self.bridge = CvBridge()

        # Open webcam
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open webcam at index {camera_index}")
            raise RuntimeError("Webcam initialization failed")

        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Enable auto-focus and auto-exposure for better image quality
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = auto mode

        # Log actual camera settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.get_logger().info(f"Webcam opened: {actual_width}x{actual_height} @ {actual_fps} FPS")

        # Timer based on FPS
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.publish_frame)

        self.frame_count = 0

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame from webcam")
            return
        
        # Optional: Apply basic image enhancement
        # Slight sharpening can help with detection
        # frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)  # Increase contrast/brightness
        
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        
        self.publisher.publish(msg)
        
        self.frame_count += 1
        if self.frame_count % 300 == 0:  # Log every 10 seconds at 30 FPS
            self.get_logger().info(f"Published {self.frame_count} frames")

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main():
    rclpy.init()
    try:
        node = WebcamPublisher()
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
