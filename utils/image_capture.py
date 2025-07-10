import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2 as cv
import uuid
import os
import numpy as np
import datetime
import time
from sensor_msgs.msg import Image


class ImageSubscriber(Node):
  def __init__(self):
    super().__init__('ImageSubscriber')
    # 创建sub节点，订阅image_raw话题
    self.declare_parameter("fps",2.0)
    self.declare_parameter("sub_img_topic","/image_raw")
    self.declare_parameter("track_img_folder","track_image")
    self.declare_parameter("obstacle_img_folder","obstacle_image")
    self.subscription = self.create_subscription(
      Image,
      self.get_parameter('sub_img_topic').get_parameter_value().string_value,
      self.listener_callback,
      1)
    # 创建CvBridge实例
    self.bridge = CvBridge()
    self.uuid = -1
    self.line_follow_image = np.zeros((640, 224, 3))
    self.detection_image = np.zeros((640, 480, 3))
    self.start_time = time.time()
    self.time_diff_threshold = 1 / self.get_parameter('fps').get_parameter_value().double_value

    self.line_follow_image_folder = self.get_parameter('track_img_folder').get_parameter_value().string_value
    self.detection_image_folder = self.get_parameter('obstacle_img_folder').get_parameter_value().string_value
    if not os.path.exists(self.line_follow_image_folder):
      os.makedirs(self.line_follow_image_folder)
    if not os.path.exists(self.detection_image_folder):
      os.makedirs(self.detection_image_folder)
    self.subscription
  
  def listener_callback(self, msg):
    end_time = time.time()
    time_diff = end_time - self.start_time
    if time_diff > self.time_diff_threshold:
      self.start_time = end_time
      self.detection_image = self.bridge.imgmsg_to_cv2(msg)
      self.line_follow_image = self.detection_image[255:479,:,:].copy()
      self.uuid = uuid.uuid1()
      cv.imwrite(os.path.join(self.line_follow_image_folder, str(self.uuid) + '.jpg'), self.line_follow_image)
      cv.imwrite(os.path.join(self.detection_image_folder, str(self.uuid) + '.jpg'), self.detection_image)
      print("Saved " + str(self.uuid) + ".jpg")


def main(args=None):
  rclpy.init(args=args)
  image_subscriber = ImageSubscriber()
  rclpy.spin(image_subscriber)
  image_subscriber.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()
