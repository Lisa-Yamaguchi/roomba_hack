#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('camera/color/image_raw', Image, self.image_callback)
        self.image_counter = 0
        self.output_folder = 'saved_images'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.save_image = False

    def image_callback(self, data):
        if self.save_image:
            try:
                # ROS ImageメッセージをOpenCVの画像に変換
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except Exception as e:
                rospy.logerr(e)
                return
            
            # 画像を保存
            image_filename = os.path.join(self.output_folder, f"image_{self.image_counter}.jpg")
            cv2.imwrite(image_filename, cv_image)
            self.image_counter += 1
            rospy.loginfo(f"Saved image: {image_filename}")
            self.save_image = False

if __name__ == '__main__':
    rospy.init_node('image_saver', anonymous=True)
    image_subscriber = ImageSubscriber()
    
    while not rospy.is_shutdown():
        user_input = raw_input("Press Enter to save the next image, or type 'exit' to quit: ")  # Python 2
        # user_input = input("Press Enter to save the next image, or type 'exit' to quit: ")  # Python 3
        
        if user_input.lower() == 'exit':
            break
        elif user_input == '':
            image_subscriber.save_image = True
    
    rospy.signal_shutdown("Image saving program terminated.")
