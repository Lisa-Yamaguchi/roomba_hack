#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import os
from navigation_tutorial.srv import TakeImage, TakeImageResponse

class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_saver', anonymous=True)
        self.bridge = CvBridge()

        # Subscriber
        self.image_sub = rospy.Subscriber('camera/color/image_raw', Image, self.image_callback)
        self.image_counter = 0
        self.output_folder = 'saved_images'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.save_image = True

        #self.step_sub = rospy.Subscriber('task2_pub', Int32, self.step_callback)
        #self.step_pub = rospy.Publisher('task2_images', Image)
        
        # Service Server
        rospy.Service('/take_image', TakeImage, self.srv_callback)


    def image_callback(self, data):
        if self.save_image:
            try:
                # ROS ImageメッセージをOpenCVの画像に変換
                self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except Exception as e:
                rospy.logerr(e)
                return
            
            """# 画像を保存
            image_filename = os.path.join(self.output_folder, f"image_{self.image_counter}.jpg")
            cv2.imwrite(image_filename, cv_image)
            self.image_counter += 1
            rospy.loginfo(f"Saved image: {image_filename}")"""
        
    def get_image(self):
        images = []
        for i in range(5):
            images.append(self.cv_image)
            rospy.sleep(1.0)
        return images
    
    def srv_callback(self, req):
        res = TakeImageResponse()
        try:
            if req :
                imgsub = ImageSubscriber()
                images = imgsub.get_image()
                res.images = images
                return res

        except rospy.ROSInterruptException:
            res.images = None
            return res

        """if self.step == 2:
            imgsub = ImageSubscriber()
            images = imgsub.get_image()
            self.step_pub.Publish(images)"""


"""if __name__ == '__main__':
    
    image_subscriber = ImageSubscriber()
    
    while not rospy.is_shutdown():
        user_input = input("Press Enter to save the next image, or type 'exit' to quit: ")
        
        if user_input.lower() == 'exit':
            break
        elif user_input == '':
            image_subscriber.save_image = not image_subscriber.save_image
    
    rospy.signal_shutdown("Image saving program terminated.")
"""