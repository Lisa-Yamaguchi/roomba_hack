#!/usr/bin/env python3
import rospy
from navigation_tutorial.srv import MoveTrigger, MoveTriggerRequest
from navigation_tutorial.srv import TakeImage, TakeImageRequest
import cv2
from keypoint_rcnn import judgement
from cv_bridge import CvBridge

class TaskManager:
    def __init__(self):
        rospy.init_node('task_manager')
        self.bridge = CvBridge()
        
        # service client
        self.move_robot = rospy.ServiceProxy('/move_robot', MoveTrigger)
        self.take_image = rospy.ServiceProxy('/take_image', TakeImage)

        self.move_robot.wait_for_service()
        rospy.loginfo("Service /move_robot is ready!")
        self.take_image.wait_for_service()
        rospy.loginfo("Service /take_image is ready!")

    def move(self, straight, turn):
        command = MoveTriggerRequest()
        command.straight = straight
        command.turn = turn
        rospy.loginfo(f"request: straight={straight}, turn={turn}")
        response = self.move_robot(command) # service call
        return response.success
    
    def takeimage(self, start):
        command = TakeImageRequest()
        command.start = start
        rospy.loginfo(f"request: start={start}")
        response = self.take_image(command) # service call
        return response.images
    
    def callback_image(self, images):
        cut_images = []
        for i in range(len(images)):
            # 画像読み込み
            image = cv2.imread(images[i])
            # img[top : bottom, left : right]
            # サンプルの切り出し
            cut_image = image[0 : 480, 320: 960]
            cut_images.append(cut_image)
        
        judges = []
        for j in range(len(cut_images)):
            judges.append(judgement(cut_images[j]))
        
        if all(i == judges[0] for i in judges):#もし全ての要素が同じだったら

            if judges[0] == "left":
                return "lett"
            elif judges[0] == "right":
                return "right"
            else:
                rospy.loginfo("judge failed")
                #画像を撮り直す

    def main(self):
        """
        タスクの流れを手続的に記述する
        """
        result = self.move(1.0, 0.0)
        if result:
          rospy.loginfo("move success!")
        else:
          rospy.loginfo("move failed!")
        
        result = self.move(0.0, 90.0)
        if result:
          rospy.loginfo("turn success!")
        else:
          rospy.loginfo("turn failed!")

        result = self.move(1.0, 0.0)
        if result:
          rospy.loginfo("move success!")
        else:
          rospy.loginfo("move failed!")

        result = self.move(0.0, 90.0)
        if result:
          rospy.loginfo("turn success!")
        else:
          rospy.loginfo("turn failed!")

        result = self.takeimage()
        if result == None:
           rospy.loginfo("take images failed!")

        else:
            rospy.loginfo("take images successed!")
            goto = self.callback_image(self.bridge.imgmsg_to_cv2(result, "bgr8"))
            if goto == "left":
                result = self.move(0.0, 30.0)
                result = self.move(3.0, 0.0)
                if result:
                    rospy.loginfo("move success!")
                else:
                    rospy.loginfo("move failed!")
            elif goto == "right":
                result = self.move(0.0, -30.0)
                result = self.move(3.0, 0.0)
                if result:
                    rospy.loginfo("move success!")
                else:
                    rospy.loginfo("move failed!")
        


        rospy.loginfo("task completed!")

if __name__ == '__main__':
    task_manager = TaskManager()
    task_manager.main()