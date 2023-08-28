#!/usr/bin/env python3

#from simple_control2 import SimpleController
from three_dimensions_tutorial.scripts.keypoint_rcnn import judgement
#from three_dimensions_tutorial.scripts import take_image
import rospy
import cv2


step = 0

class TaskManager:
    def __init__(self) :
        rospy.init_node('task_manager')
        self.step = 0
        step_sub = rospy.Subscriber('task2_step1', Int32, self.callback)
        images_sub = rospy.Subscriber('task2_images', list, self.callback_image)
        self.pub = rospy.Publisher('task2_pub', Int32)
        self.move_pub = rospy.Publisher('task2_move', Int32)

    def callback(self, msg):
        self.step = msg.data
        #self.step=1
        if self.step == 1:
            self.pub.Publish(2)

    def callback_image(self, msg):
        images = msg.data
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
                self.move_pub.Publish(4)
            elif judges[0] == "right":
                self.move_pub.Publish(6)
            else:
                self.pub.Publish(2)
                #画像を撮り直す


    """def main(self):
        #サービスとパブリッシャーの呼び出し
        #手続き
        if self.step == 1:
            self.pub.publish(2)
        if self.step == 2:
            self.pub.publish(3)
    """
        
        

        

if __name__ == '__main__':
    try:
        tm = TaskManager()
        #tm.main()
    except rospy.ROSInitException:
        pass






"""

if __name__=='__main__':
    try:
        simple_controller = SimpleController()
        simple_controller.go_straight(1.0)
        simple_controller.turn_left(90)
        simple_controller.go_straight(1.0)
        simple_controller.turn_left(90)
        step = 1 #撮影地点まで移動
        print(step)
    except rospy.ROSInitException:
        print("except")
        pass

if step == 1:
    ##写真を撮る
    imgsub = take_image.ImageSubscriber()
    images = imgsub.get_image()
    step = 2
    print(step)

if step == 2:
    ##写真を切り取る
    cut_images = []
    for i in range(len(images)):
        # 画像読み込み
        image = cv2.imread(images[i])
        # img[top : bottom, left : right]
        # サンプルの切り出し
        cut_image = image[0 : 480, 320: 960]
        cut_images.append(cut_image)
    step = 3

if step == 3:
    ##手を振っている人を判断
    judges = []
    for j in range(len(cut_images)):
        judges.append(judgement(cut_images[j]))
    
    step = 4

if step == 4:
    if all(i == judges[0] for i in judges):#もし全ての要素が同じだったら

        if judges[0] == "left":
            simple_controller.turn_left(30)
            simple_controller.go_straight(3.0)
        elif judges[0] == "right":
            simple_controller.turn_right(30)
            simple_controller.go_straight(3.0)
        else :
            step = 3
    else:
        step = 3

"""