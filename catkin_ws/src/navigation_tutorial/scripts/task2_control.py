#!/usr/bin/env python3


from simple_control2 import SimpleController 
from keypoint_rcnn import judgement
import take_image
import rospy
import cv2


step = 0

class TaskManager:
    def __init__(self):
        rospy.init_node('task_manager')

    def main():
        
if __name__ == '__main__':
    tm = TaskManager()
    tm.main()
        

if __name__=='__main__':
    try:
        simple_controller = SimpleController()
        simple_controller.go_straight(1.0)
        simple_controller.turn_left(80)
        simple_controller.go_straight(1.0)
        simple_controller.turn_left(80)
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