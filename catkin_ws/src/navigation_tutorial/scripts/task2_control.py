from simple_control2 import simple_controller
from three_dimensions_tutorial.scripts.keypoint_rcnn import judgement
import rospy


step = 0

if __name__=='__main__':
    try:
        simple_controller.go_straight(1.0)
        simple_controller.turn_left(90)
        simple_controller.turn_right(90)
        step = 1 #撮影地点まで移動
    except rospy.ROSInitException:
        pass

if step == 1:
    ##写真を撮る
    images = []
    images.append()
    step = 2

if step == 2:
    ##写真を切り取る？
    cut_images = []
    for i in range(len(images)):
        cut_images.append(images[i])
    step = 3

if step == 3:
    ##手を振っている人を判断
    judge = []
    for j in range(len(cut_images)):
        judge.append(judgement(cut_images[j]))
    if judge == "None":
        pass
    else :
        step = 4

if step == 4:
    if judge == "left":
        simple_controller.turn_left(30)
        simple_controller.go_straight(3.0)
    else :
        simple_controller.turn_right(30)
        simple_controller.go_straight(3.0)