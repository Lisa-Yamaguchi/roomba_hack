#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image as pilImage
from chainercv.visualizations import vis_bbox, vis_point

import rospy
from sensor_msgs.msg import Image as rosImage

import cv2, cv_bridge


class ImageForKeypoint:

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        #cv2.nameWindow("window", 1)
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', rosImage, self.image_callback)

    #画像を取得する
    def image_callback(self, msg):

        #画像をopenCVに渡してcv2にする
        image_cv2 = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        #cv2.imshow("window", image_cv2)
        #cv2.waitKey(3)

        #jpg画像として保存
        cv2.imwrite('save.jpg', image_cv2)

        #keypoint画像に変換（コピペ）
        image_path = 'save.jpg'
        image_keypoint = pilImage.open(image_path).convert('RGB')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        model = model.to(device)
        model.eval()

        image_tensor = torchvision.transforms.functional.to_tensor(image_keypoint)
        x = [image_tensor.to(device)]


        prediction = model(x)[0]

        bboxes_np = prediction['boxes'].to(torch.int16).cpu().numpy()
        labels_np = prediction['labels'].byte().cpu().numpy()
        scores_np = prediction['scores'].cpu().detach().numpy()
        keypoints_np = prediction['keypoints'].to(torch.int16).cpu().numpy()
        keypoints_scores_np = prediction['keypoints_scores'].cpu().detach().numpy()

        bboxes = []
        labels = []
        scores = []
        keypoints = []

        for i, bbox in enumerate(bboxes_np):
            score = scores_np[i]
            if score < 0.8:
                continue

            label = labels_np[i]
            keypoint = keypoints_np[i]
            
            bboxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])
            labels.append(label - 1)
            scores.append(score)
            keypoints.append(keypoint)

        bboxes = np.array(bboxes)
        labels = np.array(labels)
        scores = np.array(scores)
        keypoints = np.array(keypoints)

        points = np.dstack([keypoints[:, :, 1], keypoints[:, :, 0]])

        img = image_tensor.mul(255).byte().numpy()
        vis_bbox(img, bboxes, labels, scores, label_names=('person',))
        vis_point(img, points)

        plt.savefig("source.png") 
        #plt.show()


        #画像から左右を決める
       

        # 0 <= h <= 179 (色相)　OpenCVではmax=179なのでR:0(180),G:60,B:120となる
        # 0 <= s <= 255 (彩度)　黒や白の値が抽出されるときはこの閾値を大きくする
        # 0 <= v <= 255 (明度)　これが大きいと明るく，小さいと暗い
        # ここでは青色を抽出するので120±20を閾値とした
        LOW_COLOR_ELBOW = np.array([55, 155, 205])
        HIGH_COLOR_ELBOW = np.array([65, 255, 255])
        LOW_COLOR_WRIST = np.array([80, 155, 205])
        HIGH_COLOR_WRIST = np.array([95, 255, 255])


        # 抽出する青色の塊のしきい値
        AREA_RATIO_THRESHOLD = 0.001

        def find_specific_color(frame,AREA_RATIO_THRESHOLD,LOW_COLOR,HIGH_COLOR):
            """
            指定した範囲の色の物体の座標を取得する関数
            frame: 画像
            AREA_RATIO_THRESHOLD: area_ratio未満の塊は無視する
            LOW_COLOR: 抽出する色の下限(h,s,v)
            HIGH_COLOR: 抽出する色の上限(h,s,v)
            """
            # 高さ，幅，チャンネル数
            h,w,c = frame.shape

            # hsv色空間に変換
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            
            # 色を抽出する
            ex_img = cv2.inRange(hsv,LOW_COLOR,HIGH_COLOR)
            cv2.imwrite("ex_result.png", ex_img)


            # 輪郭抽出
            contours,hierarchy = cv2.findContours(ex_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            # 面積を計算
            areas = np.array(list(map(cv2.contourArea,contours)))
            print(areas)

            if len(areas) == 0 : #or np.max(areas) / (h*w) < AREA_RATIO_THRESHOLD:
                # 見つからなかったらNoneを返す
                print("the area is too small")
                return None
            else:
                # 面積が最大の塊の重心を計算し返す
                max_idx = np.argmax(areas)
                max_area = areas[max_idx]
                result = cv2.moments(contours[max_idx])
                x = int(result["m10"]/result["m00"])
                y = int(result["m01"]/result["m00"])
                return (x,y)

        
        img = cv2.imread("source.png")

            # 位置を抽出
        pos_ELBOW = find_specific_color(
                img,
                AREA_RATIO_THRESHOLD,
                LOW_COLOR_ELBOW,
                HIGH_COLOR_ELBOW
                )
        
        pos_WRIST = find_specific_color(
                img,
                AREA_RATIO_THRESHOLD,
                LOW_COLOR_WRIST,
                HIGH_COLOR_WRIST
                )

        if pos_ELBOW is not None:
            
            print(pos_ELBOW)
            
        if pos_WRIST is not None:
            
            print(pos_WRIST)



rospy.init_node('imageforkeypoint')
imageforkeypoint = ImageForKeypoint()
rospy.spin()
