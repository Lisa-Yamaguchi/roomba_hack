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


def judgement(image_path):
        #keypoint画像に変換（コピペ）       https://tech.fusic.co.jp/posts/2019-07-18-torchvision-keypoint-r-cnn/
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
        


        #画像から左右を決める       https://tony-mooori.blogspot.com/2015/10/python_27.html
       

        # 0 <= h <= 179 (色相)　OpenCVではmax=179なのでR:0(180),G:60,B:120となる
        # 0 <= s <= 255 (彩度)　黒や白の値が抽出されるときはこの閾値を大きくする
        # 0 <= v <= 255 (明度)　これが大きいと明るく，小さいと暗い
        # ここでは青色を抽出するので120±20を閾値とした
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
            
            # 輪郭抽出      https://pystyle.info/opencv-find-contours/
            contours,hierarchy = cv2.findContours(ex_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            # 面積を計算
            areas = np.array(list(map(cv2.contourArea,contours)))
        

            if len(areas) == 0 : #or np.max(areas) / (h*w) < AREA_RATIO_THRESHOLD:
                # 見つからなかったらNoneを返す
                print("the area is too small")
                return "None"
            else:
                # 面積が最大の塊の重心を計算し返す
                X = []
                Y = []
                for i, contour in enumerate(contours):
                # 重心の計算
                    m = cv2.moments(contour)
                    if m['m00']==0:
                        pass
                    else:
                        x,y= m['m10']/m['m00'] , m['m01']/m['m00']
                        print(f"Weight Center = ({x}, {y})")
                        # 座標を四捨五入
                        x, y = round(x), round(y)
                        X.append(x)
                        Y.append(y)
                first = Y[0]-Y[1]
                second = Y[-2]-Y[-1]
            
                if first^2>second^2:
                    return "left"
                else:
                    return "right"
                


        
        img = cv2.imread("source.png")

         # 位置を抽出
        
        pos_WRIST = find_specific_color(
                img,
                AREA_RATIO_THRESHOLD,
                LOW_COLOR_WRIST,
                HIGH_COLOR_WRIST
                )

        


"""rospy.init_node('imageforkeypoint')
imageforkeypoint = ImageForKeypoint()
rospy.spin()"""
