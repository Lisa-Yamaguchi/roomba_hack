import copy
from typing import List

import cv2
import message_filters
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np


class DetectionDistance:
    def __init__(self):
        rospy.init_node('detection_distance', anonymous=True)

        # Publisher
        self.detection_result_pub = rospy.Publisher('/detection_result', Image, queue_size=10)

        # Subscriber
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 1.0).registerCallback(self.callback_rgbd)

        self.bridge = CvBridge()
        self.rgb_image, self.depth_image = None, None

        self.model = YOLO('yolov8n.pt')

    def callback_rgbd(self, data1, data2):
        cv_array = self.bridge.imgmsg_to_cv2(data1, 'bgr8')
        cv_array = cv2.cvtColor(cv_array, cv2.COLOR_BGR2RGB)
        self.rgb_image = cv_array

        cv_array = self.bridge.imgmsg_to_cv2(data2, 'passthrough')
        self.depth_image = cv_array

    def process(self):
        while not rospy.is_shutdown():
            if self.rgb_image is None:
                continue

            # inference
            tmp_image = copy.copy(self.rgb_image)

            results: List[Results] = self.model.predict(self.rgb_image, verbose=False)

            # plot bouding box
            for result in results:
                boxes = result.boxes.cpu().numpy()
                names = result.names
                if len(boxes.xyxy) == 0:
                    continue
                x1, y1, x2, y2 = map(int, boxes.xyxy[0][:4])
                cls_pred = boxes.cls[0]
                tmp_image = cv2.rectangle(tmp_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                tmp_image = cv2.putText(tmp_image, names[cls_pred], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                print(names[cls_pred], self.depth_image[cy][cx]/1000, "m")

            # masking
            width, height = tmp_image.size
            
            mask = np.zeros((width, height,3), np.uint8)
            mask = cv2.rectangle(mask, (x1,y1),(x2,y2),(255,255,255), -1)
            
            tmp_image_AND = cv2.bitwise_and(tmp_image, mask)

            # publish image
            tmp_image_AND = cv2.cvtColor(tmp_image_AND, cv2.COLOR_RGB2BGR)
            detection_result = self.bridge.cv2_to_imgmsg(tmp_image_AND, "bgr8")
            self.detection_result_pub.publish(detection_result)

        # 変換されたpointからmap座標系での位置を取得する（detection_result => map）
        def transform_position(ps_list, source_link='map', target_link='detection_result'):
            listener = tf.TransformListener()
            ps = PointStamped() 
            ps.header.frame_id = target_link
            ps.header.stamp = rospy.Time()
            ps.point.x = ps_list[0] # カメラローカルのx座標
            ps.point.y = ps_list[1] # カメラローカルのy座標
            ps.point.z = ps_list[2] # カメラローカルのz座標
            try:
                listener.waitForTransform(source_link, target_link, rospy.Time(), rospy.Duration(10))
                tf_ps = listener.transformPoint(source_link, ps)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                raise e
            return tf_ps # PointStamped型で返却されることに注意


if __name__ == '__main__':
    dd = DetectionDistance()
    try:
        dd.process()
    except rospy.ROSInitException:
        pass