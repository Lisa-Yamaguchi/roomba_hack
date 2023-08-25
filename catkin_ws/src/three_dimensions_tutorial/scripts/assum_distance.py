import cv2
import numpy as np
import os
import glob

# チェスボード画像から算出したカメラパラメータを設定
fx = 1520.34696
fy = 1533.06074
Cx = 1637.74266
Cy = 1239.28645
mtx = np.array([[fx, 0, Cx],[0, fy, Cy],[0, 0, 1]])

# チェスボード画像から算出した歪係数を設定
k1 = 0.03117193
k2 = -0.0282226
p1 = 0.00190641
p2 = 0.00157755
k3 = 0.01867061
dist = np.array([[k1, k2, p1, p2, k3]])

# img_resizedフォルダー内のjpg画像を読み込む
images = glob.glob('*.jpg')
# Using the derived camera parameters to undistort the image
for filepath in images:

    img = cv2.imread(filepath)
    h,w = img.shape[:2]
    # Refining the camera matrix using parameters obtained by calibration
    # ROI:Region Of Interest(対象領域)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Method 1 to undistort the image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # undistort関数と同じ結果が返されるので、今回はコメントアウト(initUndistortRectifyMap()関数)
    # Method 2 to undistort the image
    # mapx,mapy=cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    # dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

    # 歪み補正した画像をimg_undistortフォルダへ保存
    cv2.imwrite('./img_undistort/undistort_' + str(filepath), dst)
    cv2.waitKey(0)