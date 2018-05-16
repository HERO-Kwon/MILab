import numpy as np
import cv2
import math
import pandas as pd
from rotate import rotate, rotate_image2
from scipy.spatial import distance
import sys, os, fnmatch

points_folder_path = '/home/han/Desktop/Photo-Sketch/CUFS/CUHK DB (188 faces)/Training (88 faces)/sketch_points/'
image_folder_path = '/home/han/Desktop/Photo-Sketch/CUFS/CUHK DB (188 faces)/Training (88 faces)/sketch/'
save_path = '/home/han/Desktop/Hetero/CUHK/Train_Sketch/'

for root, dirs, files in os.walk(points_folder_path): #select a volume among volume list
    # print files
    for points_name in files:
        print points_name
        points_path = points_folder_path+str(points_name)
        df = pd.read_csv(points_path, header=None, delimiter=' ')

        points = df.values
        eye_left = (int(points[16][0]), int(points[16][1]))
        eye_right = (int(points[18][0]), int(points[18][1]))
        mid_point = ((eye_left[0]+eye_right[0])/2, (eye_left[1]+eye_right[1])/2)

        dist = distance.euclidean(eye_left, eye_right)
        ratio = abs(75./dist)

        image_name = points_name[:-4] + '.jpg'

        img_path = image_folder_path+str(image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # cv2.circle(img, eye_left, 7, (0,0,255), -1)
        # cv2.circle(img, eye_right, 7, (0,0,255), -1)
        # cv2.circle(img, mid_point, 7, (0,0,255), -1)

        # res_size = cv2.resize(img, (512, 768), interpolation=cv2.INTER_AREA)

        dy = eye_left[1] - eye_right[1]
        dx = eye_left[0] - eye_right[0]
        angle = np.arctan2(dy, dx)
        angle_deg = np.rad2deg(angle)+180
        img_rotation = rotate_image2(img, mid_point, angle_deg)

        rotated_mid = rotate(mid_point, mid_point, angle)

        # 75 pixels between two eyes
        mid_point_new = (int(rotated_mid[0]*ratio), int(rotated_mid[1]*ratio))

        res_size2 = cv2.resize(img_rotation, None, fx=ratio, fy =ratio, interpolation=cv2.INTER_AREA)
        # cv2.circle(res_size2, mid_point_new, 5, (0,255,255), -1)

        cropped_img = res_size2[mid_point_new[1]-115:mid_point_new[1]+135, mid_point_new[0]-100:mid_point_new[0]+100]

        save = save_path + image_name
        cv2.imwrite(save,cropped_img)




# cv2.imshow('haha', res_size2)
# cv2.waitKey(0)
