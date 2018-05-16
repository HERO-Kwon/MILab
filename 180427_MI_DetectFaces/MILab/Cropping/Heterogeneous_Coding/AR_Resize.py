import numpy as np
import cv2
import math
import pandas as pd
from rotate import rotate, rotate_image, rotate_image2
from scipy.spatial import distance

name = 'm-013-1'
pts_path = '/home/han/Desktop/Photo-Sketch/CUFS/AR DB (123 faces)/photo/'+ name +'.dat'
img_path = '/home/han/Desktop/Hetero/AR/AR_Converted/AR_Selected/'+ name +'.bmp'
save_path = '/home/han/Desktop/Python/'+ name +'.bmp'

df = pd.read_csv(pts_path, header=None, delimiter=' ')
points = df.values
# print points.shape
# print points
# font = cv2.FONT_HERSHEY_SIMPLEX
# img = cv2.imread(img_path)
# for i, point in enumerate(points):
#     # print point
#     cv2.circle(img, tuple(point), 7, (0,0,255), -1)
#     cv2.putText(img, str(i), tuple(point), font, .7,(255,255,255),2)
# cv2.imshow('jaja', img)
# cv2.waitKey(0)

eye_left = (int(points[16][0]), int(points[16][1]))
eye_right = (int(points[18][0]), int(points[18][1]))
mid_point = ((eye_left[0]+eye_right[0])/2, (eye_left[1]+eye_right[1])/2)

dist = distance.euclidean(eye_left, eye_right)
ratio = abs(75./dist)

img = cv2.imread(img_path)

cv2.circle(img, eye_left, 7, (0,0,255), -1)
cv2.circle(img, eye_right, 7, (0,0,255), -1)
cv2.circle(img, mid_point, 7, (0,0,255), -1)
cv2.line(img,eye_left,eye_right,(255,0,0),2)

cv2.imwrite('test1.tif',img)

dy = eye_left[1] - eye_right[1]
dx = eye_left[0] - eye_right[0]
angle = np.arctan2(dy, dx)
angle_deg = np.rad2deg(angle)+180

img_rotation = rotate_image2(img, mid_point, angle_deg)
cv2.imwrite('test3.tif',img_rotation)

rotated_mid = rotate(mid_point, mid_point, angle)
cv2.imwrite('test4.tif',img_rotation)

# 75 pixels between two eyes
mid_point_new = (int(rotated_mid[0]*ratio), int(rotated_mid[1]*ratio))

res_size2 = cv2.resize(img_rotation, None, fx=ratio, fy =ratio, interpolation=cv2.INTER_AREA)
cv2.circle(res_size2, mid_point_new, 5, (0,255,255), -1)
cv2.imwrite('test5.tif',res_size2)
a = 115
b = 250-a
cropped_img = res_size2[mid_point_new[1]-a:mid_point_new[1]+b, mid_point_new[0]-100:mid_point_new[0]+100]
cv2.imshow('haha', cropped_img)
cv2.imwrite('test6.tif',cropped_img)
cv2.imwrite(save_path,cropped_img)
# cv2.waitKey(0)
