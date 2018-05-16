import numpy as np
import cv2
import math
import pandas as pd
from rotate import rotate, rotate_image, rotate_image2
from scipy.spatial import distance

a = 106
b = 250-a
print a, b

name = '01006fa010_960627'
pts_path = '/home/han/Desktop/Hetero/photo_points/'+ name +'.3pts'
img_path = '/home/han/Desktop/Hetero/FERET/FERET_Selected_Image/'+ name +'.tif'
save_path = '/home/han/Desktop/Hetero/FERET/FERET_Cropped_Image/'+ name +'.tif'

df = pd.read_csv(pts_path, header=None, delimiter=' ')
points = df.values

eye_left = (int(points[0][0]), int(points[0][1]))
eye_right = (int(points[1][0]), int(points[1][1]))
mid_point = ((eye_left[0]+eye_right[0])/2, (eye_left[1]+eye_right[1])/2)

dist = distance.euclidean(eye_left, eye_right)
ratio = abs(75./dist)

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
res_size = cv2.resize(img, (512, 768), interpolation=cv2.INTER_AREA)
res_padded = cv2.copyMakeBorder(res_size, 0, 0, 100, 100, cv2.BORDER_CONSTANT) # Padding 100 pixels to left and right sides
cv2.imwrite('test1.tif',res_padded)

cv2.circle(res_padded, eye_left, 7, (0,0,255), -1)
cv2.circle(res_padded, eye_right, 7, (0,0,255), -1)
cv2.circle(res_padded, mid_point, 7, (0,0,255), -1)
cv2.line(res_padded,eye_left,eye_right,(0,0,255),2)


dy = eye_left[1] - eye_right[1]
dx = eye_left[0] - eye_right[0]
angle = np.arctan2(dy, dx)
angle_deg = np.rad2deg(angle)+180

img_rotation, dec, image_center = rotate_image(res_padded, angle_deg)
img_rotation = rotate_image2(res_padded, mid_point, angle_deg)
cv2.imwrite('test3.tif',img_rotation)

rotated_mid = rotate(mid_point, mid_point, angle)
cv2.imwrite('test4.tif',img_rotation)

# 75 pixels between two eyes
mid_point_new = (int(rotated_mid[0]*ratio), int(rotated_mid[1]*ratio))

res_size2 = cv2.resize(img_rotation, None, fx=ratio, fy =ratio, interpolation=cv2.INTER_AREA)
cv2.circle(res_size2, mid_point_new, 5, (0,0,255), -1)
cv2.imwrite('test5.tif',res_size2)

cropped_img = res_size2[mid_point_new[1]-a:mid_point_new[1]+b, mid_point_new[0]-100:mid_point_new[0]+100]
cv2.imshow('haha', cropped_img)
cv2.imwrite('test6.tif',cropped_img)
cv2.imwrite(save_path,cropped_img)
# cv2.waitKey(0)
