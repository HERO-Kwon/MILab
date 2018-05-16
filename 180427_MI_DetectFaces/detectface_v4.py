#!/usr/bin/env python

# Copyright 2015 Google, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Draws squares around detected faces in the given image."""

import argparse
import os
import pandas as pd
import numpy as np
# [START import_client_library]
from google.cloud import vision
# [END import_client_library]
from google.cloud.vision import types
#from PIL import Image, ImageDraw
import base64
from io import BytesIO
import glob
import cv2

# [START def_detect_face]
def detect_face(face_b64, max_results=4):
    """Uses the Vision API to detect faces in the given file.

    Args:
        face_file: A file-like object containing an image with faces.

    Returns:
        An array of Face objects with information about the picture.
    """
    # [START get_vision_service]
    client = vision.ImageAnnotatorClient()
    # [END get_vision_service]
    
    content = face_b64

    # To B64
    buffered = BytesIO()
    content.save(buffered,format='png')
    img64 = buffered.getvalue()

    image = types.Image(content=img64)

    return client.face_detection(image=image).face_annotations
# [END def_detect_face]


# [START def_highlight_faces]
def highlight_faces(image, faces, output_filename):
    """Draws a polygon around the faces, then saves to output_filename.

    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
      output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
    """
    im = Image.open(image)
    draw = ImageDraw.Draw(im)

    for face in faces:
        box = [(face.landmarks[1].position.x,face.landmarks[1].position.y)]
        draw.line(box + [box[0]], width=15, fill='#00ff00')

    im.save(output_filename)
# [END def_highlight_faces]

# max people in image
max_results = 1
res_ratio = 3

# For Windows
#file_path = 'E:\\HERO\\ND-NIVL\\D90IndoorCloseUpPNGs'
file_path = 'E:\\HERO\\ND-NIVL\\NIRAfterUpgradeResizedPNGs'
#file_path = 'D:\Matlab_Drive\Data\ORLDB'
# For Linux
# file_path = '/home/hero/Matlab_Drive/Data/ORLDB'

file_list = os.listdir(file_path)
file_list = [s for s in file_list if s.endswith(".png")]

# in csv
in_csv = pd.DataFrame([])
for counter, file in enumerate(glob.glob("*eyepos*.csv")):
    namedf = pd.read_csv(file)
    in_csv = in_csv.append(namedf)
    #print(file)

list_diff = list(set(file_list) - set(list(in_csv['file'].values)))

# for sampling
#file_list_s = file_list[0], file_list[int(len(file_list) * 0.2)],file_list[int(len(file_list) * 0.4)],file_list[int(len(file_list) * 0.6)],file_list[int(len(file_list) * 0.8)]
#file_list_s = list(file_list_s)

image_eyepos = pd.DataFrame(columns = ['file','left_eye','right_eye','box'])
image_raw = {}

i = 0
for file in list_diff:

    input_filename = os.path.join(file_path,file)
    with Image.open(input_filename) as image:
        #Image Resize
            try:
                size = int(image.size[0]/res_ratio),int(image.size[1]/res_ratio)
                image_resize = image.resize(size,Image.ANTIALIAS)
            
                faces = detect_face(image_resize, max_results)
                print(str(i),'File:',file,'/ Found {} face{}'.format(
                    len(faces), '' if len(faces) == 1 else 's'))
            except:
                faces=[]
    if len(faces) != 0:
        left_eye = faces[0].landmarks[0].position.x * res_ratio, faces[0].landmarks[0].position.y  * res_ratio, faces[0].landmarks[0].position.z * res_ratio
        right_eye = faces[0].landmarks[1].position.x * res_ratio, faces[0].landmarks[1].position.y * res_ratio, faces[0].landmarks[1].position.z * res_ratio
        box = [(vertex.x * res_ratio, vertex.y * res_ratio) for vertex in faces[0].bounding_poly.vertices]

        pos_read = {'file':file,'left_eye':left_eye,'right_eye':right_eye,'box':box}
#
        image_raw[file] = faces
        
    else : pos_read = {'file':file}
            
    image_eyepos = image_eyepos.append(pos_read,ignore_index=True)
    i += 1
        
    if ((i % 500 == 0) | (i == len(list_diff))):
        print('csv writing : ' + ('eyepos' + str(i) + '.csv'))
        image_eyepos.to_csv('nir_eyepos' + str(i) + '.csv')
        image_eyepos = pd.DataFrame(columns = ['file','left_eye','right_eye','box'])


# chk sample
# [START def_highlight_faces]
def highlight_faces(df, i, output_filename):

    image_location = cwd+'\\nir\\'+df.file[i]
    
    im = Image.open(image_location)
    draw = ImageDraw.Draw(im)

    eye_left = literal_eval(df.left_eye[i])[0],literal_eval(df.left_eye[i])[1]
    eye_right = literal_eval(df.right_eye[i])[0],literal_eval(df.right_eye[i])[1]
    #box = literal_eval(df.box)
    
    draw.line([eye_left,eye_right],width=20,fill=255)
    #draw.line(box + [box[0]], width=15, fill='#00ff00')

    #im.show()
    im.save(output_filename)
# [END def_highlight_faces]#


for i in range(0,5) :
    if isinstance(eyepos_nir.left_eye[i],str):
        highlight_faces(eyepos_nir,i,'res_nir'+str(i)+'.png')
    else: 
        pass


# in csv
in_csv = pd.DataFrame([])
for counter, file in enumerate(glob.glob("*eyepos*.csv")):
    namedf = pd.read_csv(file)
    in_csv = in_csv.append(namedf)
    #print(file)


# Cropping

import sys
source_dir = r'C:\Users\herok\OneDrive\4_Sources\MI_Got\Cropping\Heterogeneous_Coding'
sys.path.append(source_dir)
from rotate import rotate,rotate_image2
import ast
from scipy.spatial import distance

i=0
corrupt_file = []

for file in in_csv['file'].values:
    df = in_csv[in_csv['file']==file].iloc[0]

    points = df[['left_eye','right_eye']].values
    eye_left = ast.literal_eval(points[0])[:2]
    eye_right = ast.literal_eval(points[1])[:2]

    mid_point = (eye_left[0] + eye_right[0]) / 2, (eye_left[1] + eye_right[1]) / 2
    dist = distance.euclidean(eye_left,eye_right)
    ratio = abs(75./dist)

    image_name = file.replace('.png','_sz1.png')

    img_path = file_path +'\\'+ file
    
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        dy = eye_left[1] - eye_right[1]
        dx = eye_left[0] - eye_right[0]
        angle = np.arctan2(dy, dx)
        angle_deg = np.rad2deg(angle)+180

        img_rotation = rotate_image2(img, mid_point, angle_deg)
        rotated_mid = rotate(mid_point, mid_point, angle)
        mid_point_new = (int(rotated_mid[0]*ratio), int(rotated_mid[1]*ratio))
        res_size2 = cv2.resize(img_rotation, None, fx=ratio, fy =ratio, interpolation=cv2.INTER_AREA)


        crop_size = (200,250)

        crop_y1 = mid_point_new[1] - np.min([115,mid_point_new[1]])
        crop_y2 =  crop_size[1] + crop_y1
        crop_x1 = mid_point_new[0] - np.min([100,mid_point_new[0]])
        crop_x2 = crop_size[0] + crop_x1

        cropped_img = res_size2[crop_y1:crop_y2,crop_x1:crop_x2]


        save = file_path + '\\sz1\\' + image_name
        cv2.imwrite(save,cropped_img)
    except:
        corrupt_file.append(file)


    curr_percent = 100.0 * i / len(in_csv)
    if (curr_percent % 5)==0.0 : print(int(curr_percent),end="|")
    
    i += 1

pd.DataFrame(corrupt_file).to_csv("corrupt_file.csv")