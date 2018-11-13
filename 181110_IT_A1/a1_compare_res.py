import pandas as pd
import os
import numpy as np

model_list = ['ssd_mobilenet_v1_coco_2017_11_17',
'faster_rcnn_inception_v2_coco_2018_01_28',
'faster_rcnn_resnet50_coco_2018_01_28',
'ssd_inception_v2_coco_2018_01_28',
'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
'df_res_yolov3']

label_path = '/home/herokwon/Git/PyTorch-YOLOv3/data/coco/labels/val2014'
label_list = os.listdir(label_path)
img_path = '/home/herokwon/Git/PyTorch-YOLOv3/data/coco/images/val2014'
img_list = os.listdir(img_path)[0:500]
coco_cats = pd.read_csv('coco_cats.csv')

res_truth = pd.DataFrame()
for model_name in model_list:
    res_model = pd.read_csv(model_name+'.csv')
    res_model = res_model[res_model.file.isin(img_list)]
    if 'area' in res_model.columns:
        res_model['box_size'] = res_model['area']
        df_merged = res_model.merge(coco_cats,left_on='cls',right_on='name',how='left')
        res_model['class'] = df_merged.id
    else:
        res_model['box_size'] = (res_model.xmax-res_model.xmin) * (res_model.ymax-res_model.ymin)

    idx_area = res_model.groupby('file')['box_size'].idxmax()
    res_area = res_model.loc[idx_area,]

    truth_label = pd.read_csv('val2014_truth.csv')
    truth_label = truth_label[truth_label.file.isin(img_list)]
    idx_truth_area = truth_label.groupby('file')['area'].idxmax()
    truth_area = truth_label.loc[idx_truth_area,]

    
    for filename in truth_area.file:
        cats = pd.Series()
        cats['truth_cat'] = truth_area[truth_area.file == filename]['category_id'].values[0]
        try: cats['res_cat'] = res_area[res_area.file == filename]['class'].values[0]
        except: cats['res_cat'] = 999.0
        cats['matches'] = (cats['truth_cat']==cats['res_cat'])
        cats['file'] = filename
        cats['model_name'] = model_name
        res_truth = res_truth.append(cats,ignore_index=True)


model_acc = res_truth.groupby('model_name')['matches'].mean()


import shutil, os
img_path = '/home/herokwon/Git/PyTorch-YOLOv3/data/coco/images/val2014'
img1000_path = '/home/herokwon/Git/PyTorch-YOLOv3/data/coco/images/val2014_img1000'
img_list = os.listdir(img_path)[0:1000]
for f in img_list:
    shutil.copy(img_path+'/'+f,img1000_path)