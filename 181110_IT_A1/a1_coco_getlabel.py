import os
import pandas as pd

img_path = '/home/herokwon/Git/PyTorch-YOLOv3/data/coco/images/val2014'
img_list = os.listdir(img_path)[0:1000]
pd_anns = pd.DataFrame()
for img in img_list:
    img_id = int(img[13:25])
    imgIds = coco.getImgIds(imgIds = img_id)
    img_ann = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    for annid in annIds:
        anns = coco.loadAnns(annid)
        ser_anns = pd.Series()
        ser_anns['file'] = img
        ser_anns['area'] = anns[0]['area']
        ser_anns['image_id'] = anns[0]['image_id']
        ser_anns['bbox0'] = anns[0]['bbox'][0]
        ser_anns['bbox1'] = anns[0]['bbox'][1]
        ser_anns['bbox2'] = anns[0]['bbox'][2]
        ser_anns['bbox3'] = anns[0]['bbox'][3]
        ser_anns['category_id'] = anns[0]['category_id']
        ser_anns['anns_id'] = anns[0]['id']
        ser_anns['img_height'] = img_ann['height']
        ser_anns['img_width'] = img_ann['width']
        pd_anns = pd_anns.append(ser_anns,ignore_index=True)