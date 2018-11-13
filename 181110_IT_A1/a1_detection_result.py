import pandas as pd
import os

#detected_obj = pd.Series(columns=['file','num','class','score','ymin', 'xmin', 'ymax', 'xmax'])
detected_obj = pd.Series()
detected_objs = pd.DataFrame()

img_path = '/home/herokwon/Git/PyTorch-YOLOv3/data/coco/images/val2014'
img_list = os.listdir(img_path)[0:5]
for file in img_list:
    image = Image.open(img_path+'/'+file)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    detected_obj = pd.Series()
    detect_num = output_dict['num_detections']
    for i in range(detect_num):
        detected_obj['file'] = file
        detected_obj['num']=i
        detected_obj['class']=output_dict['detection_classes'][i]
        detected_obj['score']=output_dict['detection_scores'][i]
        detected_obj['ymin']=output_dict['detection_boxes'][i][0]
        detected_obj['xmin']=output_dict['detection_boxes'][i][1]
        detected_obj['ymax']=output_dict['detection_boxes'][i][2]
        detected_obj['xmax']=output_dict['detection_boxes'][i][3]

        detected_objs = detected_objs.append(detected_obj,ignore_index=True)
detected_objs.to_csv(MODEL_NAME+'.csv')