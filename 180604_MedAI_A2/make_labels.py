import os
import numpy as np
import pandas as pd

file_path = 'D:\\Matlab_Drive\\Data\\MedAI_A2\\AnnotationsByMD'
file_list = os.listdir(file_path)

train_labels=pd.DataFrame(columns=['filename','width','height','class','xmin','ymin','xmax','ymax'])

for file in file_list:
    print(file)
    data = pd.read_csv(os.path.join(file_path,file),header=None)
    for j in range(19):
        d_read = pd.Series()
        d_read['filename'] = file[0:3] + '.jpg'
        d_read['width'] = 645
        d_read['height'] = 800
        d_read['class'] = str(j+1)
        d_read['xmin'] = int(data.iloc[j][0]/3) - 65
        d_read['ymin'] = int(data.iloc[j][1]/3) - 80
        d_read['xmax'] = int(data.iloc[j][0]/3) + 65
        d_read['ymax'] = int(data.iloc[j][1]/3) + 80
        train_labels = train_labels.append(d_read,ignore_index=True)


train_labels.to_csv('train_labels.csv',index=False)