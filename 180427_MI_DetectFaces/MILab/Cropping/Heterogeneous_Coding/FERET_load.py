import numpy as np
import pandas as pd
import sys, os, fnmatch

def FERET_load_points(folder_path):
    item_path_list = []
    for root, dirs, files in os.walk(folder_path): #select a volume among volume list
        # print files
        for points_name in files:
            image_name = points_name[:-5]
            
            # print name
    #         for i, item in enumerate(selected):
    #             if fnmatch.fnmatch(name, item[:-1]):
    #                 item_path_list.append(os.path.join(root, name))
    #             else:
    #                 continue
    # # return item_path_list

FERET_load_points('/home/han/Desktop/Hetero/FERET_photo_points')

# df = pd.read_csv('/home/han/Desktop/Python/00003fa010_930831.3pts', header=None, delimiter=' ')
# points = df.values
