import os, fnmatch
import shutil
import numpy as np
from load_path import search_items, move_item, detect



path_1 = '/home/han/Desktop/Photo-Sketch/CUFS/AR/AR'


files = '/home/han/Desktop/Python/Hetero/AR_file_names_of_photos.txt'


with open(files) as f:
    data = f.readlines()

# print data
destine = '/home/han/Desktop/Python/Hetero/AR'
#
paht_1_searched = search_items(path_1, data)
move_item(paht_1_searched, destine)
