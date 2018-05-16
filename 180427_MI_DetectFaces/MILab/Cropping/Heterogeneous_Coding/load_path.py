import os, fnmatch
import shutil

# Move PNG files to their CASE Folder
def move_item(sources, destine):
    try:
        for source in sources:
            # destine = '/home/han/Desktop/Python/Hetero/searched/'
            shutil.move(source, destine)
    except shutil.Error:
        print "haha"
def search_items(image_path, selected):
    item_path_list = []
    for root, dirs, files in os.walk(image_path): #select a volume among volume list
        for name in files:
            # print name
            for item in selected:
                # item = item[:-1] + '.ppm'
                # print item
                # print item[:-5]
                # if name.find(item[:-1])!=-1:
                if fnmatch.fnmatch(name, item[:-1]):
                    item_path_list.append(os.path.join(root, name))
                else:
                    continue
    return item_path_list

def detect(image_path, selected):
    item_path_list = []
    for root, dirs, files in os.walk(image_path): #select a volume among volume list
        # print files
        for name in files:
            # print name
            for i, item in enumerate(selected):
                item = item[:-1] + '.ppm'
                # print item
                # print item[:-5]
                if name.find(item)!=-1:
                # if fnmatch.fnmatch(name, item):
                    # item_path_list.append(i)
                    print i
                else:
                    # print i
                    continue
    # return item_path_list







def volume_list(init_path):
    # List the volume
    volume = ['benign_*', 'cancer_*', 'normal_*']
    volume_path_list = []
    for root, dirs, files in os.walk(init_path):
        for name in dirs:
            for i in range(3):
                if fnmatch.fnmatch(name, volume[i]):
                    volume_path_list.append(os.path.join(root, name))

    return volume_path_list


def write_path_to_txt(path_to_save, item_path_list):
    """
    path to save: should set the name of file!!!
    """
    with open(path_to_save, 'w') as f:
        for path in item_path_list:
            f.write("%s\n" % path)

def replace_format(path_to_read, path_to_save):
    with open(path_to_read, 'r') as f:
        lines = f.readlines()
        with open(path_to_save, 'w') as g:
            for line in lines:
                line = line.replace("OVERLAY", "png")
                g.write("%s" % line)

def read_txt(path_to_read):
    with open(path_to_read, 'r') as f:
        lines = f.readlines()
    return lines
