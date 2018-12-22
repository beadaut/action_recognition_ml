"""
Load xml image file from UTKinect Action 3D dataset
"""
import re
import os
import cv2
import numpy as np
import xml.etree.ElementTree
import matplotlib.pyplot as plt
import glob

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def load_xml2image(xml_file, show_image=False):
    xml_root = xml.etree.ElementTree.parse(xml_file).getroot()
    data = list(xml_root[0][5].text.split('\n'))
    data = [x.strip('   ').split(' ') for x in data]

    large_list = []
    for small_list in data:
        large_list += small_list

    image = [int(x) for x in large_list[1:]]
    # image_max = np.max(image)
    image_max = 31800
    print("image max: ", image_max)
    image = np.reshape(np.array(image), [240,320])
    norm_image = image*(1.0/image_max)
    # print(norm_image)

    if show_image:
        cv2.imshow('image',norm_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return norm_image


# test for loading
# load_xml2image('sample_dataset/UTKinectAction3D_depth/s01_e01/depthImg300.xml')

# test for sorting
# files_list = glob.glob('d:/datasets/UTKinectAction3D_depth/s01_e01/*.xml')
# files_list.sort(key=natural_sort_key)
# for files in files_list:
#     print(files)

# print(len(files_list))


# general solution
dataset_dir = 'd:/datasets/UTKinectAction3D_depth'

directory_list = os.walk(dataset_dir)

for dirs in directory_list:
    file_dir = dirs[0]
    files_list = glob.glob(file_dir+'/*.xml')
    files_list.sort(key=natural_sort_key)
    for filename in files_list:
        print("\n", filename)
        image = load_xml2image(filename)
        cv2.imshow("Lidar Camera", image)
        key = cv2.waitKey(10)
        if key == 27:   # Esc key
            break
