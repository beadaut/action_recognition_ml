"""
Load xml image file from UTKinect Action 3D dataset
"""
import re
import os
# import cv2
import json
import numpy as np
import xml.etree.ElementTree
import matplotlib.pyplot as plt
import glob

from utils.config import cfg

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
    # print("image max: ", image_max)
    image = np.reshape(np.array(image), [240,320])
    norm_image = image*(1.0/image_max)
    # print(norm_image)

    if show_image:
        cv2.imshow('image',norm_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return norm_image


def make_input_bundle(filenames_list):
    input_bundle = []
    for file_i in filenames_list:
        input_i = load_xml2image(file_i)
        input_bundle.append(input_i)
    
    # reverse the order of the elements in the list
    reversed(input_bundle)
    return np.array(input_bundle)

# test for loading
# load_xml2image('sample_dataset/UTKinectAction3D_depth/s01_e01/depthImg300.xml')

# test for sorting
# files_list = glob.glob('d:/datasets/UTKinectAction3D_depth/s01_e01/*.xml')
# files_list.sort(key=natural_sort_key)
# for files in files_list:
#     print(files)

# print(len(files_list))


# general solution
# dataset_dir = 'd:/datasets/UTKinectAction3D_depth' # from ext drive
dataset_dir = '/media/tjosh/vault/UTKinectAction3D_depth_train'




directory_list = os.walk(dataset_dir)

actions = {"walk":0, "sitDown":1, "standUp":2, "pickUp":3, "carry":4, 
    "throw":5, "push":6, "pull":7, "waveHands":8,"clapHands":9}

label2actions = {0:"walk", 1:"sitDown", 2:"standUp", 3:"pickUp", 4:"carry", 
    5:"throw", 6:"push", 7:"pull", 8:"waveHands",9:"clapHands"}

with open('/media/tjosh/vault/UTKinectAction3D_depth_train/actionLabel.json') as f:
    json_file = json.load(f)
    # print(json_file.keys())
action_buffer = []
action_buffer_size = cfg.num_frames
action_counts = 0
for dirs in directory_list:
    file_dir = dirs[0]
    files_list = glob.glob(file_dir+'/*.xml')
    files_list.sort(key=natural_sort_key)
    previous_action = None
    for filename in files_list:
        print("filename: ", filename)
        # subject_key = filename.split('\\')[-2] # on windows
        subject_key = filename.split('/')[-2]
        frame_no = int(re.split('depthImg|.xml',filename)[-2])
        subject_actions = json_file[subject_key]
        
        action_label = None
        for action in subject_actions:
            action_frame = subject_actions[action]
            # print("action: {}, action frame: {}.".format(actions[action], action_frame))
            if frame_no >= action_frame[0] and frame_no <= action_frame[1]:
                action_label = actions[action]
        else:
            if action_label == None:
                print("No action label!")
                continue
        print("\n{}, frame {}.".format(subject_key, frame_no))
        save_name = filename.split('.')[-2]+'_'+str(action_label)
        save_name = save_name.replace("UTKinectAction3D_depth_train", "UTKinectAction3D_train_npy_"+str(cfg.num_frames))
        save_dir = save_name.split('depthImg')[-2]
        
        if previous_action != action_label:
            action_counts += 1
            previous_action = action_label
            # this means a new action starts
            action_buffer = []

            # report the evaluation for the last action and start a new evaluation for the next action
            
        if not os.path.exists(save_dir):
            print("does not exist!")
            os.makedirs(save_dir)
        
        print("Save name: ", save_name)
        # print("Save dir: ", save_dir)
        print("action label: ", action_label)
        print("action counts: ", action_counts)
        action_buffer.append(filename)

        if len(action_buffer) <= action_buffer_size:
          continue


        if len(action_buffer) > action_buffer_size:
            action_buffer.pop(0)
        
        # print("action buffer: ", action_buffer)

        # function to bundle dataset input here:
        input_bundle = make_input_bundle(action_buffer)
        print(np.shape(input_bundle))

        # save the dataset bundle here:
        np.save(save_name, input_bundle)

        # for action_file in action_buffer:
        #     print(action_file)
        # print("action label: ", actions[0])

        # image = load_xml2image(filename)
        # cv2.imshow("Lidar Camera", image)
        # key = cv2.waitKey(10)
        # if key == 27:   # Esc key
        #     break


# for dirs in directory_list:
#     file_dir = dirs[0]
#     subject_key = file_dir.split('\\')[-1]
#     if subject_key[0] != 's':
#         continue
#     print(subject_key)
