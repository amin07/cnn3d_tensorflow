'''
Created on Jul 18, 2018

@author: Amin

Given input directories of video this file does any preprocessing
Then prepare and return tf dataset iterators as part of dictionary
'''

import sys
import os
import argparse
import numpy as np
import cv2
from sklearn.model_selection import StratifiedShuffleSplit


'''
This method will prepare data array if not present
Then return the tf dataset dictionary to main
'''
def list_print(lst, upto=10):
    for l in lst[:upto]:
        print (l)

def get_file_paths(input_dir='', output_dir='', num_class=5):
    dir_list = os.listdir(input_dir)
    dir_list = dir_list[:num_class]
    paths_to_file = []      # contains whole path to files
    for dir in dir_list:
        curr_dir = os.path.join(input_dir, dir)
        file_names = os.listdir(curr_dir)
        paths_to_file += [(os.path.join(curr_dir, file_n)) for file_n in file_names]
    return paths_to_file

def to_numeric(lst):
    classes = []
    for l in lst:
        if l not in classes:
            classes.append(l)
    real2fake_dict = {r:f for f,r in enumerate(classes)} 
    numeric_label = [real2fake_dict[r] for r in lst]
    print (numeric_label, flush=True)
    return numeric_label
    
    
def prepare_data(file_paths=[], output_path='', height=50, width=60, T=20):
    data_array = []
    label_array = []    
    for fp in file_paths[:]:
        print (fp, flush=True)
        label_array.append((fp.split('/')[-2]))
        cap = cv2.VideoCapture(fp)
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frames = [x * nframes / T for x in range(T)]      
        framearray = []
        for i in range(T):  # sampling
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            frame = cv2.resize(frame, (height, width))
            framearray.append(frame)
        cap.release()
        data_array.append(framearray)
    label_array = to_numeric(label_array)
    label_array = np.array(label_array)
    label_array = np.expand_dims(label_array, axis=1)
    data_array = np.array(data_array)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=43)
    sss.get_n_splits(data_array, label_array)
    split_indexes = sss.split(data_array, label_array)
    train_index, test_index = next(split_indexes)
    np.savez(output_path, X_train=data_array[train_index], X_test=data_array[test_index], \
             Y_train = label_array[train_index], Y_test = label_array[test_index])
    
    
