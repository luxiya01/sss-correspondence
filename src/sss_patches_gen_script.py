import numpy as np
import json
import os
from matplotlib import pyplot as plt
from data import train_test_split, plot_train_test_split
from sss_patches import SSSPatch, generate_sss_patches

data_dir = '/home/li/Documents/sss-correspondence/data/GullmarsfjordSMaRC20210209_ssh_annotations/survey2_better_resolution'
data_info_dict_path = f'{data_dir}/data_info_dict.json'
ref_sss_file_id = 'SSH-0170'
test_size = .2
annotations_dir = f'{data_dir}/annotations'
patch_size = 240
step_size = 40
patch_outpath = f'{data_dir}/patch_size{patch_size}_step{step_size}'

# Split into training and testing data and store the info to disk
split_dict = train_test_split(data_info_dict_path, ref_sss_file_id, test_size)
split_dict_file = os.path.join(data_dir, 'train_test_split.json')
with open(split_dict_file, 'w', encoding='utf-8') as f:
    json.dump(split_dict_file, f)
split_dict_savefig_path = os.path.join(data_dir, 'train_test_split.pdf')
plot_train_test_split(split_dict, split_dict_savefig_path)

for file_id, file_info in split_dict.items():
    generate_sss_patches(file_id, file_info['path'], file_info['valid_idx'],
                         annotations_dir, patch_size, step_size, patch_outpath)

#TODO: split patches into train and test dir
