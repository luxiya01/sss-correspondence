import numpy as np
import json
import os
import pickle
import shutil
from matplotlib import pyplot as plt
from data import train_test_split, plot_train_test_split, generate_sss_patches
from sss_patches import SSSPatch

data_dir = '/home/li/Documents/sss-correspondence/data/GullmarsfjordSMaRC20210209_ssh_annotations/survey2_better_resolution/9-0169to0182-nbr_pings-1301_annotated/'
data_info_dict_path = f'{data_dir}/data_info_dict.json'
ref_sss_file_id = 'SSH-0170'
test_size = .1
annotations_dir = f'{data_dir}/annotations'
patch_size = 240
step_size = 40
patch_outpath = f'{data_dir}/patch{patch_size}_step{step_size}_test{test_size}_ref{ref_sss_file_id}'
train_outpath = f'{patch_outpath}/train'
test_outpath = f'{patch_outpath}/test'

print(f'Generate patchces for all sss_meas_data in the data_dir {data_dir}...')
patch_id_init_val = 0
with open(data_info_dict_path, 'r', encoding='utf-8') as f:
    data_info_dict = json.load(f)
    for file_id, file_info in data_info_dict.items():
        new_patch_id_init_val = generate_sss_patches(
            file_id, file_info['path'], file_info['valid_idx'],
            annotations_dir, patch_size, step_size, patch_outpath,
            patch_id_init_val)
        patch_id_init_val = new_patch_id_init_val

print('Splitting sss_meas_data into training and testing data...')
split_dict = train_test_split(data_info_dict_path, ref_sss_file_id, test_size)
split_dict_file = os.path.join(patch_outpath, 'train_test_split.json')
with open(split_dict_file, 'w', encoding='utf-8') as f:
    json.dump(split_dict, f)
split_dict_savefig_path = os.path.join(patch_outpath, 'train_test_split.pdf')
plot_train_test_split(split_dict, split_dict_savefig_path, show=False)

print(
    'Splitting patches into training and testing and recording pos extent for each patch...'
)
train_patches = {}
test_patches = {}
for patch_filename in os.listdir(patch_outpath):
    if patch_filename.split('.')[-1] != 'pkl':
        continue
    with open(os.path.join(patch_outpath, patch_filename), 'rb') as f:
        patch = pickle.load(f)
        test_indices = split_dict[patch.file_id]['test_idx']
        is_test = False
        for (t_start, t_end) in test_indices:
            if t_start <= patch.start_ping <= t_end or t_start <= patch.end_ping <= t_end:
                is_test = True
                continue
        if is_test:
            test_patches[patch_filename] = patch.sss_hits_bounds
        else:
            train_patches[patch_filename] = patch.sss_hits_bounds

print('Moving the patches into their corresponding folder (train or test)...')
os.makedirs(train_outpath)
os.makedirs(test_outpath)
for filename in train_patches:
    shutil.move(os.path.join(patch_outpath, filename), train_outpath)
for filename in test_patches:
    shutil.move(os.path.join(patch_outpath, filename), test_outpath)

print('Patch dataset generation complete.')
num_patches = len(train_patches) + len(test_patches)
print(f'Total number of patches: {num_patches}')
print(
    f'Number of training patches: {len(train_patches)}, {len(train_patches)/num_patches*100}% '
    f'of total number of patches')
print(
    f'Number of testing patches: {len(test_patches)}, {len(test_patches)/num_patches*100}% '
    f'of total number of patches')
