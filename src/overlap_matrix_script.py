import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import compute_overlap_between_two_rectangles
from tqdm import tqdm

#TODO: refactor this script and sss_patches_gen_script
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
overlap_file = f'{patch_outpath}/overlap.npz'
overlap_plot_file = f'{patch_outpath}/overlap.pdf'
overlap_nbr_kps_plot_file = f'{patch_outpath}/overlap_nbr_kps.pdf'

all_files = [
    os.path.join(train_outpath, x) for x in os.listdir(train_outpath)
    if x.split('.')[-1] == 'pkl'
]
all_files.extend([
    os.path.join(test_outpath, x) for x in os.listdir(test_outpath)
    if x.split('.')[-1] == 'pkl'
])
all_files = sorted(all_files)
num_files = len(all_files)

print('Generate overlap matrix for training and testing patchces...')
overlap_matrix = np.zeros((num_files, num_files))
overlap_nbr_kps = np.zeros_like(overlap_matrix)
for i, patch1_path in tqdm(enumerate(all_files)):
    with open(patch1_path, 'rb') as f1:
        patch1 = pickle.load(f1)

        for j, patch2_path in tqdm(enumerate(all_files[i:])):
            with open(patch2_path, 'rb') as f2:
                patch2 = pickle.load(f2)
                # ignore overlaps from patches that are created from the same sss_meas_data
                if patch2.file_id == patch1.file_id:
                    continue

                overlap = compute_overlap_between_two_rectangles(
                    patch1.sss_hits_bounds, patch2.sss_hits_bounds)
                nbr_overlapping_keypoints = len(
                    set(patch1.annotated_keypoints.keys()).intersection(
                        patch2.annotated_keypoints.keys()))

                overlap_matrix[patch1.patch_id, patch2.patch_id] = overlap
                overlap_matrix[patch2.patch_id, patch1.patch_id] = overlap
                overlap_nbr_kps[patch1.patch_id,
                                patch2.patch_id] = nbr_overlapping_keypoints
                overlap_nbr_kps[patch2.patch_id,
                                patch1.patch_id] = nbr_overlapping_keypoints

    np.savez(overlap_file,
             overlap_matrix=overlap_matrix,
             overlap_nbr_kps=overlap_nbr_kps)

    fig, ax = plt.subplots()
    im0 = ax.imshow(overlap_matrix)
    ax.set_title('%Overlap in sss_hits between all SSSPatches')
    ax.set_ylabel('Patch ID')
    ax.set_xlabel('Patch ID')
    fig.colorbar(im0, ax=ax, orientation='vertical')
    plt.savefig(overlap_plot_file)

    fig, ax = plt.subplots()
    im1 = ax.imshow(overlap_nbr_kps)
    ax.set_title('Number of shared keypoints between all SSSPatches')
    ax.set_xlabel('Patch ID')
    fig.colorbar(im1, ax=ax, orientation='vertical')
    plt.savefig(overlap_nbr_kps_plot_file)

    plt.close('all')
