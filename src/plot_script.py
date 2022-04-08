import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#TODO: refactor this script
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

overlap_matrix = np.load(overlap_file)['overlap_matrix']
# Trim half of the overlap matrix to remove redundant information due to symmetry
overlap_matrix = np.triu(overlap_matrix)
max_patch_id = overlap_matrix.shape[0]


def plot_ssspatches_in_folder(folder: str):
    files = {
        int(x.split('_')[0][5:]): os.path.join(folder, x)
        for x in os.listdir(folder) if x.split('.')[-1] == 'pkl'
    }

    for patch1_id, patch1_path in tqdm(files.items()):
        with open(patch1_path, 'rb') as f1:
            patch1 = pickle.load(f1)
            patch1.plot()
            plt.savefig(os.path.join(folder, f'patch{patch1_id}'))
            plt.close()

            files_overlapping_with_patch1 = np.where(
                overlap_matrix[patch1_id, :] > 0)[0]

            for patch2_id in files_overlapping_with_patch1:
                overlap = overlap_matrix[patch1_id, patch2_id] * 100
                if patch2_id in files:
                    with open(files[patch2_id], 'rb') as f2:
                        patch2 = pickle.load(f2)
                        ax = patch1.plot_corresponding_keypoints(patch2)
                        ax.set_title(
                            f'Keypoints and correspondences between SSSPatches '
                            f'patch_id={patch1_id} (left) and '
                            f'patch_id={patch2_id} (right) with {overlap:.2f}% overlap'
                        )
                        plt.savefig(
                            os.path.join(
                                folder,
                                f'patch{patch1_id}_and_patch{patch2_id}'))
                        plt.close()


plot_ssspatches_in_folder(train_outpath)
plot_ssspatches_in_folder(test_outpath)
