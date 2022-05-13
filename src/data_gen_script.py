"""
Script to split a folder of sss_meas_data into SSSPatch, split the generated SSSPatches into
training and testing data, compute overlap matrix and store images.
"""
from argparse import ArgumentParser
from collections import defaultdict
import json
import os
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import generate_sss_patches, train_test_split, plot_train_test_split
from utils import (compute_overlap_between_two_rectangles,
                   get_sorted_patches_list, get_gt_overlap_between_two_patches)
from plot import (plot_ssspatch_with_annotated_keypoints,
                  plot_ssspatch_intensities,
                  plot_ssspatch_intensities_normalized,
                  plot_corresponding_keypoints)
from sss_patches import SSSPatch


def get_args():
    """Returns command line arguments for the data generator."""
    parser = ArgumentParser(
        description='Split a folder of sss_meas_data into SSSPatch.')
    parser.add_argument(
        '-d',
        '--data_dir',
        help=
        ('Path to the directory with sss_meas_data. The directory is expected to contain a'
         'data_info_dict.json file and a sub-dir named "annotations" with keypoint annotations'
         ),
        default=
        ('/home/li/Documents/sss-correspondence/data/GullmarsfjordSMaRC20210209_ssh_annotations'
         '/survey2_better_resolution/9-0169to0182-nbr_pings-1301_annotated/'))
    parser.add_argument(
        '-r',
        '--ref_sss',
        help='Reference SSS file ID used to for train and test split.',
        default='SSH-0170')
    parser.add_argument('-t',
                        '--test_size',
                        help='Portion of the data used for testing',
                        default=.1,
                        type=float)
    parser.add_argument('-p',
                        '--patch_size',
                        help='Number of pings in each SSSPatch',
                        default=240,
                        type=int)
    parser.add_argument(
        '-s',
        '--step_size',
        help='The number of pings each consecutive patch would differ.',
        default=40,
        type=int)
    parser.add_argument(
        '-o',
        '--overlap-thresh',
        help=
        'The amount of overlapping sss_hits required for two patches to be considered overlapping',
        default=.1,
        type=float)
    return parser.parse_args()


def generate_sss_patches_for_dir(data_info_dict_path: str,
                                 annotations_dir: str, patch_size: int,
                                 step_size: int, patch_outpath: str) -> int:
    """Generate SSSPatch from sss_meas_data in data_dir. Returns the total number of patches being
    generated."""

    print(
        f'Generate patchces for all sss_meas_data in {data_info_dict_path}...')
    patch_id_init_val = 0
    with open(data_info_dict_path, 'r', encoding='utf-8') as f:
        data_info_dict = json.load(f)
        for file_id, file_info in data_info_dict.items():
            new_patch_id_init_val = generate_sss_patches(
                file_id, file_info['path'], file_info['valid_idx'],
                annotations_dir, patch_size, step_size, patch_outpath,
                patch_id_init_val)
            patch_id_init_val = new_patch_id_init_val
    return patch_id_init_val


def split_patches_into_training_and_testing(data_info_dict_path: str,
                                            ref_sss_file_id: str,
                                            test_size: int, patch_outpath: str,
                                            train_outpath: str,
                                            test_outpath: str):
    print('Splitting sss_meas_data into training and testing data...')
    split_dict = train_test_split(data_info_dict_path, ref_sss_file_id,
                                  test_size)
    split_dict_file = os.path.join(patch_outpath, 'train_test_split.json')
    with open(split_dict_file, 'w', encoding='utf-8') as f:
        json.dump(split_dict, f)
    split_dict_savefig_path = os.path.join(patch_outpath,
                                           'train_test_split.pdf')
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

    print(
        'Moving the patches into their corresponding folder (train or test)...'
    )

    if os.path.exists(train_outpath):
        shutil.rmtree(train_outpath)
    if os.path.exists(test_outpath):
        shutil.rmtree(test_outpath)

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


def _update_overlap_matrix(patch1: SSSPatch, patch2: SSSPatch,
                           overlap_matrix: np.array) -> np.array:
    """Update the value of the given overlap_matrix for patch1 and patch2"""
    overlap = compute_overlap_between_two_rectangles(patch1.sss_hits_bounds,
                                                     patch2.sss_hits_bounds)
    overlap_matrix[patch1.patch_id, patch2.patch_id] = overlap
    overlap_matrix[patch2.patch_id, patch1.patch_id] = overlap
    return overlap_matrix


def _update_overlap_keypoints(patch1: SSSPatch, patch2: SSSPatch,
                              overlap_nbr_kps: np.array) -> np.array:
    """Update the value of the given overlap_nbr_kps for patch1 and patch2 with the number of shared
    keypoints between patch1 and patch2."""

    nbr_overlapping_keypoints = len(
        set(patch1.annotated_keypoints.keys()).intersection(
            patch2.annotated_keypoints.keys()))

    overlap_nbr_kps[patch1.patch_id,
                    patch2.patch_id] = nbr_overlapping_keypoints
    overlap_nbr_kps[patch2.patch_id,
                    patch1.patch_id] = nbr_overlapping_keypoints
    return overlap_nbr_kps


def _plot_patch(patch: SSSPatch, folder: str):
    """Construct three plots for the given patch and store them to folder.

    The three plots include:
        - SSSPatch with keypoint annotations
        - normalized intensity
        - raw intensity
    """
    ax = plot_ssspatch_with_annotated_keypoints(patch)
    plt.savefig(
        os.path.join(folder,
                     f'patch{patch.patch_id}_norm_intensity_with_keypoints'))

    plot_ssspatch_intensities(patch,
                              outpath=os.path.join(
                                  folder,
                                  f'patch{patch.patch_id}_intensity.png'))

    plot_ssspatch_intensities_normalized(
        patch, os.path.join(folder,
                            f'patch{patch.patch_id}_norm_intensity.png'))
    plt.close()


def _plot_correspondence(patch1: SSSPatch, patch2: SSSPatch, overlap: float,
                         folder: str):
    ax = plot_corresponding_keypoints(patch1, patch2)
    ax.set_title(
        f'Keypoints and correspondences between SSSPatches '
        f'patch_id={patch1.patch_id} (left) and '
        f'patch_id={patch2.patch_id} (right) with {overlap:.2f}% overlap')
    plt.savefig(
        os.path.join(folder,
                     f'patch{patch1.patch_id}_and_patch{patch2.patch_id}'))
    plt.close()


def plot_ssspatches_in_folder(folder: str, overlap_thresh: float = 0.1):
    """Assumes that there is an overlap.npz file in the folder"""
    patches_in_dir = get_sorted_patches_list(folder)
    overlap_matrix = np.load(os.path.join(folder,
                                          'overlap.npz'))['overlap_matrix']
    nbr_pairs_above_overlap_thresh = 0

    for i, patch1_path in tqdm(enumerate(patches_in_dir)):
        with open(patch1_path, 'rb') as f1:
            patch1 = pickle.load(f1)
            _plot_patch(patch1, folder)

            for patch2_path in patches_in_dir[i:]:
                patch2_id = int(patch2_path.split('/')[-1].split('_')[0][5:])
                overlap = overlap_matrix[patch1.patch_id, patch2_id]

                if overlap > overlap_thresh:
                    with open(patch2_path, 'rb') as f2:
                        patch2 = pickle.load(f2)
                        _plot_correspondence(patch1, patch2, overlap * 100,
                                             folder)
                    nbr_pairs_above_overlap_thresh += 1
    return nbr_pairs_above_overlap_thresh


def compute_overlap_matrix(
        patch_dir: str, total_num_patches: int) -> (np.array, np.array, dict):
    """Computes the overlap matrix and number of shared keypoints between all pairs of patches in
    patch_dir."""
    print(f'Computing overlap matrix for {patch_dir}...')
    overlap_matrix = np.zeros((total_num_patches, total_num_patches))
    overlap_nbr_kps = np.zeros_like(overlap_matrix)
    overlap_kps = defaultdict(dict)

    patches_in_dir = get_sorted_patches_list(patch_dir)

    for i, patch1_path in tqdm(enumerate(patches_in_dir)):
        with open(patch1_path, 'rb') as f1:
            patch1 = pickle.load(f1)
            overlap_kps[patch1.patch_id][
                'sorted_kps_hash'] = patch1.annotated_keypoints_sorted[0]
            overlap_kps[patch1.patch_id][
                'sorted_kps_pos'] = patch1.annotated_keypoints_sorted[1]

            for patch2_path in patches_in_dir[i:]:
                with open(patch2_path, 'rb') as f2:
                    patch2 = pickle.load(f2)

                    # ignore overlaps from patches that are created from the same sss_meas_data
                    if patch2.file_id == patch1.file_id:
                        continue

                    overlap_matrix = _update_overlap_matrix(
                        patch1, patch2, overlap_matrix)
                    overlap_nbr_kps = _update_overlap_keypoints(
                        patch1, patch2, overlap_nbr_kps)

                    overlap_kps[patch1.patch_id][
                        patch2.patch_id] = get_gt_overlap_between_two_patches(
                            patch1, patch2)
                    overlap_kps[patch2.patch_id][
                        patch1.patch_id] = get_gt_overlap_between_two_patches(
                            patch2, patch1)
    return overlap_matrix, overlap_nbr_kps, overlap_kps


def store_and_plot_overlap_matrix_and_kps(overlap_matrix: np.array,
                                          overlap_nbr_kps: np.array,
                                          overlap_kps, folder: str):
    fig, ax = plt.subplots()
    im0 = ax.imshow(overlap_matrix)
    ax.set_title('%Overlap in sss_hits between all SSSPatches')
    ax.set_ylabel('Patch ID')
    ax.set_xlabel('Patch ID')
    fig.colorbar(im0, ax=ax, orientation='vertical')
    plt.savefig(os.path.join(folder, 'overlap.pdf'))

    fig, ax = plt.subplots()
    im1 = ax.imshow(overlap_nbr_kps)
    ax.set_title('Number of shared keypoints between all SSSPatches')
    ax.set_xlabel('Patch ID')
    fig.colorbar(im1, ax=ax, orientation='vertical')
    plt.savefig(os.path.join(folder, 'overlap_nbr_kps.pdf'))

    plt.close('all')

    np.savez(os.path.join(folder, 'overlap'),
             overlap_matrix=overlap_matrix,
             overlap_nbr_kps=overlap_nbr_kps)

    with open(os.path.join(folder, 'overlap_kps.json'), 'w',
              encoding='utf-8') as f:
        json.dump(overlap_kps, f)


def _write_pairs_to_file(array: np.array, thresh: float, folder: str,
                         out_filename: str) -> int:
    indices_above_thresh = np.argwhere(array > thresh)
    nbr_pairs = indices_above_thresh.shape[0]
    patch_id_to_filename = lambda patch_id, suffix: f'patch{patch_id}_{suffix}.png'

    pairs_raw_intensity_str = []
    pairs_norm_intensity_str = []
    for i, j in indices_above_thresh:
        pairs_norm_intensity_str.append(
            f'{patch_id_to_filename(i, "norm_intensity")} {patch_id_to_filename(j, "norm_intensity")}'
        )
        pairs_raw_intensity_str.append(
            f'{patch_id_to_filename(i, "intensity")} {patch_id_to_filename(j, "intensity")}'
        )

    with open(os.path.join(folder, f'{out_filename}_norm_intensity.txt'),
              'w') as f:
        f.writelines('\n'.join(pairs_norm_intensity_str))
    with open(os.path.join(folder, f'{out_filename}_raw_intensity.txt'),
              'w') as f:
        f.writelines('\n'.join(pairs_raw_intensity_str))
    return nbr_pairs


def generate_overlap_pairs_txt(folder: str, overlap_thresh: float = .1):
    """Assumes there being an overlap.npz file in  the folder. Construct two txt files:
    - From the overlap_matrix, list all pairs of patces with ovelap > overlap_thresh and write the
      pairs to pairs_with_over_{overlap_thresh}_overlap.txt.
    - From the overlap_nbr_kps, list all pairs of patches sharing at least 1 keypoint and write the
      pairs to pairs_sharinig_kps.txt.
    """
    overlap_file = np.load(os.path.join(folder, 'overlap.npz'))
    nbr_pairs_above_overlap_thresh = _write_pairs_to_file(
        np.triu(overlap_file['overlap_matrix']), overlap_thresh, folder,
        f'pairs_with_over_{overlap_thresh}_overlap')
    nbr_pairs_sharing_kps = _write_pairs_to_file(
        np.triu(overlap_file['overlap_nbr_kps']), 0, folder,
        'pairs_sharinig_kps')
    print(
        f'In {folder}, {nbr_pairs_above_overlap_thresh} pairs have > {overlap_thresh} overlap, '
        f'{nbr_pairs_sharing_kps} pairs share at least one keypoint')
    return nbr_pairs_above_overlap_thresh, nbr_pairs_sharing_kps


def handle_folder(folder: str,
                  total_num_patches: int,
                  overlap_thresh: float = .1) -> int:
    """Compute overlap matrix, number of overlapping keypoints for patch pairs in the folder, as
    well as plotting the intensities and overlaps"""
    overlap_matrix, overlap_nbr_kps, overlap_kps = compute_overlap_matrix(
        folder, total_num_patches)

    store_and_plot_overlap_matrix_and_kps(overlap_matrix, overlap_nbr_kps,
                                          overlap_kps, folder)
    nbr_pairs_above_overlap_thresh = plot_ssspatches_in_folder(
        folder, overlap_thresh)
    generate_overlap_pairs_txt(folder, overlap_thresh)
    return nbr_pairs_above_overlap_thresh


def main():
    args = get_args()

    data_info_dict_path = f'{args.data_dir}/data_info_dict.json'
    annotations_dir = f'{args.data_dir}/annotations'
    patch_outpath = (
        f'{args.data_dir}/patch{args.patch_size}_step{args.step_size}'
        f'_test{args.test_size}_ref{args.ref_sss}')
    train_outpath = f'{patch_outpath}/train'
    test_outpath = f'{patch_outpath}/test'

    total_num_patches = 590
    #    total_num_patches = generate_sss_patches_for_dir(data_info_dict_path,
    #                                                     annotations_dir,
    #                                                    args.patch_size,
    #                                                    args.step_size,
    #                                                    patch_outpath)
    #   split_patches_into_training_and_testing(data_info_dict_path, args.ref_sss,
    #                                           args.test_size, patch_outpath,
    #                                           train_outpath, test_outpath)
    #    nbr_test_pairs = handle_folder(test_outpath, total_num_patches,
    #                                   args.overlap_thresh)
    nbr_train_pairs = handle_folder(train_outpath, total_num_patches,
                                    args.overlap_thresh)
    print(
        f'Number of train pairs > {args.overlap_thresh*100:.2f} overlap: {nbr_train_pairs}'
    )


#    print(
#        f'Number of test pairs > {args.overlap_thresh*100:.2f} overlap: {nbr_test_pairs}'
#    )

if __name__ == '__main__':
    main()
