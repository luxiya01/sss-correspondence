"""
Script to split a folder of sss_meas_data into SSSPatch, split the generated SSSPatches into
training and testing data, compute overlap matrix and store images.
"""
from argparse import ArgumentParser
import json
import os
import pickle
import shutil
from data import generate_sss_patches, train_test_split, plot_train_test_split


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
    return parser.parse_args()


def generate_sss_patches_for_dir(data_info_dict_path: str,
                                 annotations_dir: str, patch_size: int,
                                 step_size: int, patch_outpath: str):
    """Generate SSSPatch from sss_meas_data in data_dir."""

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


def main():
    args = get_args()

    data_info_dict_path = f'{args.data_dir}/data_info_dict.json'
    annotations_dir = f'{args.data_dir}/annotations'
    patch_outpath = (
        f'{args.data_dir}/patch{args.patch_size}_step{args.step_size}'
        f'_test{args.test_size}_ref{args.ref_sss}')
    train_outpath = f'{patch_outpath}/train'
    test_outpath = f'{patch_outpath}/test'

    generate_sss_patches_for_dir(data_info_dict_path, annotations_dir,
                                 args.patch_size, args.step_size,
                                 patch_outpath)
    split_patches_into_training_and_testing(data_info_dict_path, args.ref_sss,
                                            args.test_size, patch_outpath,
                                            train_outpath, test_outpath)


if __name__ == '__main__':
    main()
