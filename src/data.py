"""
This module contains functions for raw data manipulations, including splitting data into
training and testing segments, and generating SSSPatches from sss_meas_data.

Exposed functions:
    - train_test_split(data_info_dict: str, ref_sss_file_id: str, test_size: float = .2,
                       random_seed: int = 0)
    - plot_train_test_split(split_dict: dict, savefig_path: str = None)
    - generate_sss_patches(file_id: str, path: str, valid_idx: list[tuple], annotations_dir: str,
                         patch_size: int, step_size: int, patch_outpath: str,
                         patch_id_init_val: int = 0)
"""
from collections import defaultdict
import json
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from auvlib.bathy_maps.map_draper import sss_meas_data
from simple_shapes import Point2D, Line2D, get_intercepts_between_line1_normal_and_line2
from sss_patches import SSSPatch


def _pos_is_between_start_and_end_pos(pos: np.array, start_pos: Point2D,
                                      end_pos: Point2D):
    """Check whether the given pos param is between start_pos and end_pos

    Parameters
    ----------
    pos: np.array
        A 1D array with the AUV pos to be checked.
    start_pos: Point2D
        Point2D indicating the starting pos of the segment.
    end_pos: Point2D
        Point2D indicating the ending pos of the segment.

    Returns
    -------
        Boolean that indicates whether the given pos is in between start_pos and end_pos.
    """
    x_in_between = min(start_pos.x, end_pos.x) <= pos[0] <= max(
        start_pos.x, end_pos.x)
    y_in_between = min(start_pos.y, end_pos.y) <= pos[1] <= max(
        start_pos.y, end_pos.y)
    return x_in_between and y_in_between


def train_test_split(data_info_dict: str,
                     ref_sss_file_id: str,
                     test_size: float = .2,
                     random_seed: int = 0):
    """
    Given a data_info_dict json file, this function splits the sss_meas_data contained in the
    data_info_dict into training and test data.

    Parameters
    ----------
    data_info_dict: str
        The path to the json file with sss_meas_data info.
        The dict has the following structure:
            {
                file-id:
                    {
                        "path": path to sss_meas_data file
                        "filename": sss_meas_data filename (the last part of "path")
                        "valid_idx": list of tuples, each tuple indicates the start and end
                                     index/ping id of a valid segment of the sss_meas_data.
                                     (In this context, valid segment is a segment where the AUV is
                                     not turning too rapidly in a short period of time.)
                    }
            }
    ref_sss_file_id: str
        The file-id of the sss_meas_data that is to be used as reference data for the training and
        test split. Here the file-id refers to the file-id in the data_info_dict. This function
        requires choosing a sss_meas_data that only contains one segment of valid_idx, i.e. that
        doesn't include more than 1 AUV turn.
    test_size: float
        Float value between 0 and 1 indicating the proportion of the ref_sss_file_id used for
        testing.
    random_seed: int
        Controls the random number generator for reproducibility.

    Returns
    split_dict: dict
        Dictionary containing training and testing indices of each sss_meas_data
        The dict has the following structure:
            {
                file-id (same as param data_info_dict):
                    {
                        "path": same as in param data_info_dict
                        "filename": same as in param data_info_dict
                        "valid_idx": same as in param data_info_dict
                        "train_idx": list of tuples, each tuple indicates the start and end ping
                        id of a training segment.
                        "test_idx": list of tuples, each tuple indicates the start and end ping id
                        of a test segment.
                    }
            }
    -------
    """
    #TODO: refactor this function!
    rng = np.random.default_rng(random_seed)

    data_info = None
    with open(data_info_dict, 'r', encoding='utf-8') as data_info_file:
        data_info = json.load(data_info_file)

    ref_sss_file_info = data_info[ref_sss_file_id]
    if len(ref_sss_file_info['valid_idx']) > 1:
        raise ValueError(
            f'Please use an sss file with one single valid_idx segment as reference for train/test '
            f'split\n.'
            f'The provided reference sss file contains {len(ref_sss_file_info["valid_idx"])} '
            f'segments of valid indices.')

    ref_sss_file = sss_meas_data.read_single(ref_sss_file_info['path'])

    ref_sss_valid_indices = ref_sss_file_info['valid_idx'][0]
    nbr_valid_idx = ref_sss_valid_indices[1] - ref_sss_valid_indices[0]
    nbr_train_idx = np.floor((1 - test_size) * nbr_valid_idx)
    nbr_test_idx = nbr_valid_idx - nbr_train_idx

    ref_sss_test_idx_start = rng.integers(low=0, high=nbr_train_idx,
                                          dtype=int) + ref_sss_valid_indices[0]
    ref_sss_test_idx_end = int(ref_sss_test_idx_start + nbr_test_idx)
    print(ref_sss_test_idx_start, ref_sss_test_idx_end)

    ref_sss_test_line = Line2D(
        Point2D.from_array(ref_sss_file.pos[ref_sss_test_idx_start]),
        Point2D.from_array(ref_sss_file.pos[ref_sss_test_idx_end]))

    split_dict = {}
    for file_id, info in data_info.items():
        file_split_dict = info
        if file_id == ref_sss_file_id:
            file_split_dict['test_idx'] = [[
                ref_sss_test_idx_start, ref_sss_test_idx_end
            ]]
            file_split_dict['train_idx'] = [[
                ref_sss_valid_indices[0], ref_sss_test_idx_start
            ], [ref_sss_test_idx_end, ref_sss_valid_indices[1]]]
        else:
            sss_file = sss_meas_data.read_single(info['path'])
            file_split_dict['train_idx'] = []
            file_split_dict['test_idx'] = []
            for (segment_start_idx, segment_end_idx) in info['valid_idx']:
                line = Line2D(
                    Point2D.from_array(sss_file.pos[segment_start_idx]),
                    Point2D.from_array(sss_file.pos[segment_end_idx]))
                test_pos1, test_pos2 = get_intercepts_between_line1_normal_and_line2(
                    ref_sss_test_line, line)

                test_idx = []
                for idx, pos in zip(
                        range(segment_start_idx, segment_end_idx),
                        sss_file.pos[segment_start_idx:segment_end_idx]):
                    if _pos_is_between_start_and_end_pos(
                            pos, test_pos1, test_pos2):
                        test_idx.append(idx)

                # None of the pings have AUV pos in the test range -> all ping to train
                if len(test_idx) == 0:
                    file_split_dict['train_idx'].append(
                        [segment_start_idx, segment_end_idx])

                # Make sure that the test indices are consecutive
                else:
                    assert test_idx[-1] - test_idx[0] == len(test_idx) - 1

                    test_start_idx = min(test_idx)
                    test_end_idx = max(test_idx)
                    file_split_dict['test_idx'].append(
                        [test_start_idx, test_end_idx])
                    file_split_dict['train_idx'].append(
                        [segment_start_idx, test_start_idx])
                    file_split_dict['train_idx'].append(
                        [test_end_idx, segment_end_idx])
        split_dict[file_id] = file_split_dict
    return split_dict


def plot_train_test_split(split_dict: dict,
                          savefig_path: str = None,
                          show: bool = True):
    """Given a split_dict generated by the function train_test_split, plot the training and testing
    data in a single plot."""
    train_pos = defaultdict(list)
    test_pos = []
    for file_id, file_split_dict in split_dict.items():
        if len(file_split_dict['valid_idx']) <= 0:
            continue
        data = sss_meas_data.read_single(file_split_dict['path'])
        for (start_idx, end_idx) in file_split_dict['train_idx']:
            train_pos[file_id].extend(data.pos[start_idx:end_idx + 1])
        for (start_idx, end_idx) in file_split_dict['test_idx']:
            test_pos.extend(data.pos[start_idx:end_idx + 1])

    with sns.color_palette('Blues', n_colors=len(train_pos)):
        fig, ax = plt.subplots()
        for file_id, pos in train_pos.items():
            pos_array = np.array(pos)
            ax.plot(pos_array[:, 0],
                    pos_array[:, 1],
                    label=f'{file_id} training data')
        test_pos_array = np.array(test_pos)
        ax.plot(test_pos_array[:, 0],
                test_pos_array[:, 1],
                label='Testing data',
                c='r')

        ax.legend(markerscale=10, loc='lower right')
        ax.set_title('AUV positions included in training and test data')
        ax.set_xlabel('X position (easting)')
        ax.set_ylabel('Y position (northing)')
        if savefig_path is not None:
            plt.savefig(savefig_path)
    if show:
        plt.show()


def _get_annotated_keypoints_in_patch(path: str, annotations_dir: str,
                                      start_ping: int, end_ping: int,
                                      start_bin: int, end_bin: int) -> dict:
    """
    Returns a list of annotated keypoints found in the patch bounded by start and end pings and
    bins.

    Parameters
    ----------
    path: str
        File path to sss_meas_data file used for patch generation.
    annotations_dir: str
        Path to the directory containing subdirectories with annotations. The annotations are json
        files with names of 'correspondence_annotations_{file_ids}.json'
    start_ping: int
        The index of the first ping in the patch.
    end_ping: int
        The index of the first ping after the patch, i.e. the patch contains pings inside the slice
        of [start_ping:end_ping]
    start_bin: int
        The index of the first bin in the patch
    end_bin: int
        The index of the first bin after the patch, i.e. the patch contains bins inside the slice of
        [start_bin:end_bin]

    Returns
    -------
    keypoints: dict
        A dictionary of keypoint hahshes whose locations fall into the patch.
        The dictionary has the following structure:
            {keypoint hash: {"pos": (ping_idx, bin_idx), "annotation_file": path to the annotation
            file containing this keypoint}
        Note that the keypoint position in "pos" are given in the index of the patch.
        i.e. for a keypoint with (ping_idx, bin_idx), the same keypoint is found in the original
        sss_meas_data at (ping_idx+start_ping, bin_idx + start_bin)
    """
    patch_filename = os.path.basename(path)
    keypoints = {}

    for (dirpath, _, filenames) in os.walk(annotations_dir):
        for filename in filenames:
            if not 'correspondence_annotations' in filename:
                continue
            annotation_filepath = os.path.join(dirpath, filename)
            with open(annotation_filepath, 'r',
                      encoding='utf-8') as annotations_file:
                annotations = json.load(annotations_file)
                for kp_hash, annotations_dict in annotations.items():
                    if patch_filename not in annotations_dict.keys():
                        continue
                    kp_ping_nbr, kp_bin_nbr = annotations_dict[patch_filename]
                    if start_ping <= kp_ping_nbr < end_ping and start_bin <= kp_bin_nbr < end_bin:
                        keypoints[kp_hash] = {
                            "pos":
                            (kp_ping_nbr - start_ping, kp_bin_nbr - start_bin),
                            "annotation_file":
                            annotation_filepath
                        }
    return keypoints


def generate_sss_patches(file_id: str,
                         path: str,
                         valid_idx: list[tuple],
                         annotations_dir: str,
                         patch_size: int,
                         step_size: int,
                         patch_outpath: str,
                         patch_id_init_val: int = 0) -> int:
    """
    Generates patches of class SSSPatch from the sss_meas_data with the required specifications.

    Parameters
    ----------
    file_id: str
        File id of the sss_meas_data used for patch generation.
    path: str
        File path to sss_meas_data file used for patch generation.
    valid_idx: list[tuple]
        A list of tuples that indicates the ping ids/indices to be included in the patch
        creation. Each tuple contains a start and end index for a segment of valid pings
        for patch generation.
    annotations_dir: str
        Path to the directory containing subdirectories with annotations. The annotations are json
        files with names of 'correspondence_annotations_{file_ids}.json'
    patch_size: int
        The number of pings to be included in each patch, i.e. the patch height.
        Note that the patch width is determined by the width of the sss_meas_data.
    step_size: int
        The number of pings each consecutive patch would differ.
    patch_outpath: str
        The path to the directory where the newly generated SSSPatch objects should be stored.
    patch_id_init_val: int
        The initial value of patch_id. This value is set so that the patch_id for each SSSPatch
        is unique in one dataset.

    Returns
    -------
    patch_id: int
        The first unused patch_id.
    """
    sss_data = sss_meas_data.read_single(path)
    nbr_pings, nbr_bins = sss_data.sss_waterfall_image.shape
    nadir = int(nbr_bins / 2)
    stbd_bins = (0, nadir)
    port_bins = (nadir, nbr_bins)
    pos = np.array(sss_data.pos)
    rpy = np.array(sss_data.rpy)
    sss_hits = np.stack([
        sss_data.sss_waterfall_hits_X, sss_data.sss_waterfall_hits_Y,
        sss_data.sss_waterfall_hits_Z
    ],
                        axis=-1)

    if not os.path.isdir(patch_outpath):
        os.makedirs(patch_outpath)

    patch_id = patch_id_init_val
    for (seg_start_ping, seg_end_ping) in valid_idx:
        start_ping = seg_start_ping
        end_ping = start_ping + patch_size

        while end_ping <= seg_end_ping:
            for start_bin, end_bin in [stbd_bins, port_bins]:
                kps = _get_annotated_keypoints_in_patch(path,
                                                        annotations_dir,
                                                        start_ping=start_ping,
                                                        end_ping=end_ping,
                                                        start_bin=start_bin,
                                                        end_bin=end_bin)
                is_port = (start_bin == port_bins[0])
                patch = SSSPatch(
                    patch_id=patch_id,
                    file_id=file_id,
                    filename=os.path.basename(path),
                    start_ping=start_ping,
                    end_ping=end_ping,
                    start_bin=start_bin,
                    end_bin=end_bin,
                    pos=pos[start_ping:end_ping, :],
                    rpy=rpy[start_ping:end_ping, :],
                    sss_waterfall_image=sss_data.sss_waterfall_image[
                        start_ping:end_ping, start_bin:end_bin],
                    sss_hits=sss_hits[start_ping:end_ping, start_bin:end_bin],
                    is_port=is_port,
                    annotated_keypoints=kps)
                patch_filename = (
                    f'patch{patch_id}_{file_id}_pings_{start_ping}to{end_ping}_'
                    f'bins_{start_bin}to{end_bin}_isport_{is_port}.pkl')
                with open(os.path.join(patch_outpath, patch_filename),
                          'wb') as f:
                    pickle.dump(patch, f)

                patch_id += 1
            # Update start and end idx for the generation of a new SSSPatch
            start_ping += step_size
            end_ping = start_ping + patch_size
    return patch_id
