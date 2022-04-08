"""
This module contains functions for splitting data into training and testing segments.

Exposed functions:
    - train_test_split(data_info_dict: str, ref_sss_file_id: str, test_size: float = .2,
                       random_seed: int = 0)
    - plot_train_test_split(split_dict: dict, savefig_path: str = None)
"""
from collections import defaultdict
import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from auvlib.bathy_maps.map_draper import sss_meas_data
from simple_shapes import Point2D, Line2D, get_intercepts_between_line1_normal_and_line2


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

    ref_sss_test_idx_start = rng.integers(
        low=0, high=nbr_train_idx) + ref_sss_valid_indices[0]
    ref_sss_test_idx_end = int(ref_sss_test_idx_start + nbr_test_idx)
    print(ref_sss_test_idx_start, ref_sss_test_idx_end)

    ref_sss_test_line = Line2D(
        Point2D.from_array(ref_sss_file.pos[ref_sss_test_idx_start]),
        Point2D.from_array(ref_sss_file.pos[ref_sss_test_idx_end]))

    split_dict = {}
    for file_id, info in data_info.items():
        file_split_dict = info
        if file_id == ref_sss_file_id:
            file_split_dict['test_idx'] = [(ref_sss_test_idx_start,
                                            ref_sss_test_idx_end)]
            file_split_dict['train_idx'] = [
                (ref_sss_valid_indices[0], ref_sss_test_idx_start),
                (ref_sss_test_idx_end, ref_sss_valid_indices[1])
            ]
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
                        (segment_start_idx, segment_end_idx))

                # Make sure that the test indices are consecutive
                else:
                    assert test_idx[-1] - test_idx[0] == len(test_idx) - 1

                    test_start_idx = min(test_idx)
                    test_end_idx = max(test_idx)
                    file_split_dict['test_idx'].append(
                        (test_start_idx, test_end_idx))
                    file_split_dict['train_idx'].append(
                        (segment_start_idx, test_start_idx))
                    file_split_dict['train_idx'].append(
                        (test_end_idx, segment_end_idx))
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
