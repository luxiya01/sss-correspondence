from dataclasses import dataclass
from collections import defaultdict
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from auvlib.bathy_maps.map_draper import sss_meas_data


@dataclass
class SSSPatch:
    """Class representing a patch of sss_meas_data.
    Note that index 0 denotes the first ping collected, i.e. the patch should be viewed as the data
    collected when the AUV traverses down from the top of the image.

    Parameters
    ----------
    file_id: str
        The file id of the original sss_meas_data from which the SSSPatch is created.
    start_idx: int
        The start index of the sss patch from the sss_meas_data
    end_idx: int
        The end index of the sss patch from the sss_meas_data
    pos: np.array
        The AUV's dead-reckoning 3D positions when collecting each ping in the patch.
        shape = (self.length, 3)
    rpy: np.array
        The AUV's roll, pitch and yaw when collecting each ping in the the patch.
        shape = (self.length, 3)
    sss_waterfall_image: np.aray
        The waterfall image constructed by stacking the hit intensities of the sss pings included in
        the patch vertically. This corresponds to a segment of sss_waterfall_image in the original
        sss_meas_data.
        shape = (self.length, self.width)
    sss_hits: np.array
        The positions where each sss ping hits the mesh. This corresponds to a segment of the
        stacked (sss_waterfall_hits_X, sss_waterfall_hits_Y, sss_waterfall_hits_Z) in the original
        sss_meas_data.
        shape = (self.length, self.width, 3)
    is_port: bool
        True if the patch is extracted from the port side, False if it is from the starboard side
    annotated_keypoints: dict
        A dictionary of keypoint hahshes whose locations fall into the patch.
        The dictionary has the following structure:
            {annotation-file-path: list of keypoint hashes from this annotation file that fall
                                      into the current patch.}
    """
    file_id: str
    start_idx: int
    end_idx: int
    pos: np.array
    rpy: np.array
    sss_waterfall_image: np.array
    sss_hits: np.array
    is_port: bool
    annotated_keypoints: dict

    @property
    def length(self):
        """Returns the number of pings included in the patch."""
        return self.end_idx - self.start_idx

    @property
    def keypoints_count(self):
        """Returns the total number of keypoints in the patch"""
        return sum([
            len(kp_hashes) for kp_hashes in self.annotated_keypoints.values()
        ])


def generate_sss_patches(file_id: str, path: str, valid_idx: list[tuple],
                         annotations_dir: str, patch_size: int, step_size: int,
                         patch_outpath: str):
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

    patch_id = 0
    for (seg_start_idx, seg_end_idx) in valid_idx:
        start_idx = seg_start_idx
        end_idx = start_idx + patch_size

        while end_idx <= seg_end_idx:
            print(f'start_idx: {start_idx}, end_idx: {end_idx}')
            for start_bin, end_bin in [stbd_bins, port_bins]:
                print(f'\tstart_bin: {start_bin}, end_bin: {end_bin}')
                kps = get_annotated_keypoints_in_patch(path,
                                                       annotations_dir,
                                                       start_ping=start_idx,
                                                       end_ping=end_idx,
                                                       start_bin=start_bin,
                                                       end_bin=end_bin)
                is_port = (start_bin == port_bins[0])
                patch = SSSPatch(
                    file_id=file_id,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    pos=pos[start_idx:end_idx, :],
                    rpy=rpy[start_idx:end_idx, :],
                    sss_waterfall_image=sss_data.sss_waterfall_image[
                        start_idx:end_idx, start_bin:end_bin],
                    sss_hits=sss_hits[start_idx:end_idx, start_bin:end_bin],
                    is_port=is_port,
                    annotated_keypoints=kps)
                patch_filename = (
                    f'{file_id}_patch{patch_id}_pings_{start_idx}to{end_idx}_'
                    f'bins_{start_bin}to{end_bin}_isport_{is_port}.pkl')
                with open(os.path.join(patch_outpath, patch_filename),
                          'wb') as f:
                    pickle.dump(patch, f)

                patch_id += 1
            # Update start and end idx for the generation of a new SSSPatch
            start_idx += step_size
            end_idx = start_idx + patch_size


def get_annotated_keypoints_in_patch(path: str, annotations_dir: str,
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
            {annotation-file-path: list of keypoint hashes from this annotation file that fall
                                      into the current patch.}
    """
    patch_filename = os.path.basename(path)
    keypoints = defaultdict(list)

    for (dirpath, _, filenames) in os.walk(annotations_dir):
        for filename in filenames:
            if not 'correspondence_annotations' in filename:
                continue
            annotation_filepath = os.path.join(dirpath, filename)
            print(f'Looking into {annotation_filepath}')
            with open(annotation_filepath, 'r',
                      encoding='utf-8') as annotations_file:
                annotations = json.load(annotations_file)
                for kp_hash, annotations_dict in annotations.items():
                    if patch_filename not in annotations_dict.keys():
                        continue
                    kp_ping_nbr, kp_bin_nbr = annotations_dict[patch_filename]
                    if start_ping <= kp_ping_nbr < end_ping and start_bin <= kp_bin_nbr < end_bin:
                        keypoints[annotation_filepath].append(kp_hash)
    return keypoints
