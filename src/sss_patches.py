"""
This module contains the definition of SSSPatch class, as well as functions to generate patches from
one sss_meas_data.

Exposes dataclass:
    - SSSPatch

Exposed functions:
    - generate_sss_patches(file_id: str, path: str, valid_idx: list[tuple],
                             annotations_dir: str, patch_size: int, step_size: int,
                             patch_outpath: str)
"""
from dataclasses import dataclass
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from auvlib.bathy_maps.map_draper import sss_meas_data
from utils import normalize_waterfall_image
from simple_shapes import Rectangle


@dataclass
class SSSPatch:
    """Class representing a patch of sss_meas_data.
    Note that index 0 denotes the first ping collected, i.e. the patch should be viewed as the data
    collected when the AUV traverses down from the top of the image.

    Parameters
    ----------
    patch_id: int
        Unique id for the SSSPatch.
    file_id: str
        The file id of the original sss_meas_data from which the SSSPatch is created.
    filename: str
        The name of the sss_meas_data file from which the patch is extracted.
    start_ping: int
        The start ping index of the sss patch from the sss_meas_data
    end_ping: int
        The end ping index of the sss patch from the sss_meas_data
    start_bin: int
        The start bin index of the sss patch from the sss_meas_data
    end_bin: int
        The end bin index of the sss patch from the sss_meas_data
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
            {keypoint hash: {"pos": (ping_idx, bin_idx), "annotation_file": path to the annotation
            file containing this keypoint}
        Note that the keypoint position in "pos" are given in the index of the patch.
        i.e. for a keypoint with (ping_idx, bin_idx), the same keypoint is found in the original
        sss_meas_data at (ping_idx+start_ping, bin_idx + start_bin)
    """
    patch_id: int
    file_id: str
    filename: str
    start_ping: int
    end_ping: int
    start_bin: int
    end_bin: int
    pos: np.array
    rpy: np.array
    sss_waterfall_image: np.array
    sss_hits: np.array
    is_port: bool
    annotated_keypoints: dict

    @property
    def nbr_pings(self):
        """Returns the number of pings included in the patch."""
        return self.end_ping - self.start_ping

    @property
    def nbr_bins(self):
        """Returns the number of bins included in the patch."""
        return self.end_bin - self.start_bin

    @property
    def height(self):
        """Alias to self.nbr_pings"""
        return self.nbr_pings

    @property
    def width(self):
        """Alias to self.nbr_bins"""
        return self.nbr_bins

    @property
    def keypoints_count(self):
        """Returns the total number of keypoints in the patch"""
        return len(self.annotated_keypoints)

    @property
    def sss_hits_bounds(self):
        """Returns a Rectangle class that represents the range of the sss_hits inside the patch."""
        #TODO: make pos threshold an input parameter
        thresh = 1.
        non_zero_sss_hits = self.sss_hits[np.logical_and(
            self.sss_hits[:, :, 0] > thresh, self.sss_hits[:, :, 1] > thresh)]
        x_min_hit = non_zero_sss_hits[:, 0].min()
        x_max_hit = non_zero_sss_hits[:, 0].max()
        y_min_hit = non_zero_sss_hits[:, 1].min()
        y_max_hit = non_zero_sss_hits[:, 1].max()
        return Rectangle(xmin=x_min_hit,
                         xmax=x_max_hit,
                         ymin=y_min_hit,
                         ymax=y_max_hit)

    def plot(self):
        """Returns a matplotlib figure showing the sss_waterfall_image and annotated_keypoints in
        the patch."""
        fig, ax = plt.subplots()
        ax.set_title(
            f'SSSPatch {self.patch_id}: pings ({self.start_ping}, {self.end_ping}), bins ({self.start_bin},'
            f'{self.end_bin}) from {self.file_id}')
        ax.imshow(normalize_waterfall_image(self.sss_waterfall_image),
                  extent=[
                      self.start_bin, self.end_bin, self.end_ping,
                      self.start_ping
                  ])
        ax.scatter([
            x['pos'][1] + self.start_bin
            for x in self.annotated_keypoints.values()
        ], [
            x['pos'][0] + self.start_ping
            for x in self.annotated_keypoints.values()
        ],
                   c='y')
        return ax

    def plot_corresponding_keypoints(self, p2):
        """Given another SSSPatch, stack the sss_waterfall_image horizontally, plot the
        annotated_keypoints and draw lines between the keypoints from the two image that correspond to
        one another.

        Parameters
        ----------
        p2: SSSPatch
            Another SSSPatch object
        """
        image = np.hstack([self.sss_waterfall_image, p2.sss_waterfall_image])

        # show the stacked sss_waterfall_image
        fig, ax = plt.subplots(figsize=(20, 5))
        plt.imshow(normalize_waterfall_image(image))

        # show keypoints from self
        ax.scatter([x['pos'][1] for x in self.annotated_keypoints.values()],
                   [x['pos'][0] for x in self.annotated_keypoints.values()],
                   c='r',
                   s=5)

        # show keypoints from p2
        bin_offset = self.sss_waterfall_image.shape[1]
        ax.scatter([
            x['pos'][1] + bin_offset for x in p2.annotated_keypoints.values()
        ], [x['pos'][0] for x in p2.annotated_keypoints.values()],
                   c='r',
                   s=5)

        # show corresponding keypoints
        overlapping_keypoints = set(
            self.annotated_keypoints.keys()).intersection(
                p2.annotated_keypoints.keys())
        for kp_hash in overlapping_keypoints:
            kp_self = self.annotated_keypoints[kp_hash]['pos']
            kp_p2 = p2.annotated_keypoints[kp_hash]['pos']
            ax.plot([kp_self[1], kp_p2[1] + bin_offset],
                    [kp_self[0], kp_p2[0]],
                    c='y',
                    linewidth=.8)
            ax.scatter([kp_self[1], kp_p2[1] + bin_offset],
                       [kp_self[0], kp_p2[0]],
                       c='y',
                       s=5)

        # show divider between self and p2
        ax.plot([bin_offset] * self.nbr_pings * 100,
                np.linspace(0, self.nbr_pings - 1, self.nbr_pings * 100),
                c='red')
        plt.axis('off')

        return ax


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
