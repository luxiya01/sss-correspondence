from dataclasses import dataclass
import os
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
    annotated_keypoints: list(str)
        The hashes of the annotated keypoints that reside in the current patch.
    """
    file_id: str
    start_idx: int
    end_idx: int
    pos: np.array
    rpy: np.array
    sss_waterfall_image: np.array
    sss_hits: np.array
    annotated_keypoints: list(str)

    @property
    def length(self):
        """Returns the number of pings included in the patch."""
        return self.end_idx - self.start_idx + 1


def generate_sss_patches(file_id: str, path: str, valid_idx: list[tuple],
                         patch_size: int, step_size: int):
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
    patch_size: int
        The number of pings to be included in each patch, i.e. the patch height.
        Note that the patch width is determined by the width of the sss_meas_data.
    step_size: int
        The number of pings each consecutive patch would differ.
    """
    sss_data = sss_meas_data.read_single(path)
    for (start_idx, end_idx) in valid_idx:
        pass
