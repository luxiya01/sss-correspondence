"""
This module contains the definition of SSSPatch class.

Exposes dataclass:
    - SSSPatch
"""
from dataclasses import dataclass
import numpy as np
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
    sss_waterfall_image: np.array
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
    def annotated_keypoints_sorted(self):
        """Returns the annotated keypoints hashes and points as both as sorted list. The list
        elements consist of keypoints in the form of (ping_idx, bin_idx) extracted from the
        original annotated_keypoints' "pos" attribute, i.e. they are given in the index of
        the patch. The list is sorted first according to ping_idx, then according to bin_idx."""
        sorted_hashes = sorted(
            self.annotated_keypoints,
            key=lambda k: self.annotated_keypoints[k]['pos'])
        sorted_pos = [
            self.annotated_keypoints[k]['pos'] for k in sorted_hashes
        ]
        return sorted_hashes, sorted_pos

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
