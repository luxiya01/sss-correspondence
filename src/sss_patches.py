"""
This module contains the definition of SSSPatch class.

Exposes dataclass:
    - SSSPatch
"""
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
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
