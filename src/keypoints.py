import cv2
import numpy as np
from utils import normalize_waterfall_image
from sss_patches import SSSPatch

def compute_descriptors_at_annotated_locations(patch: SSSPatch, algo: cv2.Feature2D, kp_size: int =
        16, use_orig_sss_intensities: bool=False):
    """Compute traditional descriptors using OpenCV's cv2.Feature2D class for a given SSSPatch.

    Parameters
    ----------
    patch: SSSPatch
        The descriptors are computed from the normalized 8 bit image of the sss_waterfall_image of
        the patch.
    algo: cv2.Feature2D
        A Feature2D instance from cv2 used to compute the descriptors. e.g. SIFT, SURF, ORB.
    kp_size: int = 16
        The diameter of the neighbourhood to be included in the descriptor computation of a given
        keypoint.
    use_orig_sss_intensities: bool = False
        If False, the normalized 8 bit image used to compute descriptors will first be normalized by
        utils.normalize_waterfall_image. If True, the 8 bit image comes directly from normalizing
        the original intensities in sss_waterfall_image.

    Returns
    -------
    annotated_kps: list[cv2.KeyPoint]
        The list of annotated keypoints, each keypoint is converted into a cv2.KeyPoint with
        location at (bin_nbr (x value), ping_nbr (y value)), size = kp_size
    desc: np.array
        The descriptors computed at the annotated keypoint locations. Shape = (nbr_kps, 128)
    """
    annotated_kps = []
    for kp in patch.annotated_keypoints.values():
        ping_nbr, bin_nbr = kp['pos'][0], kp['pos'][1]
        annotated_kps.append(cv2.KeyPoint(bin_nbr, ping_nbr, size=kp_size))

    img = patch.sss_waterfall_image
    if not use_orig_sss_intensities:
        img = normalize_waterfall_image(img)

    normalized_8bit_img = cv2.normalize(img,  None,
            0, 255, cv2.NORM_MINMAX).astype('uint8')
    annotated_kps, desc = algo.compute(normalized_8bit_img, annotated_kps)
    return annotated_kps, desc

def compute_flattened_neighbourhood_pixel_values(patch: SSSPatch, ping_neighbour: int = 8,
        bin_neighbour: int = 8, use_orig_sss_intensities: bool = False):
    """For all annotated keypoints in a given SSSPatch, compute a descriptor in the form of flatten
    pixel values around the keypoint location.

    Parameters
    ----------
    patch: SSSPatch
        The annotated keypoints from this patch will be used for descriptor computations.
    ping_neighbour: int = 8
        The number of pings prior and after the keypoint ping to be counted into its
        "neighbourhood". A ping_neighbour of 8 means that the descriptor is computed from the 8
        pings proceeding the keypoint ping +  the keypoint ping + the 8 pings after keypoint ping =
        8 + 1 + 8 = 17 pings.
    bin_neighbour: int = 8
        The number of bins prior and after the keypoint bin to be counted into its "neighbourhood".
        Similar to ping_neighbour.
    use_orig_sss_intensities: bool = False
        If False, use the pixel values after applying utils.normalize_waterfall_image to
        sss_waterfall_image. If True, use the pixel values from sss_waterfall_image directly.

    Returns
    -------
    annotated_kps: list[(int, int)]
        The list consist of tuples of the form (bin_nbr (x value), ping_nbr (y value)).
    desc: np.array
        The descriptors computed by flattening the pixel array defined by ping_neighbour and
        bin_neighbour. Shape = (nbr_kps, (ping_neighbour*2 + 1) * (bin_neighbour*2 + 1)).
    """
    img = patch.sss_waterfall_image
    if not use_orig_sss_intensities:
        img = normalize_waterfall_image(img)

    annotated_kps = []
    desc = []
    for kp in patch.annotated_keypoints.values():
        ping_nbr, bin_nbr = kp['pos'][0], kp['pos'][1]
        annotated_kps.append((bin_nbr, ping_nbr))


        #TODO: debug code when the keypoint is at the edge
        min_ping_nbr = max(ping_nbr - ping_neighbour, 0)
        max_ping_nbr = min(ping_nbr + ping_neighbour + 1, img.shape[0])
        if min_ping_nbr == 0:
            pass

        flatten_pixels = img[ping_nbr - ping_neighbour:ping_nbr + ping_neighbour + 1, bin_nbr-bin_neighbour:bin_nbr+bin_neighbour+1].flatten()
        desc.append(flatten_pixels)
        print(ping_nbr, bin_nbr, len(flatten_pixels))
    return annotated_kps, desc #np.array(desc)
