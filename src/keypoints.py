import cv2
from utils import normalize_waterfall_image
from sss_patches import SSSPatch

def compute_descriptors_at_annotated_locations(patch: SSSPatch, algo: cv2.Feature2D, kp_size: int =
        16, use_orig_sss_intensities: bool=False):
    """Compute traditional descriptors using OpenCV's cv2.Feature2D class for a given SSSPatch.

    Parameters
    ----------
    patch: SSSPatch
        The descriptors are computued from the normalized 8 bit image of the sss_waterfall_image of
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
        ping_nbr, bin_nbr = kp['pos'][1], kp['pos'][0]
        annotated_kps.append(cv2.KeyPoint(bin_nbr, ping_nbr, size=kp_size))

    img = patch.sss_waterfall_image
    if not use_orig_sss_intensities:
        img = normalize_waterfall_image(img)

    normalized_8bit_img = cv2.normalize(img,  None,
            0, 255, cv2.NORM_MINMAX).astype('uint8')
    annotated_kps, desc = algo.compute(normalized_8bit_img, annotated_kps)
    return annotated_kps, desc
