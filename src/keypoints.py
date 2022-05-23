import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import cv2
from utils import normalize_waterfall_image, get_sorted_patches_list
from sss_patches import SSSPatch


def compute_and_store_descriptors(folder: str, algo: str = 'sift'):
    patches = get_sorted_patches_list(folder)
    sift = cv2.SIFT_create()

    outfolder = os.path.join(folder, algo)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    for patch_path in patches:
        with open(patch_path, 'rb') as f:
            patch = pickle.load(f)
            if len(patch.annotated_keypoints.keys()) <= 0:
                print(f'Ignoring patch {patch.patch_id} from file {patch.filename} since it contains 0 keypoint.')
                continue

            if 'sift' in algo:
                kp_raw, desc_raw = compute_descriptors_at_annotated_locations(
                    patch, sift, use_orig_sss_intensities=True)
                kp_norm, desc_norm = compute_descriptors_at_annotated_locations(
                    patch, sift, use_orig_sss_intensities=False)

                # L2 normalization
                if algo == 'sift':
                    desc_raw = desc_raw / (np.linalg.norm(desc_raw, axis=1, ord=2, keepdims=True)
                        + np.finfo(float).eps)
                    desc_norm = desc_norm / (np.linalg.norm(desc_norm, axis=1, ord=2, keepdims=True)
                        + np.finfo(float).eps)


                # L1 normalization + Sqrt
                if algo == 'rootsift':
                    desc_raw = np.sqrt(desc_raw / (np.linalg.norm(desc_raw, axis=1, ord=1, keepdims=True)
                        + np.finfo(float).eps))
                    desc_norm = np.sqrt(desc_norm / (np.linalg.norm(desc_norm, axis=1, ord=1, keepdims=True)
                        + np.finfo(float).eps))

            elif algo == 'neighbour':
                kp_raw, desc_raw = compute_flattened_neighbourhood_pixel_values(
                    patch, use_orig_sss_intensities=True)
                kp_norm, desc_norm = compute_flattened_neighbourhood_pixel_values(
                    patch, use_orig_sss_intensities=False)

            np.savez(os.path.join(outfolder, f'patch{patch.patch_id}'),
                     kp_raw=kp_raw,
                     desc_raw=desc_raw,
                     kp_norm=kp_norm,
                     desc_norm=desc_norm)


def compute_descriptors_at_annotated_locations(
        patch: SSSPatch,
        algo: cv2.Feature2D,
        kp_size: int = 16,
        use_orig_sss_intensities: bool = False):
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
    annotated_kps: np.array
        An array of annotated keypoints of shape (nbr_kps, 2).
        Each array contains the (bin_nbr (x value), ping_nbr (y value)) of one keypoint.
    desc: np.array
        The descriptors computed at the annotated keypoint locations. Shape = (nbr_kps, 128)
    """
    annotated_kps = []
    for kp in patch.annotated_keypoints.values():
        ping_nbr, bin_nbr = kp['pos'][0], kp['pos'][1]
        annotated_kps.append([bin_nbr, ping_nbr])

    annotated_kps_as_cv2_kp = [
        cv2.KeyPoint(bin_nbr, ping_nbr, size=kp_size)
        for (bin_nbr, ping_nbr) in annotated_kps
    ]

    img = patch.sss_waterfall_image
    if not use_orig_sss_intensities:
        img = normalize_waterfall_image(img)

    normalized_8bit_img = cv2.normalize(img, None, 0, 255,
                                        cv2.NORM_MINMAX).astype('uint8')
    annotated_kps_as_cv2_kp, desc = algo.compute(normalized_8bit_img,
                                                 annotated_kps_as_cv2_kp)
    return np.array(annotated_kps, dtype=np.float32), desc


def compute_flattened_neighbourhood_pixel_values(
        patch: SSSPatch,
        ping_neighbour: int = 8,
        bin_neighbour: int = 8,
        use_orig_sss_intensities: bool = False):
    """For all annotated keypoints in a given SSSPatch, compute a descriptor in the form of flatten
    pixel values around the keypoint location. If the keypoint is at the edge, the neighbourhood
    that is outside the image will be filled with 0.

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
    annotated_kps: np.array
        An array of annotated keypoints of shape (nbr_kps, 2).
        Each array contains the (bin_nbr (x value), ping_nbr (y value)) of one keypoint.
    desc: np.array
        The descriptors computed by flattening the pixel array defined by ping_neighbour and
        bin_neighbour. Shape = (nbr_kps, (ping_neighbour*2 + 1) * (bin_neighbour*2 + 1)).
    """
    img = patch.sss_waterfall_image
    if not use_orig_sss_intensities:
        img = normalize_waterfall_image(img)

    # Create a copy of the image so that all descriptors will be of the same size
    pixel_vals = np.zeros(
        (ping_neighbour * 2 + img.shape[0], bin_neighbour * 2 + img.shape[1]))
    print(
        img.shape, pixel_vals[ping_neighbour:-ping_neighbour,
                              bin_neighbour:-bin_neighbour].shape)
    pixel_vals[ping_neighbour:-ping_neighbour,
               bin_neighbour:-bin_neighbour] = img

    annotated_kps = []
    desc = []
    for kp in patch.annotated_keypoints.values():
        ping_nbr, bin_nbr = kp['pos'][0], kp['pos'][1]
        annotated_kps.append((bin_nbr, ping_nbr))

        flatten_pixels = pixel_vals[ping_nbr:ping_nbr + ping_neighbour * 2 + 1,
                                    bin_nbr:bin_nbr + bin_neighbour * 2 +
                                    1].flatten()
        desc.append(flatten_pixels)
        print(ping_nbr, bin_nbr, len(flatten_pixels))
    return np.array(annotated_kps, dtype=np.float32), np.array(desc, dtype=np.float32)


def draw_keypoints_and_descriptors(patch: SSSPatch,
                                   kps: np.array,
                                   desc: np.array,
                                   desc_name: str,
                                   kp_size=16):
    #TODO: add documentation for this function
    kps = [cv2.KeyPoint(x[0], x[1], size=kp_size) for x in kps]

    img = cv2.normalize(normalize_waterfall_image(patch.sss_waterfall_image),
                        None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    kp_img = cv2.drawKeypoints(img, kps, None, color=(0, 255, 255))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(kp_img)
    ax[0].set_title('Keypoint locations')
    ax[1].imshow(desc)
    ax[1].set_title('Descriptors')

    fig.suptitle(
        f'Keypoint and {desc_name} descriptors for patch {patch.patch_id}',
        fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=1.05)
    return ax
