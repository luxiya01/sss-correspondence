import numpy as np
import os
from simple_shapes import Rectangle
from sss_patches import SSSPatch


def normalize_waterfall_image(waterfall_image: np.array,
                              a_max: float = 3) -> np.array:
    """Given a waterfall image (a 2D numpy array), divide each column
     by column mean and clip values between (0, a_max) for visualization."""
    waterfall_image = waterfall_image.copy()
    col_mean = waterfall_image.mean(axis=0)
    waterfall_image = np.divide(waterfall_image,
                                col_mean,
                                where=[col_mean != 0.])
    clipped_image = np.clip(waterfall_image, a_min=0, a_max=a_max)
    return clipped_image


def compute_overlap_between_two_rectangles(r1: Rectangle,
                                           r2: Rectangle) -> float:
    """Returns the overlap between two Rectangle. The overlap is betwen [0, 1], with 0 being no
    overlap and 1 being the two rectangles cover the exact same areas."""
    dx = min(r1.xmax, r2.xmax) - max(r1.xmin, r2.xmin)
    dy = min(r1.ymax, r2.ymax) - max(r1.ymin, r2.ymin)

    overlap_percentage = 0.
    if dx > 0 and dy > 0:
        overlapping_area = dx * dy
        overlap_percentage = overlapping_area / (r1.area + r2.area -
                                                 overlapping_area)
    return overlap_percentage


def get_sorted_patches_list(folder: str) -> list:
    """Return a list of patches in the given folder, sorted in ascending order of patch id"""
    patches_in_dir = [
        os.path.join(folder, x) for x in os.listdir(folder)
        if x.split('.')[-1] == 'pkl'
        and not os.path.isdir(os.path.join(folder, x))
    ]
    patches_in_dir = sorted(
        patches_in_dir, key=lambda p: int(p.split('/')[-1].split('_')[0][5:]))
    return patches_in_dir


def get_gt_overlap_between_two_patches(patch1: SSSPatch,
                                       patch2: SSSPatch) -> list:
    """Given two SSSPatches, return a list of groundtruth keypoint correspondences.
    The length of the list is the same as the number of keypoints in patch1, and the value at index
    i is the corresponding keypoint index in patch2, i.e. list[i] = j means that
    patch1.annotated_keypoints_sorted[1][i] = patch2.annotated_keypoints_sorted[1][j]. If there is
    no corresponding keypoints, the value is set to -1."""
    patch1_kp_hash = list(patch1.annotated_keypoints.keys())
    patch2_kp_hash = list(patch2.annotated_keypoints.keys())

    patch1_kp_to_patch2_kp = [-1] * len(patch1_kp_hash)
    for i, key in enumerate(patch1_kp_hash):
        if key not in patch2_kp_hash:
            continue
        patch1_kp_to_patch2_kp[i] = patch2_kp_hash.index(key)
    return patch1_kp_to_patch2_kp
