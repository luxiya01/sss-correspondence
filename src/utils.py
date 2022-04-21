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
    patches_in_dir = sorted(
        [
            os.path.join(folder, x)
            for x in os.listdir(folder) if x.split('.')[-1] == 'pkl'
        ],
        key=lambda p: int(p.split('/')[-1].split('_')[0][5:]))
    return patches_in_dir


def write_pairs_to_file(array: np.array, thresh: float, folder: str,
                        out_filename: str) -> int:
    indices_above_thresh = np.argwhere(array > thresh)
    nbr_pairs = indices_above_thresh.shape[0]
    patch_id_to_filename = lambda patch_id, suffix: f'patch{patch_id}_{suffix}.png'

    pairs_raw_intensity_str = []
    pairs_norm_intensity_str = []
    for i, j in indices_above_thresh:
        pairs_norm_intensity_str.append(
            f'{patch_id_to_filename(i, "norm_intensity")} {patch_id_to_filename(j, "norm_intensity")}'
        )
        pairs_raw_intensity_str.append(
            f'{patch_id_to_filename(i, "intensity")} {patch_id_to_filename(j, "intensity")}'
        )

    with open(os.path.join(folder, f'{out_filename}_norm_intensity.txt'),
              'w') as f:
        f.writelines('\n'.join(pairs_norm_intensity_str))
    with open(os.path.join(folder, f'{out_filename}_raw_intensity.txt'),
              'w') as f:
        f.writelines('\n'.join(pairs_raw_intensity_str))
    return nbr_pairs


def generate_overlap_pairs_txt(folder: str, overlap_thresh: float = .1):
    """Assumes there being an overlap.npz file in  the folder. Construct two txt files:
    - From the overlap_matrix, list all pairs of patces with ovelap > overlap_thresh and write the
      pairs to pairs_with_over_{overlap_thresh}_overlap.txt.
    - From the overlap_nbr_kps, list all pairs of patches sharing at least 1 keypoint and write the
      pairs to pairs_sharinig_kps.txt.
    """
    overlap_file = np.load(os.path.join(folder, 'overlap.npz'))
    nbr_pairs_above_overlap_thresh = write_pairs_to_file(
        np.triu(overlap_file['overlap_matrix']), overlap_thresh, folder,
        f'pairs_with_over_{overlap_thresh}_overlap')
    nbr_pairs_sharing_kps = write_pairs_to_file(
        np.triu(overlap_file['overlap_nbr_kps']), 0, folder,
        'pairs_sharinig_kps')
    print(
        f'In {folder}, {nbr_pairs_above_overlap_thresh} pairs have > {overlap_thresh} overlap, '
        f'{nbr_pairs_sharing_kps} pairs share at least one keypoint')
    return nbr_pairs_above_overlap_thresh, nbr_pairs_sharing_kps
