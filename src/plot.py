import numpy as np
from matplotlib import pyplot as plt
from sss_patches import SSSPatch
from utils import normalize_waterfall_image


def plot_ssspatch_with_annotated_keypoints(patch: SSSPatch):
    """Returns a matplotlib figure showing the sss_waterfall_image and annotated_keypoints in
    the patch."""
    fig, ax = plt.subplots()
    ax.set_title(
        f'SSSPatch {patch.patch_id}: pings ({patch.start_ping}, {patch.end_ping}), bins ({patch.start_bin},'
        f'{patch.end_bin}) from {patch.file_id}')
    ax.imshow(normalize_waterfall_image(patch.sss_waterfall_image),
              extent=[
                  patch.start_bin, patch.end_bin, patch.end_ping,
                  patch.start_ping
              ])
    ax.scatter([
        x['pos'][1] + patch.start_bin
        for x in patch.annotated_keypoints.values()
    ], [
        x['pos'][0] + patch.start_ping
        for x in patch.annotated_keypoints.values()
    ],
               c='y', s=5)
    return ax

def plot_corresponding_keypoints(patch1: SSSPatch, patch2: SSSPatch):
    """Given another SSSPatch, stack the sss_waterfall_image horizontally, plot the
    annotated_keypoints and draw lines between the keypoints from the two image that correspond to
    one another.

    Parameters
    ----------
    patch2: SSSPatch
        Another SSSPatch object
    """
    image = np.hstack([patch1.sss_waterfall_image, patch2.sss_waterfall_image])

    # show the stacked sss_waterfall_image
    fig, ax = plt.subplots(figsize=(20, 5))
    plt.imshow(normalize_waterfall_image(image))

    # show keypoints from patch1
    ax.scatter([x['pos'][1] for x in patch1.annotated_keypoints.values()],
               [x['pos'][0] for x in patch1.annotated_keypoints.values()],
               c='r',
               s=5)

    # show keypoints from patch2
    bin_offset = patch1.sss_waterfall_image.shape[1]
    ax.scatter([
        x['pos'][1] + bin_offset for x in patch2.annotated_keypoints.values()
    ], [x['pos'][0] for x in patch2.annotated_keypoints.values()],
               c='r',
               s=5)

    # show corresponding keypoints
    overlapping_keypoints = set(
        patch1.annotated_keypoints.keys()).intersection(
            patch2.annotated_keypoints.keys())
    for kp_hash in overlapping_keypoints:
        kp_patch1 = patch1.annotated_keypoints[kp_hash]['pos']
        kp_patch2 = patch2.annotated_keypoints[kp_hash]['pos']
        ax.plot([kp_patch1[1], kp_patch2[1] + bin_offset],
                [kp_patch1[0], kp_patch2[0]],
                c='y',
                linewidth=.8)
        ax.scatter([kp_patch1[1], kp_patch2[1] + bin_offset],
                   [kp_patch1[0], kp_patch2[0]],
                   c='y',
                   s=5)

    # show divider between patch1 and patch2
    ax.plot([bin_offset] * patch1.nbr_pings * 100,
            np.linspace(0, patch1.nbr_pings - 1, patch1.nbr_pings * 100),
            c='red')
    plt.axis('off')

    return ax
