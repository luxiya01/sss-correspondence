import numpy as np
from simple_shapes import Rectangle


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
