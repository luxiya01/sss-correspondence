import numpy as np


def normalize_waterfall_image(waterfall_image, a_max=3):
    """Given a waterfall image (a 2D numpy array), divide each column
     by column mean and clip values between (0, a_max) for visualization."""
    waterfall_image = waterfall_image.copy()
    col_mean = waterfall_image.mean(axis=0)
    waterfall_image = np.divide(waterfall_image,
                                col_mean,
                                where=[col_mean != 0.])
    clipped_image = np.clip(waterfall_image, a_min=0, a_max=a_max)
    return clipped_image
