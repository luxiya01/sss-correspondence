from random import random


def train_test_split(sss_data_folder: str,
                     ref_sss_data: str,
                     test_size: float = .2,
                     random_seed: int = 0):
    """
    Splits the sss_meas_data found in the sss_data_folder into training and test sets.
    This function assumes that the sss_data_folder contains the following data:
        - sss_meas_data .cereal files,
        - a json file named 'valid_idx.json' containing the valid indices for every sss_meas_data

    Parameters
    ----------
    sss_data_folder: str
        The path to the folder with sss_meas_data to be used for training and testing.
    ref_sss_data: str
        The filename of the sss_meas_data that is to be used as reference data for the training and
        test split.
    test_size: float
        Float value between 0 and 1 indicating the proportion of the ref_sss_data used for testing.
    random_state: int
        Controls the random number generator for reproducibility. 

    Returns
    -------
    """
    #TODO: decide on training data
    pass
