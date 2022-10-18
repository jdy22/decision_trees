def calc_entropy(dataset):
    """ Calculate entropy of dataset.

    Args:
        dataset (np.array) : Nx8 array where N = number of samples, column 0 to 6 are attributes and 7 is label.

    Returns:
        float : Entropy of dataset
    """
    return None


def calc_info_gain(parent_dataset, left_dataset, right_dataset):
    """ Calculate information gain from splitting parent_datset into left_dataset and right_dataset.
    Call calc_entropy. Remember to weight the left and right entropies according to dataset size!

    Args:
        parent_dataset (np.array) : Nx8 array
        left_dataset (np.array) : Kx8 array
        right_dataset (np.array) : (N-K)x8 array

    Returns:
        float : Information gain of split
    """
    return None
