import info_gain

def find_split(dataset):
    """ Find best split for dataset based on maximum information gain.

    Go through each attribute (sorted) and each possible threshold and compute the information gain for each split.
    (Possibly write split class).
    Store information gains in np.array(?) and return split that gives maximum information gain.

    Args:
        dataset (np.array) : Nx8 array where N = number of samples, column 0 to 6 are attributes and 7 is label.

    Returns:
        int : Index of attribute to split on.
        float : Threshold for split
    """
    return None
