import numpy as np

def calc_entropy(dataset):
    """ Calculate entropy of dataset.

    Args:
        dataset (np.array) : Nx8 array where N = number of samples, column 0 to 6 are attributes and 7 is label.

    Returns:
        float : Entropy of dataset
    """
    labels = dataset[:,-1]
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_count = np.sum(counts)
    weights = counts/total_count
    entropy = np.sum(-weights*np.log2(weights))

    return entropy


def calc_info_gain(parent_dataset, left_dataset, right_dataset):
    """ Calculate information gain from splitting parent_datset into left_dataset and right_dataset.

    Args:
        parent_dataset (np.array) : Nx8 array
        left_dataset (np.array) : Kx8 array
        right_dataset (np.array) : (N-K)x8 array

    Returns:
        float : Information gain of split
    """
    size_parent = len(parent_dataset)
    size_left = len(left_dataset)
    size_right = len(right_dataset)

    entropy_parent = calc_entropy(parent_dataset)
    entropy_left = calc_entropy(left_dataset)
    entropy_right = calc_entropy(right_dataset)

    # calculate info_gain from difference of entropy_parent and weighted left and right child entropies
    info_gain = entropy_parent - (size_left/size_parent)*entropy_left - (size_right/size_parent)*entropy_right

    return info_gain


if __name__ == "__main__":
    parent_dataset = np.array([[1, 2], [3, 2], [5, 4], [7, 4]])
    left_dataset = parent_dataset[0:2, :]
    right_dataset = parent_dataset[2:, :]
    info_gain = calc_info_gain(parent_dataset, left_dataset, right_dataset)
    print(info_gain)
    #