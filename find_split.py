import numpy as np
from info_gain import calc_info_gain


class Split:
    def __init__(self, attribute_index=None, threshold=None):
        self.attribute_index = attribute_index
        self.threshold = threshold


def find_split(parent_dataset):
    """ Find best split for dataset based on maximum information gain.

    Go through each attribute (sorted) and each possible threshold and compute the information gain for each split.
    Store information gains in np.array and return split that gives maximum information gain.

    Args:
        parent_dataset (np.array) : Nx8 array where N = number of samples, column 0 to 6 are attributes and 7 is label.

    Returns:
        Split : Split object containing optimal attribute index and threshold.
    """
    splits = []
    info_gains = []

    data_rows, data_columns = np.shape(parent_dataset)

    for attribute_index in range(data_columns-1):
        attribute_values = parent_dataset[:, attribute_index]
        sorted_attribute_values = np.sort(attribute_values)
        for i in range(len(sorted_attribute_values)-1):
            if sorted_attribute_values[i] == sorted_attribute_values[i+1]:
                continue
            threshold = (sorted_attribute_values[i] + sorted_attribute_values[i+1])/2
            split = Split(attribute_index, threshold)
            left_dataset = parent_dataset[parent_dataset[:,attribute_index]<=threshold]
            right_dataset = parent_dataset[parent_dataset[:,attribute_index]>threshold]
            info_gain = calc_info_gain(parent_dataset, left_dataset, right_dataset)
            splits.append(split)
            info_gains.append(info_gain)

    max_index = info_gains.index(max(info_gains))
    optimal_split = splits[max_index]

    return optimal_split


if __name__ == "__main__":
    dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    split = find_split(dataset)
    print(type(split))
    print(split.attribute_index)
    print(split.threshold)
            



