from find_split import find_split, Split
from node_class import Node
import numpy as np


def decision_tree_learning(dataset, depth=0):
    """ Recursive function to build the tree.

    Args:
        dataset (np.array) : Nx8 array where N = number of samples, column 0 to 6 are attributes and 7 is label.
        depth (int) : Current depth of node.

    Returns:
        Node : Root node.
    """

    # if all labels are the same, return leaf_node
    if len(np.unique(dataset[:,-1])) == 1:
        leaf_node = Node(split=Split(),depth=depth,is_leaf=True,label=int(dataset[0, -1]))
        return leaf_node

    # else, create a new node with new Split object to split the dataset
    else:
        optimal_split = find_split(dataset)
        left_dataset = dataset[dataset[:,optimal_split.attribute_index]<=optimal_split.threshold]
        right_dataset = dataset[dataset[:,optimal_split.attribute_index]>optimal_split.threshold]

        left_child=decision_tree_learning(left_dataset, depth=depth+1)
        right_child=decision_tree_learning(right_dataset, depth=depth+1)
        current_node = Node(split=optimal_split, left_child=left_child, right_child=right_child, depth=depth)

    return current_node

if __name__ == "__main__":
    dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    root_node = decision_tree_learning(dataset)
    print(root_node)
    print(root_node.right_child)
    print(root_node.left_child.depth)
    print(root_node.split.attribute_index)
    print(root_node.split.threshold)
    print(root_node.max_depth())

