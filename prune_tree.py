import numpy as np
from numpy.random import default_rng
from build_tree import decision_tree_learning
from prediction import predict_label
from evaluation import calc_accuracy_direct

def prune_tree_once(root_node, training_set, validation_set):
    """ Prune tree according to validation error.

    One pass through the tree to determine which nodes can be pruned.

    Args:
        root_node (Node) : Root node of tree to prune.
        training_set (np.array):  Nx8 array where N = number of training samples, column 0 to 6 are attributes and 7 is label.
        validation_set (np.array):  Kx8 array where K = number of validation samples, column 0 to 6 are attributes and 7 is label.

    Returns:
        Node : Root node of pruned tree.
        bool : Indicates whether or not the tree has been pruned.
    """
    is_pruned = False

    left_child = root_node.left_child
    right_child = root_node.right_child

    if left_child.is_leaf and right_child.is_leaf:
        if len(validation_set) == 0:
            return root_node, False

        predicted_labels = predict_label(root_node, validation_set).astype(int)
        # print(f"Predicted labels not pruned: {predicted_labels}")
        correct_labels = validation_set[:, -1].astype(int)
        # print(f"Correct labels: {correct_labels}")
        accuracy_not_pruned = calc_accuracy_direct(correct_labels, predicted_labels)
        # print(f"Accuracy not pruned: {accuracy_not_pruned}")

        training_set_labels = training_set[:, -1].astype(int)
        majority_label = np.argmax(np.bincount(training_set_labels))
        predicted_labels_pruned = (majority_label * np.ones(len(correct_labels))).astype(int)
        # print(f"Predicted labels pruned: {predicted_labels_pruned}")
        accuracy_pruned = calc_accuracy_direct(correct_labels, predicted_labels_pruned)
        # print(f"Accuracy pruned: {accuracy_pruned}")

        if accuracy_pruned >= accuracy_not_pruned:
            root_node.left_child = None
            root_node.right_child = None
            root_node.is_leaf = True
            root_node.label = majority_label
            # print("Pruned")
            return root_node, True
        else:
            # print("Not Pruned")
            return root_node, False

    else:
        split = root_node.split
        left_training_dataset = training_set[training_set[:,split.attribute_index]<=split.threshold]
        right_training_dataset = training_set[training_set[:,split.attribute_index]>split.threshold]
        left_validation_dataset = validation_set[validation_set[:,split.attribute_index]<=split.threshold]
        right_validation_dataset = validation_set[validation_set[:,split.attribute_index]>split.threshold]

        is_left_pruned = False
        is_right_pruned = False
        if not left_child.is_leaf:
            root_node.left_child, is_left_pruned = prune_tree_once(left_child, left_training_dataset, left_validation_dataset)
        if not right_child.is_leaf:
            root_node.right_child, is_right_pruned = prune_tree_once(right_child, right_training_dataset, right_validation_dataset)

        if is_left_pruned or is_right_pruned:
            is_pruned = True

    return root_node, is_pruned


def prune_tree(root_node, training_set, validation_set):
    """ Prune tree according to validation error.

    Calls prune_tree_once over and over until no more nodes need pruning.

    Args:
        root_node (Node) : Root node of tree to prune.
        training_set (np.array):  Nx8 array where N = number of training samples, column 0 to 6 are attributes and 7 is label.
        validation_set (np.array):  Kx8 array where K = number of validation samples, column 0 to 6 are attributes and 7 is label.

    Returns:
        Node : Root node of pruned tree.
    """
    continue_pruning = True

    while continue_pruning:
        # print("Pruning once")
        root_node, continue_pruning = prune_tree_once(root_node, training_set, validation_set)
        # print(continue_pruning)
        # print()

    # print("Pruning finished")
    return root_node


if __name__ == "__main__":
    seed = 25000
    random_generator = default_rng(seed)
    
    dataset = np.loadtxt("wifi_db/clean_dataset.txt")

    shuffled_indices = random_generator.permutation(len(dataset))
    validation_dataset = dataset[shuffled_indices[:200]] 
    training_dataset = dataset[shuffled_indices[200:]]
    
    root_node = decision_tree_learning(training_dataset)
    new_root_node = prune_tree(root_node, training_dataset, validation_dataset)
