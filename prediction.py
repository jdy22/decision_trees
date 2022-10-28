import numpy as np
from node_class import Node
from build_tree import decision_tree_learning



def predict_label(trained_tree: Node, test_db):
    """ Predict label for test_db using the trained_tree.

    Args:
        trained_tree (Node) : Root node of trained decision tree
        test_db (np.array) : Nx7-dimensional array of attributes for test set

    Outputs:
        predicted_labels (np.array) : (N,)-dimensional vector of predicted label for test set
    """

    predicted_labels = np.zeros(test_db.shape[0], dtype=int)
    for index, instance in enumerate(test_db): # cater for multiple test cases/instances
        current_node = trained_tree
        # traversing between left and right node depending on current node's split threshold until leaf node is reached
        while not current_node.is_leaf:
            if instance[current_node.split.attribute_index] <= current_node.split.threshold: # removed int from the attribute value as spec states they should be treated as continuous variables
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        # storing predicted label of leaf node
        predicted_labels[index] = current_node.label

    return predicted_labels


if __name__ == "__main__":
    dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    root_node = decision_tree_learning(dataset)

    # testing on actual training data from dataset (result should be 1,2,3 and 4)
    print(predict_label(root_node, np.array([[-65, -61, -65, -67, -69, -87, -84], [-45, -55, -54, -43, -71, -89, -77], [-46, -52, -53, -44, -60, -79, -86], [-61, -55, -52, -61, -44, -88, -92]])))

    # testing on made-up values (can't verify, sadly > have to wait for Task 3 eval portion)
    print(predict_label(root_node, np.array([[-40, -60, -68, -67, -79, -55, -78], [-90, -66, -50, -80, -45, -80, -99]])))
    print(predict_label(root_node, np.array([[-55, -68, -78, -59, -65, -60, -80]]))) # has to be 2D array even for single instance

