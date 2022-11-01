import numpy as np
from numpy.random import default_rng
from prediction import predict_label
from build_tree import decision_tree_learning


def calc_confusion_matrix(correct_labels, predicted_labels):
    """ Compute the confusion matrix.

    Args:
        correct_labels (np.array) : the correct labels.
        predicted_labels (np.array) : the predicted labels.

    Returns:
        confusion_matrix (np.array) : KxK array where K = number of classes.
                                      Rows are the correct labels, columns are predictions.
    """
    class_labels = np.unique(np.concatenate((correct_labels, predicted_labels)))

    confusion_matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)

    for i in range(len(correct_labels)):
      correct_label = correct_labels[i]
      predicted_label = predicted_labels[i]
      correct_label_index = np.where(class_labels==correct_label)[0][0]
      predicted_label_index = np.where(class_labels==predicted_label)[0][0]
      confusion_matrix[correct_label_index][predicted_label_index] += 1

    return confusion_matrix


def calc_accuracy(confusion_matrix):
    """ Calculate the accuracy of the classifier from the confusion matrix.

    Args:
        confusion_matrix (np.array) : KxK array where K = number of classes.

    Returns:
        accuracy (float) : overall accuracy of classifier.
    """
    accuracy = np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)
    return accuracy


def calc_accuracy_direct(correct_labels, predicted_labels):
    """ Calculate the accuracy directly without the confusion matrix.

    Args:
        correct_labels (np.array) : the correct labels.
        predicted_labels (np.array) : the predicted labels.

    Returns:
        accuracy (float) : accuracy.
    """
    accuracy = np.sum(correct_labels == predicted_labels) / len(correct_labels)
    return accuracy


def calc_precision(confusion_matrix):
    """ Calculate the precision per class from the confusion matrix.
    
    Args:
        confusion_matrix (np.array) : KxK array where K = number of classes.

    Returns:
        precision (np.array) : (K,)-dimensional vector where K = number of classes.
    """
    precision = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    return precision


def calc_recall(confusion_matrix):
    """ Calculate the recall per class from the confusion matrix.
    
    Args:
        confusion_matrix (np.array) : KxK array where K = number of classes.

    Returns:
        recall (np.array) : (K,)-dimensional vector where K = number of classes.
    """
    recall = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    return recall


def calc_f1_score(precision, recall):
    """ Calculate the f1 score from precision and recall.

    Args:
        precision (np.array) : (K,)-dimensional vector where K = number of classes.
        recall (np.array) : (K,)-dimensional vector where K = number of classes.

    Returns:
        f1_score (np.array) : (K,)-dimensional vector where K = number of classes.
    """
    f1_score = 2*precision*recall / (precision+recall)
    return f1_score


def evaluate_tree(test_dataset, trained_tree):
    """ Calculate evaluation metrics for trained_tree based on test_dataset.

    Args:
        test_dataset (np.array) : Nx8 array where N = number of samples, columns 0 to 6 are attributes and 7 is label.
        trained_tree (Node) : root node of trained tree.

    Returns:
        confusion_matrix (np.array) : KxK array where K = number of classes.
        accuracy (float) : overall accuracy of classifier.
        precision (np.array) : (K,)-dimensional vector where K = number of classes.
        recall (np.array) : (K,)-dimensional vector where K = number of classes.
        f1_score (np.array) : (K,)-dimensional vector where K = number of classes.
    """
    correct_labels = test_dataset[:, -1]
    predicted_labels = predict_label(trained_tree, test_dataset[:, 0:7])
    confusion_matrix = calc_confusion_matrix(correct_labels, predicted_labels)
    accuracy = calc_accuracy(confusion_matrix)
    precision = calc_precision(confusion_matrix)
    recall = calc_recall(confusion_matrix)
    f1_score = calc_f1_score(precision, recall)

    return (confusion_matrix, accuracy, precision, recall, f1_score)


if __name__ == "__main__":
    seed = 50000
    random_generator = default_rng(seed)

    dataset = np.loadtxt("wifi_db/clean_dataset.txt")

    shuffled_indices = random_generator.permutation(len(dataset))
    test_dataset = dataset[shuffled_indices[:200]] 
    training_dataset = dataset[shuffled_indices[200:]]

    root_node = decision_tree_learning(training_dataset)
    confusion_matrix, accuracy, precision, recall, f1_score = evaluate_tree(test_dataset, root_node)

    print(confusion_matrix)
    print(accuracy)
    print(precision)
    print(recall)
    print(f1_score)
    #