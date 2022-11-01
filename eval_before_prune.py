import numpy as np
from numpy.random import default_rng
from evaluation import evaluate_tree
from build_tree import decision_tree_learning


def k_fold_split(n_splits, n_instances, random_generator):
    """ Split n_instances into n mutually exclusive splits at random.
    
    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a 
            numpy array giving the indices of the instances in that split.
    """

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.
    
    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
            with two elements: a numpy array containing the train indices, and another 
            numpy array containing the test indices.
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]

        # combine remaining splits as train
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds

def k_cross_validation(k, dataset, rg):
    """
    Option 1 of cross-validation
    Args : 
        k (int): number of folds to cross-validate
        dataset (np.array): dataset to split in ten cross validation
        rg (np.random.Generator): random number generator

    Return :
        avg_confusion_matrix (np.array): 4x4 array
        avg_accuracy (float): avg accuracy in k cross validation
        avg_recall (np.array): (R,)-dimensional vector where R = number of classes
        avg_precision (np.array): (R,)-dimensional vector where R = number of classes
        avg_f1_score (float): (R,)-dimensional vector where R = number of classes
        avg_max_depth (float): avg max depth of trees

    """
    n_instances = len(dataset[:,1])

    # initiate list to store metrics from each cross-validation and max-depth
    k_confusion_matrices = []
    k_accuracies = []
    k_recalls = []
    k_precisions = []
    k_f1_scores = []
    k_max_depth = []
    
    # cross-validation
    for (train_indices, test_indices) in train_test_k_fold(k, n_instances, rg):
        train_data = dataset[train_indices,]
        test_data = dataset[test_indices,]

        # use train data to train tree
        trained_tree = decision_tree_learning(train_data) # Node

        # evaluation metrics
        confusion_matrix, accuracy, precision, recall, f1_score = evaluate_tree(test_data, trained_tree)

        # add each metric and max-depth into list
        k_confusion_matrices.append(confusion_matrix)
        k_accuracies.append(accuracy)
        k_recalls.append(recall)
        k_precisions.append(precision)
        k_f1_scores.append(f1_score)
        k_max_depth.append(trained_tree.max_depth())

    # calculate average
    avg_confusion_matrix = sum(k_confusion_matrices)/k
    avg_accuracy = sum(k_accuracies)/k
    avg_recall = sum(k_recalls)/k
    avg_precision = sum(k_precisions)/k
    avg_f1_score = sum(k_f1_scores)/k
    avg_max_depth = sum(k_max_depth)/k

    return avg_confusion_matrix, avg_accuracy, avg_recall, avg_precision, avg_f1_score, avg_max_depth


if __name__ == "__main__":
    seed = 25000
    random_generator = default_rng(seed)
    clean_dataset = np.loadtxt("wifi_db/clean_dataset.txt") 
    noisy_dataset = np.loadtxt("wifi_db/noisy_dataset.txt")
    conf_matrix_clean, accuracy_clean, recall_clean, precision_clean, f1_clean, max_depth_clean = k_cross_validation(10, clean_dataset,random_generator)
    conf_matrix_noisy, accuracy_noisy, recall_noisy, precision_noisy, f1_noisy, max_depth_noisy = k_cross_validation(10, noisy_dataset,random_generator)
    print(conf_matrix_clean, accuracy_clean, recall_clean, precision_clean, f1_clean, max_depth_clean)
    print(conf_matrix_noisy, accuracy_noisy, recall_noisy, precision_noisy, f1_noisy, max_depth_noisy)
