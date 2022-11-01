import numpy as np
from numpy.random import default_rng
from evaluation import evaluate_tree
from build_tree import decision_tree_learning
from prune_tree import prune_tree
from eval_before_prune import train_test_k_fold


def k_cross_validation_option_two(k, inner_folds, dataset, rg):
    """
    Function to output metrics for Option 2 of k cross-validation
    Args :
        k (int): number of outer folds to cross-validate
        inner_folds(int): number of inner folds per k-fold
        dataset (np.array): dataset to split in k outer cross validation
        rg (np.random.Generator): random number generator

    Return :
        outer_avg_confusion_matrix (np.array): 4x4 array
        outer_avg_accuracy (float): overall avg accuracy in k-cross validation
        outer_avg_recall (np.array): (R,)-dimensional vector where R = number of classes for overall recall
        outer_avg_precision (np.array): (R,)-dimensional vector where R = number of classes for overall precision
        outer_avg_f1_score (float): (R,)-dimensional vector where R = number of classes for overall f1
        outer_avg_max_depth (float): overall avg max depth of pruned trees
    """

    # initiate list to store metrics and max depth from each outer cross-validation
    outer_confusion_matrices = []
    outer_accuracies = []
    outer_recalls = []
    outer_precisions = []
    outer_f1_scores = []
    outer_max_depth = []

    # cross-validation outer loop
    # split to train-val data & test data
    n_instances = len(dataset[:, 1])
    for (trainval_indices, test_indices) in train_test_k_fold(k, n_instances, rg):
        trainval_data = dataset[trainval_indices,]
        test_data = dataset[test_indices,]

        # initialise lists to store metrics and max depth from each inner cross-validation
        inner_confusion_matrices = []
        inner_accuracies = []
        inner_recalls = []
        inner_precisions = []
        inner_f1_scores = []
        inner_max_depth = []

        # inner for loop - training data and val split
        inner_n_instances = len(trainval_data[:, 1])
        for (inner_train_indices, inner_val_indices) in train_test_k_fold(inner_folds, inner_n_instances, rg):
            inner_train_data = trainval_data[inner_train_indices, ]
            inner_val_data = trainval_data[inner_val_indices, ]
            # use train data to train tree
            trained_tree = decision_tree_learning(inner_train_data)

            # use val data to decide how we should prune tree
            pruned_tree = prune_tree(trained_tree, inner_train_data, inner_val_data)

            # evaluation metrics for pruned tree
            confusion_matrix, accuracy, precision, recall, f1_score = evaluate_tree(test_data, pruned_tree)

            # add each metric and max depth into inner list
            inner_confusion_matrices.append(confusion_matrix)
            inner_accuracies.append(accuracy)
            inner_recalls.append(recall)
            inner_precisions.append(precision)
            inner_f1_scores.append(f1_score)
            inner_max_depth.append(pruned_tree.max_depth())

            # calculate inner avg for metrics and max depth
            inner_avg_confusion_matrix = sum(inner_confusion_matrices) / inner_folds
            inner_avg_accuracy = sum(inner_accuracies) / inner_folds
            inner_avg_recall = sum(inner_recalls) / inner_folds
            inner_avg_precision = sum(inner_precisions) / inner_folds
            inner_avg_f1_score = sum(inner_f1_scores) / inner_folds
            inner_avg_max_depth = sum(inner_max_depth) / inner_folds

        # add each metric and max depth into outer list
        outer_confusion_matrices.append(inner_avg_confusion_matrix)
        outer_accuracies.append(inner_avg_accuracy)
        outer_recalls.append(inner_avg_recall)
        outer_precisions.append(inner_avg_precision)
        outer_f1_scores.append(inner_avg_f1_score)
        outer_max_depth.append(inner_avg_max_depth)

        # calculate outer avg
        outer_avg_confusion_matrix = sum(outer_confusion_matrices) / k
        outer_avg_accuracy = sum(outer_accuracies) / k
        outer_avg_recall = sum(outer_recalls) / k
        outer_avg_precision = sum(outer_precisions) / k
        outer_avg_f1_score = sum(outer_f1_scores) / k
        outer_avg_max_depth = sum(outer_max_depth) / k

    return outer_avg_confusion_matrix, outer_avg_accuracy, outer_avg_recall, outer_avg_precision, outer_avg_f1_score, outer_avg_max_depth


if __name__ == "__main__":
    seed = 25000
    random_generator = default_rng(seed)
    clean_dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    noisy_dataset = np.loadtxt("wifi_db/noisy_dataset.txt")
    conf_matrix_clean, accuracy_clean, recall_clean, precision_clean, f1_clean, max_depth_clean = k_cross_validation_option_two(10, 10, clean_dataset, random_generator)
    conf_matrix_noisy, accuracy_noisy, recall_noisy, precision_noisy, f1_noisy, max_depth_noisy = k_cross_validation_option_two(10, 10, noisy_dataset, random_generator)
    print(conf_matrix_clean, accuracy_clean, recall_clean, precision_clean, f1_clean, max_depth_clean)
    print(conf_matrix_noisy, accuracy_noisy, recall_noisy, precision_noisy, f1_noisy, max_depth_noisy)
    #
