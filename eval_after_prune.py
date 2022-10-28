#rough outline to be edited!!!

import numpy as np
from numpy.random import default_rng
from evaluation import evaluate_tree
from prediction import predict_label
from build_tree import decision_tree_learning
from prune_tree import prune_tree
from eval_before_prune import train_test_k_fold

# copied from eval before prune - need to change name for this
def k_cross_validation(k, dataset, rg): # need to edit this argument so that we can vary the inner loop k value
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
    """
    n_instances = len(dataset[:, 1])

    # initiate list to store metrics from each cross-validation
    k_confusion_matrices = []
    k_accuracies = []
    k_recalls = []
    k_precisions = []
    k_f1_scores = []

    # cross-validation outer loop (split to train-val data & test data)
    # need to amend below codes
    for (train_indices, test_indices) in train_test_k_fold(k, n_instances, rg):
        train_data = dataset[train_indices,]
        test_data = dataset[test_indices,]
        # call train_test_k_fold again to split train & val

        # inner for loop - training data and val split

             # use train data to train tree
             trained_tree = decision_tree_learning(train_data)  # Node

            # use val data to prune tree

             # evaluation metrics
            confusion_matrix, accuracy, precision, recall, f1_score = evaluate_tree(test_data, trained_tree)

            # add each metric into list
            k_confusion_matrices.append(confusion_matrix)
            k_accuracies.append(accuracy)
            k_recalls.append(recall)
            k_precisions.append(precision)
            k_f1_scores.append(f1_score)

            # calculate avg
            avg_confusion_matrix = sum(k_confusion_matrices) / k
            avg_accuracy = sum(k_accuracies) / k
            avg_recall = sum(k_recalls) / k
            avg_precision = sum(k_precisions) / k
            avg_f1_score = sum(k_f1_scores) / k

        # compile the  averages and do a global average them again.

    return avg_confusion_matrix, avg_accuracy, avg_recall, avg_precision, avg_f1_score