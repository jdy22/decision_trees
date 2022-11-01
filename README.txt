To build and evaluate a tree for a dataset:
1. Change the filepath in line 7 of main.py to point to the dataset of interest
2. Run main.py
This will implement an algorithm to build a decision tree for the dataset and will evaluate this algorithm first
by 10-fold cross-validation on the unpruned tree. It will report the confusion matrix, accuracy, recall, precision
and f1-score averaged over all folds, as well as the average depth of the unpruned tree.
It will then perform 10-fold nested cross-validation on the pruned tree and will report the confusion matrix,
accuracy, recall, precision and f1-score averaged over all folds, as well as the average depth of the pruned
tree.

