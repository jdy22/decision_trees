import numpy as np
from eval_before_prune import k_cross_validation
from eval_after_prune import k_cross_validation_option_two
from numpy.random import default_rng

# load data
dataset = np.loadtxt("wifi_db/clean_dataset.txt") 

# create random generator
seed = 25000
random_generator = default_rng(seed)

# evalute before pruning
conf_matrix, accuracy, recall, precision, f1, max_depth = k_cross_validation(10, dataset, random_generator)
print("Before Pruning:")
print("The averaged confusion matrix is")
print(conf_matrix)
print("The averaged overall accuracy is", round(accuracy,5))
print("The averaged recall for each class is", recall)
print("The averaged precision for each class is", precision)
print("The averaged f1 score for each class is", f1)
print("The averaged depth of the tree is", max_depth)
print("----------------------------------------------------------------")
print("\n")

# evalute before pruning
conf_matrix_after, accuracy_after, recall_after, precision_after, f1_after, max_depth_after = k_cross_validation_option_two(10, 10, dataset, random_generator)
print("After Pruning:")
print(conf_matrix_after)
print("The averaged overall accuracy is", round(accuracy_after,5))
print("The averaged recall for each class is", recall_after)
print("The averaged precision for each class is", precision_after)
print("The averaged f1 score for each class is", f1_after)
print("The averaged depth of the tree is", max_depth_after)
print("----------------------------------------------------------------")

