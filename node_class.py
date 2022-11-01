class Node:
    def __init__(self, split, depth, left_child=None, right_child=None, is_leaf=False, label=None):
        self.split = split
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_leaf = is_leaf
        self.label = label

    # method to find max depth of tree (Node)
    def max_depth(self):
        left_child_depth = self.left_child.max_depth() if self.left_child else 0
        right_child_depth = self.right_child.max_depth() if self.right_child else 0
        return max(left_child_depth, right_child_depth) + 1
