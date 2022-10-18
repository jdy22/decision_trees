class Node:
    def __init__(self, split_attribute, split_value, left_child, right_child, depth, is_leaf=False):
        self.split_attribute = split_attribute
        self.split_value = split_value
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_leaf = is_leaf
