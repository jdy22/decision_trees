class Node:
    def __init__(self, split, left_child, right_child, depth, is_leaf=False):
        self.split = split
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_leaf = is_leaf
