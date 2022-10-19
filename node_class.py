class Node:
    def __init__(self, split, depth, left_child=None, right_child=None, is_leaf=True):
        self.split = split
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_leaf = is_leaf
