class Node:
    def __init__(self, split, depth, left_child=None, right_child=None, is_leaf=True, label=None): # added label attribute
        self.split = split
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_leaf = is_leaf
        self.label = label # to identify labelling of the leaf node
