import matplotlib
import matplotlib.pyplot as plt
from build_tree import decision_tree_learning

dataset = np.loadtxt("wifi_db/clean_dataset.txt")
root_node = decision_tree_learning(dataset)

def visualisation_label(node):
    if node.is_leaf == False:
        split_threshold = node.split.threshold
        split_attribute_index = node.split.attribute_index 
        output = f'X{split_attribute_index}<{split_threshold}'
    elif node.is_leaf == True:
        output = f'Room : {node.label}'
    return output

def tree_visual(node, tree, count, spread, drop, x, y):
    if node.is_leaf == False: 
        tree.text(x,y, visualisation_label(node), color = "black", bbox = dict(facecolor="white", edgecolor = "black"))
        distance = spread/(2**count)
        if node.left_child:
            tree_visual(node.left_child,tree, count + 1, spread,drop, x - distance, y - drop)
            tree.plot([x,x-distance], [y,y-drop])
        if node.right_child:
            tree_visual(node.right_child, tree, count + 1, spread,drop, x + distance, y - drop)
            tree.plot([x,x + distance], [y,y - drop])
    elif node.is_leaf == True:
        tree.text(x,y, visualisation_label(node) , color = "white", bbox = dict(facecolor="red", edgecolor = "black"))
        

fig, graph = plt.subplots(figsize=(150, 70))
tree_visual(root_node, graph, 1, 1500,10,0,0)
plt.axis('off')
plt.savefig("visualisation.png")