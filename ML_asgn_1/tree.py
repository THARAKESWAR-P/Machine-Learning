from cgi import test
from posixpath import split
from this import d
import numpy as np
import utils
from graphviz import Digraph

NODE_COUNT=0
class Node:
    '''
        Class defines the nodes of the decision tree
        self.attr: [String] attribute of the node
        self.val: [Float] leaf_value of the attribute
        self.avg_attr: [Float] average of the attribute

        self.left: [Node] left child of the node
        self.right: [Node] right child of the node

    '''

    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, vari_reduction=None, leaf_value=None):
        '''
            Initializes the node
        '''
        global NODE_COUNT
        NODE_COUNT+=1
        self.node_id = NODE_COUNT
        
        # for decision node
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.vari_reduction = vari_reduction
        
        # for leaf node
        self.leaf_value = leaf_value
    
    def dfs_count(self):
        ans = 1
        if not self.left is None:
            ans += self.left.dfs_count()
        if not self.right is None:
            ans += self.right.dfs_count()
        return ans

    def dfs_depth(self):
        ans = 1
        if not self.left is None:
            ans = max(ans, 1+self.left.dfs_depth())
        if not self.right is None:
            ans = max(ans, 1+self.right.dfs_depth())
        return ans

class RTRegressor():
    def __init__(self, min_samples_split=10, max_depth=10):
        self.root = None 
        self.prune_root = None
        self.node_count = 0
        
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def construct_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            best_split = self.find_best_split(dataset, num_samples, num_features)
            if best_split["vari_reduction"]>0:
                left_subtree = self.construct_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.construct_tree(best_split["dataset_right"], curr_depth+1)
                self.node_count = NODE_COUNT
                return Node(best_split["feature_idx"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["vari_reduction"])
        
        leaf_value = self.calculate_leaf_value(Y)
        self.node_count = NODE_COUNT
        return Node(leaf_value=leaf_value)
    
    def find_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        var_red_max = -float("inf")
        for feature_idx in range(num_features):
            feature_values = dataset[:, feature_idx]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_idx, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # update the best split if needed
                    if curr_var_red>var_red_max:
                        best_split["feature_idx"] = feature_idx
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["vari_reduction"] = curr_var_red
                        var_red_max = curr_var_red
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_idx, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_idx]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_idx]>threshold])
        return dataset_left, dataset_right

    def prune_tree(self, root, dataset, test_dataset):
        if(root.leaf_value):
            return root

        prune_root = Node(leaf_value=average(dataset))
        actual_acc = self.find_MSE(test_dataset)
        prune_acc = self.find_MSE_prune(prune_root, test_dataset)
        print(f"{prune_root.node_id}, {root.node_id}: ", actual_acc, prune_acc, prune_root.leaf_value)

        if prune_acc <= actual_acc:
            print(prune_root.node_id)
            root = prune_root
        else:
            dataset_left, dataset_right = self.split(dataset, root.feature_idx, threshold=root.threshold)
            test_dataset_left, test_dataset_right = self.split(test_dataset, root.feature_idx, root.threshold)
            root.left = self.prune_tree(root.left, dataset_left, test_dataset_left)
            root.right = self.prune_tree(root.right, dataset_right, test_dataset_right)
        return root
    
    def pruning(self, dataset, test_dataset):
        global NODE_COUNT
        NODE_COUNT = 0
        self.root = self.prune_tree(self.root, dataset, test_dataset)
        return self.root
    
    def variance_reduction(self, parent, l_c, r_c):
        w_l = len(l_c) / len(parent)
        w_r = len(r_c) / len(parent)
        redc = np.var(parent) - (w_l * np.var(l_c) + w_r * np.var(r_c))
        return redc
    
    def calculate_leaf_value(self, Y):
        val = np.mean(Y)
        return val
                
    def print_tree(self, tree=None, gap=" ", feature_names=[]):
        temp_fp = feature_names
        if not tree:
            tree = self.root
        if tree.leaf_value is not None:
            print(tree.leaf_value)
        else:
            print(feature_names[tree.feature_idx], "<=", tree.threshold, "?", tree.vari_reduction)
            print("%sleft:" % (gap), end="")
            self.print_tree(tree.left, gap + gap, feature_names=temp_fp)
            print("%sright:" % (gap), end="")
            self.print_tree(tree.right, gap + gap, feature_names=temp_fp)
    
    def fit(self, X, Y):
        global NODE_COUNT
        NODE_COUNT = 0
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.construct_tree(dataset)
        
    def get_prediction(self, x, tree):
        if tree.leaf_value!=None: return tree.leaf_value
        feature_val = x[tree.feature_idx]
        if feature_val<=tree.threshold:
            return self.get_prediction(x, tree.left)
        else:
            return self.get_prediction(x, tree.right)
    
    def predict(self, X):
        predictions = [self.get_prediction(x, self.root) for x in X]
        return predictions

    def predict_prune(self, root, X):
        predictions = [self.get_prediction(x, root) for x in X]
        return predictions

    def find_accuracy(self, X, y):
        y_pred = self.predict(X)
        RMSE = np.sqrt(np.sum(((y-y_pred)**2))/len(y))
        return RMSE

    def find_MSE(self, dataset):
        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values.reshape(-1,1)
        y_pred = self.predict(X)
        MSE = (np.sum(((Y-y_pred)**2))/len(Y))
        return MSE

    def find_MSE_prune(self, root, dataset):
        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values.reshape(-1,1)
        y_pred = self.predict_prune(root, X)
        MSE = (np.sum(((Y-y_pred)**2))/len(Y))
        return MSE

    def count_nodes(self):
        return self.root.dfs_count()

    def find_depth(self):
        return self.root.dfs_depth()

    def r_node(self, vertex, feature_names, count):
        if vertex.leaf_value:
            return f'ID {vertex.node_id}, \nleaf leaf_value -> {vertex.leaf_value}\n'
        return f'ID {vertex.node_id}, \n{feature_names[vertex.feature_idx]} <= {vertex.threshold} ? {vertex.vari_reduction}\n'
    
    def convert_to_gv(self, feature_names=[], tree=None, file_name="regression_tree.gv"):
        f = Digraph('Regression Tree', filename=file_name)
        f.attr('node', shape='rectangle') 
        if not tree:
            tree = self.root
        q = [tree]
        index = 0
        while len(q) > 0:
            node = q.pop(0)
            if node is None:
                continue
            if not node.left is None:
                f.edge(self.r_node(node, feature_names, index), self.r_node(
                    node.left, feature_names, index), label='True')
                index += 1
                q.append(node.left)
            if not node.right is None:
                f.edge(self.r_node(node, feature_names, index), self.r_node(
                    node.right, feature_names, index), label='False')
                index += 1
                q.append(node.right)
        f.render(f'./{file_name}', view=True)


def average(dataset):
        Y = dataset.iloc[:, -1].values.reshape(-1,1)
        return np.mean(Y)



