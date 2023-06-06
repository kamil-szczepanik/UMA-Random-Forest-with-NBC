import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report

class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""

    def __init__(self):
        self.label = None
        self.branches = []
        self.is_leaf = False


class ID3:
    """Decision Tree Classifier using ID3 algorithm."""

    def __init__(self):
        self.features = None
        self.root = None

    def _get_entropy(self, y):
        samples_count = len(y)
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / samples_count
        entropy = -np.sum(np.log2(probabilities) * probabilities)
        return entropy

    def _get_information_gain(self, X, y, feature_name):
        feature_id = self.features.tolist().index(feature_name)
        feature_vals = X[:, feature_id]
        unique_feature_vals, counts = np.unique(
            feature_vals, return_counts=True)
        y_subsets = [
            [y[i]
             for i, v in enumerate(feature_vals)
             if v == uv]
            for uv in unique_feature_vals
        ]

        info_gain_feature = sum([count / len(X) * self._get_entropy(y_subset)
                                 for count, y_subset in zip(counts, y_subsets)])
        info_gain = self._get_entropy(y) - info_gain_feature
        return info_gain

    def _get_most_informative_feature(self, X, y, feature_names):
        info_gains = [self._get_information_gain(X, y, feature_name)
                      for feature_name in feature_names]
        best_feature_name = feature_names[info_gains.index(max(info_gains))]
        return best_feature_name

    def _id3(self, X, y, feature_names):
        node = Node()

        # if all the example have the same class (pure node), return node
        if len(set(y)) == 1:
            node.is_leaf = True
            node.label = y[0]
            return node

        # if there are not more feature to compute, return node with the most probable class
        if len(feature_names) == 0:
            node.is_leaf = True
            unique_vals, counts = np.unique(y, return_counts=True)
            node.label = unique_vals[np.argmax(counts)]
            return node

        # else choose the feature that maximizes the information gain
        best_feature_name = self._get_most_informative_feature(
            X, y, feature_names)
        node.label = best_feature_name

        # value of the chosen feature for each instance
        best_feature_id = self.features.tolist().index(best_feature_name)
        feature_values = list(set(X[:, best_feature_id]))

        for feature_value in feature_values:
            branch = [feature_value, Node()]
            node.branches.append(branch)

            X_subset = X[X[:, best_feature_id] == feature_value]
            y_subset = y[X[:, best_feature_id] == feature_value]

            if len(X_subset) == 0:
                unique_vals, counts = np.unique(y, return_counts=True)
                branch[1].label = unique_vals[np.argmax(counts)]
            else:
                feature_names = [
                    a for a in feature_names if a != best_feature_name]
                branch[1] = self._id3(X_subset, y_subset, feature_names)
        return node

    def fit(self, X_train, y_train):
        feature_names = X_train.columns
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.features = np.array(feature_names)
        self.root = self._id3(np.array(X_train), np.array(y_train), feature_names)

    def predict(self, X_test):
        X_test = np.array(X_test)
        y_pred = [self._walk_down(self.root, sample) for sample in X_test]
        return pd.Series(np.array(y_pred))

    def score(self, X_test, y_test):
        X_test = np.array(X_test)
        y_pred = self.predict(X_test)
        y_pred = np.array(y_pred, dtype=str)
        y_test = np.array(y_test, dtype=str)
        acc = metrics.accuracy_score(y_test, y_pred)
        return acc
    
    def eval(self, X_test, y_test):
        acc = self.score(X_test, y_test)
        print('Accuracy:', acc)
        return acc

    def _walk_down(self, node, sample):
        if node.is_leaf:
            return node.label

        feature_name = node.label
        feature_id = self.features.tolist().index(feature_name)
        if node.branches:
            for b in node.branches:
                if b[0] == sample[feature_id]:
                    return self._walk_down(b[1], sample)

        return node.label

