import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


class ID3:
    def __init__(self):
        self.tree = {}

    def total_entropy(df, labels):
        
        entropy = 0
        total_rows = df.shape[0]
        for label in df[labels].unique():
            label_count = df[df[labels] == label].shape[0]
            class_entropy = - (label_count/total_rows)*np.log2(label_count/total_rows)
            entropy += class_entropy
            
        return entropy
        
    def attribute_entropy(attribute_data, labels):
        attribute_entropy = 0
        attribute_rows = attribute_data.shape[0]
        
        for label in attribute_data[labels].unique():
            class_entropy = 0
            label_count = attribute_data[attribute_data[labels] == label].shape[0]
            if label_count != 0:
                class_entropy = - (label_count/attribute_rows)*np.log2(label_count/attribute_rows)            
                attribute_entropy += class_entropy
        
        return attribute_entropy

    def information_gain(df, attribute, labels):
        info_gain = 0
        total_rows = df.shape[0]
        for attr_val in df[attribute].unique():
            attr_val_count = df[df[attribute] == attr_val].shape[0]
            prop_attr_val = attr_val_count/total_rows
            info_gain += prop_attr_val * ID3.attribute_entropy(df.loc[df[attribute] == attr_val], labels)
        return ID3.total_entropy(df, labels) - info_gain
        

    def find_most_informative_attribute(df, labels):
        best_attr = None
        best_info_gain = 0
        for attribute in df.columns.drop([labels]):
            info_gain = ID3.information_gain(df, attribute, labels)
            if best_info_gain < info_gain:
                best_info_gain = info_gain
                best_attr = attribute
            
        return best_attr, best_info_gain

    
    def create_node(self, df, attribute_name, labels):
        attribute_node = {}

        labels_list = df[labels].unique()
        attribute_values = df[attribute_name].unique()
        
        for attribute_value in attribute_values:
            attribute_val_df = df.loc[df[attribute_name] == attribute_value]
            for label in labels_list:
                label_count = attribute_val_df[attribute_val_df[labels] == label].shape[0]
                if label_count == attribute_val_df.shape[0]: # pure class
                    attribute_node[attribute_value] = label
                    df = df[df[attribute_name] != attribute_value]
                    break
            else: # impure class
                attribute_node[attribute_value] = None
                
        return attribute_node, df
    
    
    def create_tree(self, df, labels, root, prev_attribute_val):
        
        attribute_name, _ = ID3.find_most_informative_attribute(df, labels)
        
    
        if attribute_name != None: # is not a non-devisable examples
            attribute_node, df = self.create_node(df, attribute_name, labels)
        
        if prev_attribute_val == None:
            root[attribute_name] = attribute_node
            next_root = root[attribute_name]
    
        else:
            if attribute_name == None:
                root[prev_attribute_val] = max(df[labels].value_counts().to_dict(), key=df[labels].value_counts().to_dict().get)
            else:
                root[prev_attribute_val] = dict()
                root[prev_attribute_val][attribute_name] = attribute_node
                next_root = root[prev_attribute_val][attribute_name]

        if attribute_name != None:
            for node, branch in next_root.items():
                if branch == None:
                    attribute_val_df = df[df[attribute_name] == node]
                    self.create_tree(attribute_val_df, labels, next_root, node)
    
    def fit(self, X_train, y_train):
        df = X_train.copy()
        labels = y_train.name
        df[labels] = y_train.copy().astype(str)
        df = df.sort_values(by=df.columns.tolist()).reset_index(drop=True)
        
        self.create_tree(df, labels, self.tree, None)
        
    def predict_instance(self, root, instance, default='?'):
        if isinstance(root, dict):
            attribute = next(iter(root))
            attribute_value = instance[attribute]
            if attribute_value in root[attribute]:
                return self.predict_instance(root[attribute][attribute_value], instance)
            else:
                # tree has not seen this kind of data in the training set
                # return first attribute value - deterministic solution
                # return self.predict_instance(root[attribute][list(root[attribute].keys())[0]], instance)
                return default
        else:  # not dict so it's a leaf
            return root
        
    def predict(self, X_test):
        # preds = []
        # for index, x in X_test.iterrows():
        #     prediction = self.predict_instance(self.tree, x)
        #     preds.append(prediction)
        # return np.array(preds)
        preds_df = X_test.apply(lambda x: self.predict_instance(self.tree, x), axis=1)
        return preds_df
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred = np.array(y_pred, dtype=str)
        y_test = np.array(y_test, dtype=str)
        acc = metrics.accuracy_score(y_test, y_pred)
        return acc
    
    def eval(self, X_test, y_test):
        acc = self.score(X_test, y_test)
        print('Accuracy:', acc)
        return acc
    

