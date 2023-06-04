import pandas as pd
import numpy as np
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from scipy.special import logsumexp

class NBC_Categorical:
    def __init__(self, alpha=1.0e-10) -> None:
        self.alpha = alpha
        self.prior_probabilities = None
        self.conditional_probabilities = None
        self.labels = None
        self.attributes = None

    def merge_train_data(self, X_train, y_train):
        train_df = X_train.copy()
        train_df['label'] = y_train.copy().astype(str)
        return train_df
    
    def get_prior_probabilities(self, train_df):
        return train_df.groupby(['label']).size().div(train_df.shape[0])

    def get_conditional_probabilities(self, train_df):
        attr_probabs = {}

        for attribute_name in self.attributes:
            attribute_values = pd.unique(train_df[attribute_name])
            attr_probabs[attribute_name] = {}
            num_of_categories_of_attribute = len(pd.unique(train_df[attribute_name]))

            for attr_val in attribute_values:
                attr_probabs[attribute_name][attr_val] = {}
                mask_attribute = train_df[attribute_name] == attr_val
                train_attr = train_df[mask_attribute]
                probs_attr_label = {}
                for label in self.labels:
                    mask_label = train_df["label"] == label
                    train_label = train_df[mask_label]
                    train_attr_label = train_df[mask_attribute & mask_label]
                    probs_attr_label[label] = (len(train_attr_label)+self.alpha)/(len(train_label)+ self.alpha*num_of_categories_of_attribute)

                attr_probabs[attribute_name][attr_val] = probs_attr_label

        return attr_probabs
    
    def fit(self, X_train, y_train):
        train_df = self.merge_train_data(X_train,y_train)
        self.labels = pd.unique(train_df["label"])
        self.attributes = train_df.drop(columns=['label']).columns
        self.prior_probabilities = self.get_prior_probabilities(train_df)
        self.conditional_probabilities = self.get_conditional_probabilities(train_df)

    def predict_instance(self, instance): # instance = row
        probabilities = dict()

        for label in self.labels:
            label_probability = self.prior_probabilities[label]
            for attr_name, attr_val in instance.items():
                if attr_val in self.conditional_probabilities[attr_name].keys():
                    label_probability *= self.conditional_probabilities[attr_name][attr_val][label]
                else:
                    label_probability *= 0

            probabilities[label] = label_probability

        labels_probabilites = {}
        for label, probab in probabilities.items():
            # labels_probabilites[label] = probab/np.sum(list(probabilities.values()))
            labels_probabilites[label] = np.exp(logsumexp(probab) -  logsumexp(list(probabilities.values()) ))

        
        return max(labels_probabilites, key=labels_probabilites.get)
    
    def predict(self, X_test):
        preds_df = X_test.apply(lambda x: self.predict_instance(x), axis=1)
        return preds_df
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_test = np.array(y_test, dtype=str)
        y_pred = np.array(y_pred, dtype=str)
        acc = metrics.accuracy_score(y_test, y_pred)
        return acc
    
    def eval(self, X_test, y_test):
        acc = self.score( X_test, y_test)
        print('Accuracy:', acc)
        return acc