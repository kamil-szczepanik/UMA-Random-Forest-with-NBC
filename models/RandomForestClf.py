import pandas as pd
import numpy as np
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from scipy.stats import mode

import sys
sys.path.append('..\\models')
from ID3 import ID3
from NBC_Categorical import NBC_Categorical

class RandomForestClf:

    def __init__(self, n_clf=100, clf_list=[ID3, NBC_Categorical], clf_ratio=[0.5, 0.5], percent_samples = 0.75, percent_attributes=0.75):
        self.n_clf = n_clf
        self.clf_list = clf_list
        self.clf_ratio = clf_ratio
        self.percent_samples = percent_samples
        self.percent_attributes = percent_attributes
        self.forest = []
        self.attributes_for_clf = []

    def fit(self, X_train, y_train):
        assert len(self.clf_list)==len(self.clf_ratio), "Argument 'clf_list' must be the same length as argument 'clf_ratio'"
        for clf_class, clf_ratio in zip(self.clf_list, self.clf_ratio):
            for i in range(round(self.n_clf*clf_ratio)):
                clf = clf_class()
                X_train_bagging, y_train_bagging = self.bagging_data(X_train, y_train)
                clf.fit(X_train_bagging, y_train_bagging)
                self.forest.append(clf)

    def bagging_data(self, X_train, y_train):
        train_df = X_train.sample(frac=self.percent_attributes, axis=1) # attributes randomization
        self.attributes_for_clf.append(train_df.columns)
        # print(self.attributes_for_clf[-1])
        train_df['label'] = y_train
        train_df = train_df.sample(frac=self.percent_samples)
        return train_df.drop(columns=['label']), train_df['label']

    def predict(self, X_test):
        predictions = np.empty([len(self.forest), len(X_test)], dtype='<U32')
        for i in range (len(self.forest)):
            X_test_i = X_test[self.attributes_for_clf[i]]
            y_pred = self.forest[i].predict(X_test_i)
            predictions[i] = np.array(y_pred, dtype=str)
        return mode(predictions, keepdims=False)[0]
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_test = np.array(y_test, dtype='<U32')
        acc = metrics.accuracy_score(y_test, y_pred)
        return acc
    
    def eval(self, X_test, y_test):
        acc = self.score(X_test, y_test)
        print('Accuracy:', acc)
        return acc