# Author: Zuzanna Górecka
import pandas as pd
import numpy as np
from sklearn import metrics
from models.ID3 import ID3
from models.NBC_Categorical import NBC_Categorical

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
                X_train_bagging, y_train_bagging = self.bagging_data(X_train.copy(), y_train.copy())
                clf.fit(X_train_bagging.copy(), y_train_bagging.copy())
                self.forest.append(clf)

    def bagging_data(self, X_train, y_train):
        train_df = X_train.copy()
        train_df = X_train.sample(frac=self.percent_attributes, axis=1)
        self.attributes_for_clf.append(train_df.columns)
        train_df['label'] = y_train
        train_df = train_df.sample(frac=self.percent_samples)
        return train_df.drop(columns=['label']), train_df['label']

    def predict(self, X_test):
        predictions = np.empty([len(self.forest), len(X_test)], dtype='<U32')
        for i in range (len(self.forest)):
            X_test_i = X_test[self.attributes_for_clf[i]]
            y_pred = self.forest[i].predict(X_test_i)
            predictions[i] = np.array(y_pred, dtype=str)
        pred_df = pd.DataFrame(predictions)
        pred_df = pd.DataFrame.mode(pred_df, axis=0).T
        return pd.Series(pred_df[0])
    
    def scores(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred = np.array(y_pred, dtype='<U32')
        y_test = np.array(y_test, dtype='<U32')
        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='macro')
        return acc, f1
    
    def eval(self, X_test, y_test):
        acc, f1 = self.scores(X_test, y_test)
        print('Accuracy:', acc)
        print("F1 score: ", f1)
        return acc, f1