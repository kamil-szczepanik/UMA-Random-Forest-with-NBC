

import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import CategoricalNB
from matplotlib import pyplot as plt
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import numpy as np
# from tqdm.notebook import tqdm
from tqdm import tqdm
from itertools import product

from models.ID3 import ID3
from models.NBC_Categorical import NBC_Categorical
from models.RandomForestClf import RandomForestClf
from experiments.datasets import get_airline_dataset, get_exams_dataset, get_ecommerce_dataset
from experiments.scripts import cross_validation_score, test_accuracy, get_conf_matrix, run_experiment

import time

def experiment1():
    # Eksperyment 1
    # Porównanie własnej implementacji NBC z implementacją z biblioteki scikit-learn
    print('=================')
    print("Eksperyment 1\nPorównanie własnej implementacji NBC z implementacją z biblioteki scikit-learn")

    X, y = get_ecommerce_dataset()
    for col in X:
        X[col] = X[col].astype('category')
        X[col] = X[col].cat.codes
    y = y.astype('category')
    y = y.cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

    nbc_classifier = NBC_Categorical()
    start = time.time()
    nbc_classifier.fit(X_train, y_train)
    print('Czas fit() impl. wł. :', time.time()- start)
    y_pred =nbc_classifier.predict(X_test)
    our_preds = np.array(y_pred, dtype=int)
    our_nbc_acc = metrics.accuracy_score(y_test, our_preds)

    clf = CategoricalNB()
    start = time.time()
    clf.fit(X_train, y_train)
    print('Czas fit() sklearn:', time.time()- start)
    y_pred = clf.predict(X_test)
    sklearn_nbc_acc = metrics.accuracy_score(y_test, y_pred)

    print(f"Predykcje są takie same: {(our_preds==y_pred).all()}")
    print(f"Dokładność jest taka sama: {(our_preds==y_pred).all()} - {sklearn_nbc_acc}=={our_nbc_acc}")

def experiment2():
    # Eksperyment 2
    # 5.2.	Dobór parametrów lasu losowego
    # Eksperyment ten byl przeprowadzany z różnymi ustawieniami w celu dobrania parametrów.
    print('=================')
    print("Eksperyment 2\nDobór parametrów lasu losowego")
    experiment_repetitions = 1
    dataset_loadDataset_valMethod = [("e-commerce", get_ecommerce_dataset)] # USTAWIENIA ZBIORU DANYCH
    model_param_attribute_part = [0.25, 0.5, 0.75]                          # USTAWIENIA CZĘŚCI ATRYBUTÓW NA 1 KLASYFIKATOR
    model_param_instances_per_classifier = [0.5]                            # USTAWIENIA CZĘŚCI PRZYKŁADÓW NA 1 KLASYFIKATOR
    model_param_id3_to_NBC = [[0.50, 0.50]]                                 # USTAWIENIA PROPORCJI ID3 DO NBC
    model_param_num_of_classifiers = [96]                                   # LICZBA KLASYFIKATORÓW W LESIE

    exp1_models_results_df = run_experiment("exp2-model_param_attribute_part", experiment_repetitions, 
                dataset_loadDataset_valMethod, 
                model_param_attribute_part, 
                model_param_instances_per_classifier,
                model_param_id3_to_NBC,
                model_param_num_of_classifiers)
    
def experiment3():
    # Eksperyment 3
    #5.3.	Wyniki ewaluacji lasu losowego z dobranymi parametrami
    # Eksperyment ten wykonano dla trzech zbiorów danych
    # W rzeczywistości puszczano go pojedynczo dla każdego zbioru, na 5 notatnikach google colab po 5 powtórzen 
    # Tutaj jednocześnie robiony jest test klasycznej implementacji lasu losowego z samymi drzewami
    print('=================')
    print("Eksperyment 3\nWyniki ewaluacji lasu losowego z dobranymi parametrami")
    experiment_repetitions = 25
    dataset_loadDataset_valMethod = [("exams", get_ecommerce_dataset),("e-commerce", get_ecommerce_dataset), ("airline", get_airline_dataset)] 
    model_param_attribute_part = [0.75]
    model_param_instances_per_classifier = [1.0]
    model_param_id3_to_NBC = [[0.50, 0.50], [1, 0]]
    model_param_num_of_classifiers = [64]

    exp3_models_results_df = run_experiment("exp3", experiment_repetitions, 
                dataset_loadDataset_valMethod, 
                model_param_attribute_part, 
                model_param_instances_per_classifier,
                model_param_id3_to_NBC,
                model_param_num_of_classifiers)
    
def experiment4():
    # Eksperyment 4
    # 5.4.	Badanie wpływu parametru proporcji między rodzajami klasyfikatorów na zbiorze E-Commerce
    print('=================')
    print("Eksperyment 4\nBadanie wpływu parametru proporcji między rodzajami klasyfikatorów")
    experiment_repetitions = 1
    dataset_loadDataset_valMethod = [("e-commerce", get_ecommerce_dataset)] 
    model_param_attribute_part = [0.75]
    model_param_instances_per_classifier = [1.0]
    model_param_id3_to_NBC = [[0, 1], [0.25, 0.75], [0.50, 0.50], [0.75, 0.25], [1, 0]]
    model_param_num_of_classifiers = [96]

    exp4_models_results_df = run_experiment("exp4-model_param_id3_to_NBC", experiment_repetitions, 
                dataset_loadDataset_valMethod, 
                model_param_attribute_part, 
                model_param_instances_per_classifier,
                model_param_id3_to_NBC,
                model_param_num_of_classifiers)
    
def experiment_eval_ID3_NBC(dataset="exams"):
    print('\n=====================')
    print(f"Only ID3 and only NBC evaluation on dataset {dataset}")
    if dataset=="exams":
        X, y = get_exams_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
    elif dataset=="e-commerce":
        X, y = get_exams_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
    elif dataset=="airline":
        (X_train, y_train), (X_test, y_test) = get_airline_dataset("train"), get_airline_dataset("test")

    print("\nNaive Bayes Classifier:")
    nbc_classifier = NBC_Categorical()
    nbc_classifier.fit(X_train, y_train)
    nbc_classifier.eval(X_test, y_test)

    print("\nID3:")
    id3_tree = ID3()
    id3_tree.fit(X_train, y_train)
    id3_tree.eval(X_test, y_test)

if __name__=="__main__":
    experiment1()
    experiment2()
    experiment3()
    experiment4()
    experiment_eval_ID3_NBC("exams")
    experiment_eval_ID3_NBC("e-commerce")
    experiment_eval_ID3_NBC("airline")