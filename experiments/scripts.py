from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
# from tqdm.notebook import tqdm
from tqdm import tqdm
from itertools import product

from models.RandomForestClf import RandomForestClf
from experiments.datasets import get_airline_dataset, get_exams_dataset, get_ecommerce_dataset


def cross_validation_score(X, y, model, n_splits=5):
    kf =KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # split()  method generate indices to split data into training and test set.
    acc_scores = []
    f1_scores = []
    for train_index, test_index in kf.split(X, y):
        train_X, train_y = X.loc[train_index,:], y.loc[train_index]
        test_X, test_y = X.loc[test_index, :], y.loc[test_index]
        model.fit(train_X, train_y)
        acc, f1 = model.scores(test_X, test_y)
        acc_scores.append(acc)
        f1_scores.append(f1)
    
    acc_score = round(np.mean(acc_scores), 4)
    f1_score = round(np.mean(f1_scores), 4)
    return acc_score, f1_score

def test_accuracy(X, y, model):
    return model.scores(X, y)

def get_conf_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

def run_experiment(result_file_name, experiment_repetitions, 
               dataset_loadDataset_valMethod, 
               model_param_attribute_part, 
               model_param_instances_per_classifier,
               model_param_id3_to_NBC,
               model_param_num_of_classifiers):


    models_results_df = pd.DataFrame(columns=["model", "accuracy", "f1 score", "dataset", "attribute part", "instances per classifier", "id3 to NBC", "num of classifiers"])

    # Parameter Loops
    for dataset_name, load_func in dataset_loadDataset_valMethod:
        print("DATASET:", dataset_name)
        if dataset_name=="airline":
            (X_train, y_train), (X_test, y_test) = get_airline_dataset("train"), get_airline_dataset("test")
        else:
            X, y = load_func()
            
        params_prod = list(product(model_param_attribute_part, 
                                model_param_instances_per_classifier, 
                                model_param_id3_to_NBC, 
                                model_param_num_of_classifiers))
        for m_p_attribute_part,m_p_instances_per_classifier,m_p_id3_to_NBC,m_p_num_of_classifiers in tqdm(params_prod, total=len(params_prod), desc="Parameters variations", position=0):
                                        
            acc_scores = []
            f1_scores = []
            # Experiment repetitions loop
            for i in tqdm(range(experiment_repetitions), total=experiment_repetitions, desc="Experiment repetitions", position=1, leave=False) :
                model = RandomForestClf(n_clf=m_p_num_of_classifiers, clf_ratio=m_p_id3_to_NBC, percent_samples = m_p_instances_per_classifier, percent_attributes=m_p_attribute_part)
                # model = NBC_Categorical()
                if dataset_name=="airline":
                    model.fit(X_train, y_train)
                    acc, f1 = model.scores(X_test, y_test)
                else:
                    acc, f1 = cross_validation_score(X, y, model)
                
                acc_scores.append(acc)
                f1_scores.append(f1)
                
            model_name = f"RandomForestClf"
            final_acc_score = np.mean(acc_scores)
            final_f1_score = np.mean(f1_scores)
            models_results_df.loc[len(models_results_df)] = [model_name, 
                                                            final_acc_score, 
                                                            final_f1_score,
                                                            dataset_name, 
                                                            m_p_attribute_part, 
                                                            m_p_instances_per_classifier, 
                                                            m_p_id3_to_NBC, 
                                                            m_p_num_of_classifiers]
            
            # models_results_df.to_csv(f"{result_file_name}.csv", index=False)
            
        models_results_df.to_csv(f"{result_file_name}.csv", index=False)
    
    return models_results_df