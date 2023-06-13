import time
from datetime import timedelta
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
from tqdm import tqdm

from joblib import Parallel, delayed
from sklearn import svm, model_selection
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans

from machine_learning.ml import cross_validate_k_fold

def get_results(model, X, y, model_name, n_clusters):
    bol = [True, False]
    results = pd.DataFrame(columns=['anonymized train', 'anonymized test' , 'accuracy', 'precision', 'recall', 'f1_score'])
    for i in range(0, 2):
        for j in range(0, 2):
            new_df = pd.DataFrame([cross_validate_k_fold(X, y, bol[i], bol[j], model, model_name, n_clusters)], columns=results.columns)
            results = pd.concat([results, new_df], ignore_index=True)

    for i in range(2, len(results.columns)):
        col_name = results.columns[i]
        results[col_name] = results[col_name].apply(lambda row:"{:.3%}".format(float(row)))

    return results

def experiment(X, y):
    # results using knn for classification with K = 5 and 3-anonimity model
    results_knn = get_results(KNeighborsClassifier(n_neighbors=5), X, y, 'KNN', 3)
    print(results_knn)

    # results using decision tree for classification and 3-anonimity model
    results_dtree = get_results(DecisionTreeClassifier(), X, y, 'Decision Tree', 3)
    print(results_dtree)

    # results using random forest for classification and 3-anonimity model
    results_rfc = get_results(RandomForestClassifier(n_estimators=100), X, y, 'Random Forest', 3)
    print(results_rfc)

    # results using Gaussian NB for classification and 3-anonimity model
    results_gnb = get_results(GaussianNB(var_smoothing=1e-02), X, y, 'GaussianNB', 3)
    print(results_gnb)


    # aggregate results
    results_knn.insert(0, "model", ['KNN', 'KNN', 'KNN', 'KNN'], True)
    results_dtree.insert(0, "model", ['Decision Tree', 'Decision Tree', 'Decision Tree', 'Decision Tree'], True)
    results_rfc.insert(0, "model", ['Random Forest', 'Random Forest', 'Random Forest', 'Random Forest'], True)
    results_gnb.insert(0, "model", ['Gaussian NB', 'Gaussian NB', 'Gaussian NB', 'Gaussian NB'], True)

    results = pd.DataFrame(columns=['model', 'anonymized train', 'anonymized test' , 'accuracy', 'precision', 'recall', 'f1_score'])

    results = pd.concat([results, results_knn], ignore_index=True)
    results = pd.concat([results, results_dtree], ignore_index=True)
    results = pd.concat([results, results_rfc], ignore_index=True)
    results = pd.concat([results, results_gnb], ignore_index=True)

    return results

if __name__ == '__main__':

    # set a random seed
    np.random.seed(7)

    # read the dataset
    dataset = pd.read_csv('data/df_original_100000.csv')

    # extract the labels
    y = np.array(dataset['Label'])
    del dataset['Label']
    X = np.array(dataset)

    # print the shape of the dataset
    print(X.shape)

    # run the experiment 
    results = experiment(X, y)

    print(results)
    