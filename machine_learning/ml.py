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

from machine_learning.anonymization.anon import anonimization_clustering

def cross_validate_k_fold(X, y, anon_training, anon_test, model, model_name, n_clusters):
    kf = StratifiedKFold(n_splits=3)
    scaler = StandardScaler()

    accuracy, precision, recall, f1 = [], [], [], []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if anon_training == True:
            X_train, y_train = anonimization_clustering(X_train, y_train, n_clusters)

        if anon_test == True:
            X_test, y_test = anonimization_clustering(X_test, y_test, n_clusters)

        scaler.fit(X_train)
        scaler.fit(X_test)
            
        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))

    results = {'accuracy' : np.array(accuracy), 
           'precision' : np.array(precision),
           'recall' : np.array(recall), 
           'f1_score' : np.array(f1)}

    print(model_name, anon_training, anon_test)
    for k in results.keys():
        if k != 'fit_time' and k != 'score_time':
            print(k, '---> mean:', results[k].mean(), ' |  std:', results[k].std())

    return [ anon_training, anon_test, results['accuracy'].mean(), results['precision'].mean(), results['recall'].mean(), results['f1_score'].mean() ]

if __name__ == '__main__':
    pass