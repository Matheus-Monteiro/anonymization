import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as la
from joblib import Parallel, delayed
from machine_learning.ml import cross_validate_k_fold
from sklearn import model_selection, svm
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)

##### feature selection ######
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    pairwise_distances_argmin,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


def get_results(model, X, y, model_name, n_clusters):
    bol = [True, False]
    results = pd.DataFrame(
        columns=[
            "anonymized train",
            "anonymized test",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]
    )
    for i in range(0, 2):
        for j in range(0, 2):
            new_df = pd.DataFrame(
                [
                    cross_validate_k_fold(
                        X, y, bol[i], bol[j], model, model_name, n_clusters
                    )
                ],
                columns=results.columns,
            )
            results = pd.concat([results, new_df], ignore_index=True)

    for i in range(2, len(results.columns)):
        col_name = results.columns[i]
        results[col_name] = results[col_name].apply(
            lambda row: "{:.3%}".format(float(row))
        )

    return results


def experiment(X, y):
    # results using knn for classification with K = 5 and 3-anonimity model
    results_knn = get_results(KNeighborsClassifier(n_neighbors=5), X, y, "KNN", 3)
    print(results_knn)

    # results using decision tree for classification and 3-anonimity model
    results_dtree = get_results(DecisionTreeClassifier(), X, y, "Decision Tree", 3)
    print(results_dtree)

    # aggregate results
    results_knn.insert(0, "model", ["KNN", "KNN", "KNN", "KNN"], True)
    results_dtree.insert(
        0,
        "model",
        ["Decision Tree", "Decision Tree", "Decision Tree", "Decision Tree"],
        True,
    )
    results_rfc.insert(
        0,
        "model",
        ["Random Forest", "Random Forest", "Random Forest", "Random Forest"],
        True,
    )
    results_gnb.insert(
        0, "model", ["Gaussian NB", "Gaussian NB", "Gaussian NB", "Gaussian NB"], True
    )

    results = pd.DataFrame(
        columns=[
            "model",
            "anonymized train",
            "anonymized test",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]
    )

    results = pd.concat([results, results_knn], ignore_index=True)
    results = pd.concat([results, results_dtree], ignore_index=True)
    results = pd.concat([results, results_rfc], ignore_index=True)
    results = pd.concat([results, results_gnb], ignore_index=True)

    return results


if __name__ == "__main__":
    # set a random seed
    np.random.seed(7)

    # read the dataset
    dataset = pd.read_csv("data/df_original_100000.csv")

    print(len(dataset.columns))

    # extract the labels
    y = np.array(dataset["Label"])
    del dataset["Label"]
    X = np.array(dataset)

    # -----------------------------------  Chi 2 --------------------------------------------- #
    # for i in range(2, 11, 1):
    #     k = i  # Number of top features to select
    #     selector_ch2 = SelectKBest(chi2, k=k)
    #     X_new_ch2 = selector_ch2.fit_transform(X, y)
    #     # print the shape of the dataset
    #     print(X_new_ch2.shape)
    #     # run the experiment
    #     print(f"------------------------------------------CHI2 {i} -------------------------------------------------")
    #     results = experiment(X_new_ch2, y)
    #     print(results)

    # ------------------------------------- Lasso ---------------------------------------------------- #

    # alpha = 0.01  # Adjust regularization strength
    # lasso = Lasso(alpha=alpha)
    # lasso.fit(X, y)
    # non_zero_coefs = np.where(lasso.coef_ != 0)[0]
    # X_selected = X[:, non_zero_coefs]
    # print("XShape",X.shape)
    # print("Lasso Shape",X_selected.shape)
    # # run the experiment
    # results = experiment(X_selected, y)
    # print("------------------------------------------LASSO -------------------------------------------------")
    # print(results)

    ##-------------------------------------------- Low Variance ----------------------------------------------

    # # Criando o seletor de características com baixa variância
    # threshold = 0.1  # Ajuste este valor conforme necessário
    # variance_selector = VarianceThreshold(threshold)

    # # Aplicando o seletor aos dados de treinamento
    # X_selected = variance_selector.fit_transform(X)

    # # Imprimindo a forma (shape) do conjunto de dados após a seleção de características
    # print("Low Variance Shape", X_selected.shape)
    # results = experiment(X_selected, y)
    # print("------------------------------------------Low Varaince  -------------------------------------------------")
    # print(results)

    ##-------------------------------------------- Extra Tree Classifier ----------------------------------------------

    # for i in range(10,80,5):
    #     extra_trees_model = ExtraTreesClassifier(
    #         n_estimators=100,
    #         criterion='gini',
    #         max_depth=None,
    #         max_features= i,  # Ou um número específico de características, por exemplo, 0.5 para 50%
    #     )

    #     # Treinando o modelo
    #     extra_trees_model.fit(X, y)

    #     # Criando o seletor de características baseado nas importâncias
    #     feature_selector = SelectFromModel(extra_trees_model)

    #     # Aplicando o seletor aos dados de treinamento
    #     X_selected = feature_selector.fit_transform(X, y)

    #     results = experiment(X_selected, y)
    #     print(f"------------------------------------------Extra tree {i} -------------------------------------------------")
    #     print(results)

    # # -----------------------------------  ANOVA  --------------------------------------------- #
    # for i in range(2, 11, 1):
    #     k = i  # Number of top features to select
    #     selector_ch2 = SelectKBest(f_classif, k=k)
    #     X_selected = selector_ch2.fit_transform(X, y)
    #     # print the shape of the dataset
    #     print(X_selected.shape)
    #     # run the experiment
    #     print(f"------------------------------------------ANOVA {i} -------------------------------------------------")
    #     results = experiment(X_selected, y)
    #     print(results)

    # # -----------------------------------  Mutual INFORMATION  --------------------------------------------- #
    for i in range(20, 80, 5):
        k_best = i  # Escolha o número desejado de características
        selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
        X_selected = selector.fit_transform(X, y)
        # print the shape of the dataset
        # print(X_selected.shape)
        # run the experiment
        print(
            f"------------------------------------------Mutual INFORMATION {i} -------------------------------------------------"
        )
        results = experiment(X_selected, y)
        print(results)
