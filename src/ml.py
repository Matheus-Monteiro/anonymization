import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.anon import anonimization_clustering


def cross_validate_k_fold(
    X, y, anon_training, anon_test, model, model_name, n_clusters
):
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

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))

    results = {
        "accuracy": np.array(accuracy),
        "precision": np.array(precision),
        "recall": np.array(recall),
        "f1_score": np.array(f1),
    }

    return [
        model_name,
        anon_training,
        anon_test,
        results["accuracy"].mean(),
        results["precision"].mean(),
        results["recall"].mean(),
        results["f1_score"].mean(),
    ]


def get_results(model, X, y, model_name, n_clusters):
    bol = [True, False]
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
    for i in range(0, 2):
        for j in range(0, 2):
            new_df = pd.DataFrame(
                [
                    cross_validate_k_fold(
                        X, y, bol[i], bol[j], model, model_name, n_clusters
                    ),
                ],
                columns=results.columns,
            )

            results = pd.concat([results, new_df], ignore_index=True)

    for i in range(3, len(results.columns)):
        col_name = results.columns[i]
        results[col_name] = results[col_name].apply(
            lambda row: "{:.3%}".format(float(row))
        )

    return results


def experiment(
    models: dict[str, any], X: pd.DataFrame, y: pd.DataFrame
) -> pd.DataFrame:
    final_result = []

    for model_name, model in models.items():
        result_model = get_results(model, X, y, model_name, 3)
        final_result.append(result_model)
        print(f"{model_name} Trained")

    return pd.concat(final_result, ignore_index=True)
