from os.path import basename

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCENARIOS = [
    {"name": "not anonymized train \n not anonymized test", "index": 3},
    {"name": "anonymized train \n not anonymized test", "index": 1},
    {"name": "not anonymized train \n anonymized test", "index": 2},
    {"name": "anonymized train \n anonymized test", "index": 0},
]


def get_all_model_metric(result: pd.DataFrame, metric_name: str) -> dict[str, any]:
    model_names = result["model"].unique()

    metric_result = []

    for model_name in model_names:
        model_result = result.loc[result["model"] == model_name]
        metric_model = list(model_result[metric_name])
        metric_model = [
            float(metric.replace("%", "").replace(",", ".")) for metric in metric_model
        ]

        for scenario in SCENARIOS:
            dict_result = {
                "model": model_name,
                "scenario": scenario["name"],
                metric_name: metric_model[scenario["index"]],
            }
            metric_result.append(dict_result)

    return metric_result


def plot_by_feature_selection_method(
    title: str,
    metric_name: str,
    dataframe_file_path: str,
    plot_path: str,
    y_lim_bottom: int = 0,
) -> None:
    result = pd.read_csv(dataframe_file_path)

    metric_result = get_all_model_metric(result=result, metric_name=metric_name)

    metric_dataframe = pd.DataFrame(data=metric_result)

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        metric_dataframe, x="scenario", y=metric_name, hue="model", linewidth=2.5
    )

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            format(height, ".2f"),
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
        )

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Scenario", fontsize=10, fontweight="bold")
    plt.ylabel(f"{metric_name.capitalize()} (%)", fontsize=10, fontweight="bold")
    plt.ylim(bottom=y_lim_bottom)
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()


def get_model_metrics_in_all_files(
    csv_files: list[str], model_name: str, metric_name: str
) -> dict[str, any]:
    metric_result = []

    for csv_file in csv_files:
        result = pd.read_csv(csv_file)
        model_result = result.loc[result["model"] == model_name]
        metric_model = list(model_result[metric_name])
        metric_model = [
            float(metric.replace("%", "").replace(",", ".")) for metric in metric_model
        ]

        feature_selection_method = basename(csv_file).replace("_result.csv", "")

        for scenario in SCENARIOS:
            dict_result = {
                "feature_selection_method": feature_selection_method,
                "scenario": scenario["name"],
                metric_name: metric_model[scenario["index"]],
            }
            metric_result.append(dict_result)

    return metric_result


def plot_by_model(
    csv_files: list[str],
    title: str,
    model_name: str,
    metric_name: str,
    plot_path: str,
    y_lim_bottom: int = 0,
) -> None:
    metric_result = get_model_metrics_in_all_files(
        csv_files=csv_files, model_name=model_name, metric_name=metric_name
    )

    metric_dataframe = pd.DataFrame(data=metric_result)

    plt.figure(figsize=(18, 6))

    ax = sns.barplot(
        metric_dataframe,
        x="scenario",
        y=metric_name,
        hue="feature_selection_method",
        linewidth=2.5,
    )

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            format(height, ".2f"),
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
        )

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Scenario", fontsize=10, fontweight="bold")
    plt.ylabel(f"{metric_name.capitalize()} (%)", fontsize=10, fontweight="bold")
    plt.ylim(bottom=y_lim_bottom)
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
