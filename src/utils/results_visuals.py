import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mlflow
from src.utils.logging_config import logger


def plot_and_save_grouped_bar_mlruns_metrics(experiment_name, metrics=("r2", "mse", "rmse", "mae"), output_dir="mlruns_results"):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No experiment found with name '{experiment_name}'")
    experiment_id = experiment.experiment_id

    df = mlflow.search_runs(experiment_ids=experiment_id)

    if df.empty:
        raise ValueError("No runs found in the experiment.")

    os.makedirs(output_dir, exist_ok=True)

    available_metrics = [metric for metric in metrics if f"metrics.{metric}" in df.columns]
    if not available_metrics:
        raise ValueError("None of the specified metrics are found in the runs.")

    run_names = df['tags.mlflow.runName'].fillna(df['run_id'])
    metric_data = pd.DataFrame(index=run_names)

    for metric in available_metrics:
        metric_column = f"metrics.{metric}"
        metric_data[metric.upper()] = df[metric_column].values

    metric_data = metric_data.sort_index()
    num_metrics = len(metric_data.columns)
    num_runs = len(metric_data)

    x = np.arange(num_metrics)
    bar_width = 0.8 / num_runs

    plt.figure(figsize=(max(14, num_metrics * 2), 8))

    for i, run_name in enumerate(metric_data.index):
        values = metric_data.loc[run_name].values
        bar_positions = x + i * bar_width
        bars = plt.bar(bar_positions, values, bar_width, label=run_name)

        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.4f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, rotation=40)

    plt.xticks(x + bar_width * (num_runs - 1) / 2, metric_data.columns)
    plt.ylabel('Metric Value')
    plt.title('Grouped Metrics by MLflow Run')
    plt.legend(title='Run', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "grouped_metrics_bar_chart.jpg")
    plt.savefig(plot_path, format="jpg", dpi=300)
    plt.close()
    logger.info(f"Saved grouped bar chart to {plot_path}")

def plot_and_save_best_models_summary(experiment_name, metrics=("r2", "mse", "rmse", "mae"), output_dir="mlruns_results"):

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No experiment found with name '{experiment_name}'")
    experiment_id = experiment.experiment_id

    df = mlflow.search_runs(experiment_ids=experiment_id)

    if df.empty:
        raise ValueError("No runs found in the experiment.")

    os.makedirs(output_dir, exist_ok=True)

    run_names = df['tags.mlflow.runName'].fillna(df['run_id'])
    metric_data = pd.DataFrame(index=run_names)

    available_metrics = [metric for metric in metrics if f"metrics.{metric}" in df.columns]
    if not available_metrics:
        raise ValueError("None of the specified metrics are found in the runs.")

    for metric in available_metrics:
        metric_column = f"metrics.{metric}"
        metric_data[metric.upper()] = df[metric_column].values

    best_models = {}
    for metric in metric_data.columns:
        if metric == "R2":
            best_idx = metric_data[metric].idxmax()  # higher is better
        else:
            best_idx = metric_data[metric].idxmin()  # lower is better
        best_models[metric] = (best_idx, metric_data.loc[best_idx, metric])

 
    plt.figure(figsize=(10, 6))
    metric_names = list(best_models.keys())
    model_names = [best_models[m][0] for m in metric_names]
    values = [best_models[m][1] for m in metric_names]

    bars = plt.bar(metric_names, values, color='skyblue')
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height(),
                 f"{model_names[i]}\n{values[i]:.4f}",
                 ha='center', va='bottom', fontsize=9, rotation=40)

    plt.title("Best Model per Metric")
    plt.ylabel("Metric Value")
    plt.tight_layout()

    summary_path = os.path.join(output_dir, "best_models_summary.jpg")
    plt.savefig(summary_path, format="jpg", dpi=300)
    plt.close()
    logger.info(f"Saved best models summary to {summary_path}")
