"""
Reporting utilities for sentiment-classification experiments.

This module is responsible for persisting experiment results as:

- CSV files for tabular analysis
- JSON files for programmatic consumption
- Plain-text reports for human-readable experiment summaries
"""

import json


def save_comparison_csv(
    comparison_df,
    output_path
):
    """
    Save the model-comparison table as CSV.

    Performance metrics are rounded to four decimal places.
    Runtime is rounded to two decimal places.
    """

    comparison_export = comparison_df.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "CV F1"
    ]

    # Keep performance metrics at four decimal places
    for column in metric_columns:
        if column in comparison_export.columns:
            comparison_export[column] = (
                comparison_export[column].round(4)
            )

    # Runtime is easier to read with two decimal places
    if "Runtime (min)" in comparison_export.columns:
        comparison_export["Runtime (min)"] = (
            comparison_export["Runtime (min)"].round(2)
        )

    comparison_export.to_csv(
        output_path,
        index=False
    )

    return output_path


def save_comparison_json(
    results,
    output_path,
    total_runtime_minutes=None,
    metadata=None
):
    """
    Save experiment results as JSON.

    JSON is useful for later consumption by:
    - other Python scripts
    - APIs
    - dashboards
    - automated comparisons
    """

    models = []

    for result in results:

        model_result = {
            "model": result["Model"],
            "accuracy": round(
                float(result["Accuracy"]),
                4
            ),
            "precision": round(
                float(result["Precision"]),
                4
            ),
            "recall": round(
                float(result["Recall"]),
                4
            ),
            "f1": round(
                float(result["F1 Score"]),
                4
            ),
            "cv_f1": round(
                float(result["CV F1"]),
                4
            ),
            "runtime_minutes": round(
                float(result["Runtime (min)"]),
                2
            ),
            "best_parameters": result[
                "Best Parameters"
            ]
        }

        models.append(model_result)

    report_data = {
        "experiment": (
            "IMDb Sentiment Classification - "
            "Classical Models"
        ),
        "models": models
    }

    if metadata is not None:
        report_data["metadata"] = metadata

    if total_runtime_minutes is not None:
        report_data["total_runtime_minutes"] = round(
            float(total_runtime_minutes),
            2
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report_data,
            file,
            indent=4
        )

    return output_path


def save_experiment_report(
    results,
    comparison_df,
    output_path,
    total_runtime_minutes=None
):
    """
    Save a human-readable plain-text experiment report.

    The report contains:
    - best parameters
    - cross-validation F1
    - test metrics
    - training/search runtime
    - final model comparison
    """

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "IMDb Sentiment Classification\n"
        )

        report.write(
            "Classical Model Experiment Report\n"
        )

        report.write(
            "=" * 80 + "\n\n"
        )

        for result in results:

            report.write(
                f"Model: {result['Model']}\n"
            )

            report.write(
                f"Best Parameters: "
                f"{result['Best Parameters']}\n"
            )

            report.write(
                f"Cross-Validation F1: "
                f"{result['CV F1']:.4f}\n"
            )

            report.write(
                f"Test Accuracy: "
                f"{result['Accuracy']:.4f}\n"
            )

            report.write(
                f"Test Precision: "
                f"{result['Precision']:.4f}\n"
            )

            report.write(
                f"Test Recall: "
                f"{result['Recall']:.4f}\n"
            )

            report.write(
                f"Test F1: "
                f"{result['F1 Score']:.4f}\n"
            )

            report.write(
                "Training + Hyperparameter Search "
                f"Runtime: "
                f"{result['Runtime (min)']:.2f} "
                "minutes\n"
            )

            report.write(
                "\n" + "-" * 80 + "\n\n"
            )

        report.write(
            "Final Model Comparison\n"
        )

        report.write(
            "=" * 80 + "\n\n"
        )

        formatted_comparison = (
            comparison_df.to_string(
                index=False,
                formatters={
                    "Accuracy": "{:.4f}".format,
                    "Precision": "{:.4f}".format,
                    "Recall": "{:.4f}".format,
                    "F1 Score": "{:.4f}".format,
                    "CV F1": "{:.4f}".format,
                    "Runtime (min)": "{:.2f}".format
                }
            )
        )

        report.write(
            formatted_comparison
        )

        report.write("\n")

        if total_runtime_minutes is not None:

            report.write(
                "\n" + "=" * 80 + "\n"
            )

            report.write(
                "Total Experiment Runtime: "
                f"{total_runtime_minutes:.2f} minutes\n"
            )

            report.write(
                "=" * 80 + "\n"
            )

    return output_path

def save_neural_comparison_json(
    results,
    output_path,
    total_runtime_minutes=None,
    fasttext_preparation_minutes=None,
    metadata=None
):
    """
    Save neural experiment results as JSON.

    Includes:
    - test metrics
    - parameter count
    - final training loss
    - training runtime
    """

    models = []

    for result in results:
        model_result = {
            "model": result["Model"],
            "accuracy": round(float(result["Accuracy"]), 4),
            "precision": round(float(result["Precision"]), 4),
            "recall": round(float(result["Recall"]), 4),
            "f1": round(float(result["F1 Score"]), 4),
            "parameters": int(result["Parameters"]),
            "final_training_loss": round(
                float(result["Final Training Loss"]),
                4
            ),
            "runtime_minutes": round(
                float(result["Runtime (min)"]),
                2
            )
        }

        models.append(model_result)

    report_data = {
        "experiment": "IMDb Sentiment Classification - Neural Models",
        "models": models
    }

    if metadata is not None:
        report_data["metadata"] = metadata

    if fasttext_preparation_minutes is not None:
        report_data["fasttext_preparation_minutes"] = round(
            float(fasttext_preparation_minutes),
            2
        )

    if total_runtime_minutes is not None:
        report_data["total_runtime_minutes"] = round(
            float(total_runtime_minutes),
            2
        )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report_data, file, indent=4)

    return output_path


def save_neural_experiment_report(
    results,
    comparison_df,
    output_path,
    total_runtime_minutes=None,
    fasttext_preparation_minutes=None
):
    """
    Save a human-readable neural experiment report.
    """

    with open(output_path, "w", encoding="utf-8") as report:
        report.write("IMDb Sentiment Classification\n")
        report.write("Neural Model Experiment Report\n")
        report.write("=" * 80 + "\n\n")

        for result in results:
            report.write(f"Model: {result['Model']}\n")
            report.write(f"Parameters: {result['Parameters']:,}\n")
            report.write(f"Test Accuracy: {result['Accuracy']:.4f}\n")
            report.write(f"Test Precision: {result['Precision']:.4f}\n")
            report.write(f"Test Recall: {result['Recall']:.4f}\n")
            report.write(f"Test F1: {result['F1 Score']:.4f}\n")
            report.write(
                f"Final Training Loss: "
                f"{result['Final Training Loss']:.4f}\n"
            )
            report.write(
                f"Training Runtime: "
                f"{result['Runtime (min)']:.2f} minutes\n"
            )
            report.write("\n" + "-" * 80 + "\n\n")

        if fasttext_preparation_minutes is not None:
            report.write(
                f"FastText Preparation Runtime: "
                f"{fasttext_preparation_minutes:.2f} minutes\n\n"
            )

        report.write("Final Neural Model Comparison\n")
        report.write("=" * 80 + "\n\n")

        formatted_comparison = comparison_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format,
                "Final Training Loss": "{:.4f}".format,
                "Runtime (min)": "{:.2f}".format
            }
        )

        report.write(formatted_comparison)
        report.write("\n")

        if total_runtime_minutes is not None:
            report.write("\n" + "=" * 80 + "\n")
            report.write(
                f"Total Experiment Runtime: "
                f"{total_runtime_minutes:.2f} minutes\n"
            )
            report.write("=" * 80 + "\n")

    return output_path