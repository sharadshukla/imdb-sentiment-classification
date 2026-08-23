"""
Run the classical IMDb sentiment-classification experiments.

Models:
- Logistic Regression
- Bernoulli Naive Bayes
- LinearSVC
- Random Forest

Usage:
    Full experiment:
        python scripts/run_classical.py

    Smoke test:
        python scripts/run_classical.py --smoke-test
"""

from pathlib import Path
import argparse
import sys
import time

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from src.classical_models import (
    create_logistic_regression_search,
    create_naive_bayes_search,
    create_linearsvc_search,
    create_random_forest_search,
)
from src.data import load_imdb_data
from src.evaluation import evaluate_model
from src.preprocessing import clean_text
from src.reporting import (
    save_comparison_csv,
    save_comparison_json,
    save_experiment_report,
)
from src.validation import (
    ValidationTracker,
    is_balanced,
    is_valid_metric,
    print_smoke_mode_banner,
    select_balanced_subset,
)

# ---------------------------------------------------------------------
# Experiment Metadata
# ---------------------------------------------------------------------

experiment_metadata = {
    "run_type": "smoke" if args.smoke_test else "full",
    "train_samples": len(train_texts_clean),
    "test_samples": len(test_texts_clean),
    "cv_folds": SMOKE_CV_FOLDS if args.smoke_test else 5
}

# ---------------------------------------------------------------------
# Smoke-test configuration
# ---------------------------------------------------------------------

SMOKE_TRAIN_SIZE = 2000
SMOKE_TEST_SIZE = 1000
SMOKE_CV_FOLDS = 2


def parse_args():
    """
    Parse command-line options.
    """

    parser = argparse.ArgumentParser(
        description="Run classical IMDb sentiment-classification experiments."
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a fast integration test using a small balanced subset "
            "and a reduced hyperparameter search."
        )
    )

    return parser.parse_args()


def get_output_directories(smoke_test):
    """
    Return output directories for either a full run or smoke test.
    """

    run_type = "smoke" if smoke_test else "full"
    base_dir = PROJECT_ROOT / "results" / run_type / "classical"

    figure_dir = base_dir / "figures"
    metrics_dir = base_dir / "metrics"
    report_dir = base_dir / "reports"

    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    return figure_dir, metrics_dir, report_dir


def prepare_texts(train_texts, test_texts):
    """
    Apply the same text cleaning used in the original notebook.
    """

    print("\nCleaning IMDb reviews...")

    train_texts_clean = [clean_text(review) for review in train_texts]
    test_texts_clean = [clean_text(review) for review in test_texts]

    print("Text cleaning completed.")

    return train_texts_clean, test_texts_clean


def make_filename(model_name):
    """
    Convert a model name into a filesystem-friendly filename.
    """

    return model_name.lower().replace(" ", "_").replace("-", "_")


def configure_smoke_search(search):
    """
    Reduce GridSearchCV for fast smoke testing.

    Only the first value from each hyperparameter is used and
    cross-validation is reduced to two folds.

    The smoke test validates integration, not model quality.
    """

    reduced_grid = {
        parameter: [values[0]]
        for parameter, values in search.param_grid.items()
    }

    search.param_grid = reduced_grid
    search.cv = SMOKE_CV_FOLDS
    search.verbose = 0

    return search


def train_and_evaluate(
    model_name,
    search,
    train_texts,
    y_train,
    test_texts,
    y_test,
    figure_dir
):
    """
    Train one GridSearchCV model and evaluate its best estimator.
    """

    print("\n" + "=" * 80)
    print(f"Training: {model_name}")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Train + hyperparameter search
    # -----------------------------------------------------------------

    start_time = time.perf_counter()

    search.fit(train_texts, y_train)

    runtime = time.perf_counter() - start_time
    runtime_minutes = runtime / 60

    print(f"\nBest parameters: {search.best_params_}")
    print(f"Best cross-validation F1: {search.best_score_:.4f}")

    # -----------------------------------------------------------------
    # Test-set prediction
    # -----------------------------------------------------------------

    predictions = search.predict(test_texts)

    # -----------------------------------------------------------------
    # Evaluation + confusion-matrix artifact
    # -----------------------------------------------------------------

    figure_path = (
        figure_dir
        / f"{make_filename(model_name)}_confusion_matrix.png"
    )

    results = evaluate_model(
        model_name,
        y_test,
        predictions,
        figure_path=figure_path
    )

    # -----------------------------------------------------------------
    # Preserve experiment metadata
    # -----------------------------------------------------------------

    results["CV F1"] = search.best_score_
    results["Runtime (min)"] = runtime_minutes
    results["Best Parameters"] = search.best_params_
    results["Figure Path"] = figure_path

    print(
        f"\nTraining + hyperparameter search runtime: "
        f"{runtime_minutes:.2f} minutes"
    )

    return results


def build_comparison_table(all_results):
    """
    Build an F1-sorted classical-model comparison table.
    """

    comparison_df = pd.DataFrame([
        {
            "Model": result["Model"],
            "Accuracy": result["Accuracy"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1 Score": result["F1 Score"],
            "CV F1": result["CV F1"],
            "Runtime (min)": result["Runtime (min)"]
        }
        for result in all_results
    ])

    comparison_df = (
        comparison_df
        .sort_values(by="F1 Score", ascending=False)
        .reset_index(drop=True)
    )

    return comparison_df


def print_comparison_table(comparison_df):
    """
    Print the final classical-model comparison table.
    """

    print("\n" + "=" * 80)
    print("Final Classical Model Comparison")
    print("=" * 80)

    print(
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


def save_experiment_artifacts(
    all_results,
    comparison_df,
    total_runtime_minutes,
    metrics_dir,
    report_dir,
    smoke_test,
    experiment_metadata
):
    """
    Persist CSV, JSON, and human-readable experiment reports.
    """

    prefix = "smoke_" if smoke_test else ""

    csv_path = metrics_dir / f"{prefix}classical_model_comparison.csv"
    json_path = metrics_dir / f"{prefix}classical_model_comparison.json"
    report_path = report_dir / f"{prefix}classical_experiment_report.txt"

    save_comparison_csv(comparison_df, csv_path)

    save_comparison_json(
        all_results,
        json_path,
        total_runtime_minutes,
        metadata=experiment_metadata
    )

    save_experiment_report(
        all_results,
        comparison_df,
        report_path,
        total_runtime_minutes
    )

    print("\nArtifacts saved successfully:")
    print(f"CSV    : {csv_path}")
    print(f"JSON   : {json_path}")
    print(f"Report : {report_path}")

    return csv_path, json_path, report_path


def register_model_validation(validator, model_name, results):
    """
    Register smoke-test checks for one trained classical model.
    """

    validator.add(
        f"{model_name} F1 is valid",
        is_valid_metric(results["F1 Score"])
    )

    validator.add(
        f"{model_name} CV F1 is valid",
        is_valid_metric(results["CV F1"])
    )

    validator.add_artifact(
        f"{model_name} confusion matrix generated",
        results["Figure Path"]
    )


def main():
    """
    Run the complete classical experiment or its smoke test.
    """

    args = parse_args()

    figure_dir, metrics_dir, report_dir = get_output_directories(
        args.smoke_test
    )

    validator = (
        ValidationTracker("CLASSICAL PIPELINE")
        if args.smoke_test
        else None
    )

    total_start_time = time.perf_counter()

    print("=" * 80)
    print("IMDb Sentiment Classification - Classical Models")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Smoke-test banner
    # -----------------------------------------------------------------

    if args.smoke_test:
        print_smoke_mode_banner(
            experiment_type="Classical",
            train_size=SMOKE_TRAIN_SIZE,
            test_size=SMOKE_TEST_SIZE,
            extra_lines=[
                f"Cross-validation : {SMOKE_CV_FOLDS} folds",
                "Hyperparameters   : one configuration per model"
            ]
        )

    # -----------------------------------------------------------------
    # 1. Load IMDb
    # -----------------------------------------------------------------

    print("\nLoading Stanford IMDb dataset...")

    train_texts, y_train, test_texts, y_test = load_imdb_data()

    if args.smoke_test:
        train_texts, y_train = select_balanced_subset(
            train_texts,
            y_train,
            SMOKE_TRAIN_SIZE
        )

        test_texts, y_test = select_balanced_subset(
            test_texts,
            y_test,
            SMOKE_TEST_SIZE
        )

        validator.add(
            "IMDb dataset loaded",
            len(train_texts) == SMOKE_TRAIN_SIZE
            and len(test_texts) == SMOKE_TEST_SIZE
        )

        validator.add(
            "Balanced smoke-test labels",
            is_balanced(y_train) and is_balanced(y_test)
        )

    print(f"Training reviews: {len(train_texts):,}")
    print(f"Test reviews    : {len(test_texts):,}")

    # -----------------------------------------------------------------
    # 2. Clean text
    # -----------------------------------------------------------------

    train_texts_clean, test_texts_clean = prepare_texts(
        train_texts,
        test_texts
    )

    if args.smoke_test:
        validator.add(
            "Text preprocessing completed",
            len(train_texts_clean) == len(train_texts)
            and len(test_texts_clean) == len(test_texts)
        )

    # -----------------------------------------------------------------
    # 3. Define experiments
    # -----------------------------------------------------------------

    experiment_metadata = {
        "run_type": "smoke" if args.smoke_test else "full",
        "train_samples": len(train_texts_clean),
        "test_samples": len(test_texts_clean),
        "cv_folds": SMOKE_CV_FOLDS if args.smoke_test else 5,
        "scoring": "f1",
        "model_random_state": 42
    }
    experiments = [
        ("Logistic Regression", create_logistic_regression_search()),
        ("Bernoulli Naive Bayes", create_naive_bayes_search()),
        ("LinearSVC", create_linearsvc_search()),
        ("Random Forest", create_random_forest_search()),
    ]

    if args.smoke_test:
        experiments = [
            (model_name, configure_smoke_search(search))
            for model_name, search in experiments
        ]

    # -----------------------------------------------------------------
    # 4. Train and evaluate models
    # -----------------------------------------------------------------

    all_results = []

    for model_name, search in experiments:
        results = train_and_evaluate(
            model_name=model_name,
            search=search,
            train_texts=train_texts_clean,
            y_train=y_train,
            test_texts=test_texts_clean,
            y_test=y_test,
            figure_dir=figure_dir
        )

        all_results.append(results)

        if args.smoke_test:
            register_model_validation(
                validator,
                model_name,
                results
            )

    # -----------------------------------------------------------------
    # 5. Build final comparison
    # -----------------------------------------------------------------

    comparison_df = build_comparison_table(all_results)
    print_comparison_table(comparison_df)

    if args.smoke_test:
        validator.add(
            "All four classical models included in comparison",
            len(comparison_df) == 4
        )

    # -----------------------------------------------------------------
    # 6. Calculate total runtime
    # -----------------------------------------------------------------

    total_runtime = time.perf_counter() - total_start_time
    total_runtime_minutes = total_runtime / 60

    print("\n" + "=" * 80)
    print(f"Total experiment runtime: {total_runtime_minutes:.2f} minutes")
    print("=" * 80)

    # -----------------------------------------------------------------
    # 7. Save experiment artifacts
    # -----------------------------------------------------------------

    csv_path, json_path, report_path = save_experiment_artifacts(
        all_results,
        comparison_df,
        total_runtime_minutes,
        metrics_dir,
        report_dir,
        args.smoke_test,
        experiment_metadata
    )

    # -----------------------------------------------------------------
    # 8. Smoke-test artifact validation + summary
    # -----------------------------------------------------------------

    if args.smoke_test:
        validator.add_artifact(
            "Comparison CSV generated",
            csv_path
        )

        validator.add_artifact(
            "Comparison JSON generated",
            json_path
        )

        validator.add_artifact(
            "Experiment report generated",
            report_path
        )

        smoke_passed = validator.print_summary()

        if not smoke_passed:
            sys.exit(1)


if __name__ == "__main__":
    main()