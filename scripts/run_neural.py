"""
Run the neural IMDb sentiment-classification experiments.

Models:
- Vanilla RNN
- LSTM
- BiLSTM + FastText
- Self-Attention

Usage:
    Full experiment:
        python scripts/run_neural.py

    Smoke test:
        python scripts/run_neural.py --smoke-test
"""

from pathlib import Path
import argparse
import gc
import re
import sys
import time


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from src.data import MAX_LEN, create_dataloaders, load_imdb_data
from src.evaluation import evaluate_neural
from src.neural_models import (
    RNNSentiment,
    LSTMSentiment,
    BiLSTMSentiment,
    SelfAttnSentiment,
)
from src.preprocessing import build_vocabulary
from src.reporting import (
    save_comparison_csv,
    save_neural_comparison_json,
    save_neural_experiment_report,
)
from src.training import make_scheduler, train_model
from src.utils import (
    build_embedding_matrix,
    count_params,
    load_fasttext_embeddings,
    plot_losses,
)
from src.validation import (
    ValidationTracker,
    has_valid_sequence_lengths,
    has_zero_padding_embedding,
    is_balanced,
    is_valid_metric,
    print_smoke_mode_banner,
    select_balanced_subset,
)


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

SEED = 42

FULL_EPOCHS = 10
SMOKE_EPOCHS = 2

SMOKE_TRAIN_SIZE = 1000
SMOKE_TEST_SIZE = 500


def parse_args():
    """
    Parse command-line options.
    """

    parser = argparse.ArgumentParser(
        description="Run neural IMDb sentiment-classification experiments."
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a fast integration test using a small balanced subset, "
            "two epochs, and temporary 300-dimensional embeddings instead "
            "of downloading FastText."
        )
    )

    return parser.parse_args()


def get_output_directories(smoke_test):
    """
    Return output directories for either a full run or smoke test.
    """

    run_type = "smoke" if smoke_test else "full"
    base_dir = PROJECT_ROOT / "results" / run_type / "neural"

    figure_dir = base_dir / "figures"
    metrics_dir = base_dir / "metrics"
    report_dir = base_dir / "reports"

    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    return figure_dir, metrics_dir, report_dir


def set_seed(seed=SEED):
    """
    Set random seeds for improved reproducibility.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    """
    Prefer CUDA GPU when available, otherwise use CPU.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nSelected device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA GPU not available - using CPU.")

    return device


def create_smoke_embeddings(vocab_size, embedding_dim=300):
    """
    Create temporary embeddings for BiLSTM smoke testing.

    These are not pretrained FastText vectors.
    They validate the 300-dimensional pretrained-embedding code path.
    """

    weights = torch.empty(vocab_size, embedding_dim)
    nn.init.uniform_(weights, -0.1, 0.1)

    # PAD token remains zero
    weights[0] = 0.0

    return weights


def make_filename(model_name):
    """
    Convert a model name into a filesystem-friendly filename.
    """

    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def train_and_evaluate_neural(
    model_name,
    model,
    optimizer,
    train_loader,
    test_loader,
    criterion,
    device,
    n_epochs,
    figure_dir,
    scheduler=None
):
    """
    Train and evaluate one neural sentiment model.
    """

    print("\n" + "=" * 80)
    print(f"Training: {model_name}")
    print("=" * 80)

    parameter_count = count_params(model)

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    start_time = time.perf_counter()

    losses = train_model(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        n_epochs=n_epochs,
        scheduler=scheduler
    )

    runtime = time.perf_counter() - start_time
    runtime_minutes = runtime / 60

    # -----------------------------------------------------------------
    # Training-loss figure
    # -----------------------------------------------------------------

    filename = make_filename(model_name)
    loss_figure_path = figure_dir / f"{filename}_training_loss.png"

    plot_losses(
        losses,
        f"{model_name} - Training Loss",
        output_path=loss_figure_path
    )

    # -----------------------------------------------------------------
    # Evaluation + confusion matrix
    # -----------------------------------------------------------------

    confusion_matrix_path = figure_dir / f"{filename}_confusion_matrix.png"

    results, preds, labels = evaluate_neural(
        model=model,
        dataloader=test_loader,
        device=device,
        name=model_name,
        figure_path=confusion_matrix_path
    )

    # -----------------------------------------------------------------
    # Preserve experiment metadata
    # -----------------------------------------------------------------

    results["Parameters"] = parameter_count
    results["Final Training Loss"] = losses[-1]
    results["Runtime (min)"] = runtime_minutes
    results["Loss Figure Path"] = loss_figure_path
    results["Confusion Matrix Path"] = confusion_matrix_path

    print(f"\nTraining runtime: {runtime_minutes:.2f} minutes")

    # -----------------------------------------------------------------
    # Release accelerator memory before next model
    # -----------------------------------------------------------------

    model.to("cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return results, losses, preds, labels


def build_comparison_table(all_results):
    """
    Build an F1-sorted neural-model comparison table.
    """

    comparison_df = pd.DataFrame([
        {
            "Model": result["Model"],
            "Accuracy": result["Accuracy"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1 Score": result["F1 Score"],
            "Parameters": result["Parameters"],
            "Final Training Loss": result["Final Training Loss"],
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
    Print the final neural-model comparison table.
    """

    print("\n" + "=" * 80)
    print("Final Neural Model Comparison")
    print("=" * 80)

    print(
        comparison_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format,
                "Parameters": "{:,.0f}".format,
                "Final Training Loss": "{:.4f}".format,
                "Runtime (min)": "{:.2f}".format
            }
        )
    )


def save_loss_comparison(loss_histories, figure_dir):
    """
    Save all neural training-loss curves on one figure.
    """

    plt.figure(figsize=(9, 6))

    line_styles = {
        "RNN": "-",
        "LSTM": "--",
        "BiLSTM + FastText": "-.",
        "Self-Attention": ":"
    }

    for model_name, losses in loss_histories.items():
        epochs = range(1, len(losses) + 1)

        plt.plot(
            epochs,
            losses,
            label=model_name,
            linestyle=line_styles[model_name],
            linewidth=2
        )

    max_epochs = max(len(losses) for losses in loss_histories.values())

    plt.title("Training Loss Comparison of Neural Models")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.xticks(range(1, max_epochs + 1))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    output_path = figure_dir / "neural_loss_comparison.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Combined neural loss figure saved to: {output_path}")

    return output_path


def save_loss_metrics(loss_histories, metrics_dir):
    """
    Save epoch-by-epoch training losses as CSV.
    """

    epoch_count = len(next(iter(loss_histories.values())))

    loss_df = pd.DataFrame({
        "Epoch": range(1, epoch_count + 1),
        **loss_histories
    })

    output_path = metrics_dir / "neural_training_losses.csv"

    loss_df.to_csv(output_path, index=False, float_format="%.4f")

    return output_path


def save_experiment_artifacts(
    all_results,
    comparison_df,
    loss_histories,
    total_runtime_minutes,
    fasttext_preparation_minutes,
    figure_dir,
    metrics_dir,
    report_dir,
    smoke_test
):
    """
    Persist neural experiment artifacts.
    """

    prefix = "smoke_" if smoke_test else ""

    csv_path = metrics_dir / f"{prefix}neural_model_comparison.csv"
    json_path = metrics_dir / f"{prefix}neural_model_comparison.json"
    report_path = report_dir / f"{prefix}neural_experiment_report.txt"

    save_comparison_csv(comparison_df, csv_path)

    save_neural_comparison_json(
        all_results,
        json_path,
        total_runtime_minutes,
        fasttext_preparation_minutes
    )

    save_neural_experiment_report(
        all_results,
        comparison_df,
        report_path,
        total_runtime_minutes,
        fasttext_preparation_minutes
    )

    loss_metrics_path = save_loss_metrics(loss_histories, metrics_dir)
    loss_comparison_path = save_loss_comparison(loss_histories, figure_dir)

    print("\nArtifacts saved successfully:")
    print(f"Figures         : {figure_dir}")
    print(f"Comparison CSV  : {csv_path}")
    print(f"Comparison JSON : {json_path}")
    print(f"Loss CSV        : {loss_metrics_path}")
    print(f"Loss Figure     : {loss_comparison_path}")
    print(f"Report          : {report_path}")

    return (
        csv_path,
        json_path,
        report_path,
        loss_metrics_path,
        loss_comparison_path
    )


def register_model_validation(validator, model_name, results, losses, n_epochs):
    """
    Register smoke-test checks for one neural model.
    """

    validator.add(
        f"{model_name} training completed",
        len(losses) == n_epochs
    )

    validator.add(
        f"{model_name} F1 is valid",
        is_valid_metric(results["F1 Score"])
    )

    validator.add(
        f"{model_name} final training loss is valid",
        np.isfinite(results["Final Training Loss"])
    )

    validator.add_artifact(
        f"{model_name} loss figure generated",
        results["Loss Figure Path"]
    )

    validator.add_artifact(
        f"{model_name} confusion matrix generated",
        results["Confusion Matrix Path"]
    )


def main():
    """
    Run the complete neural experiment or its smoke test.
    """

    args = parse_args()

    figure_dir, metrics_dir, report_dir = get_output_directories(
        args.smoke_test
    )

    n_epochs = SMOKE_EPOCHS if args.smoke_test else FULL_EPOCHS

    validator = (
        ValidationTracker("NEURAL PIPELINE")
        if args.smoke_test
        else None
    )

    total_start_time = time.perf_counter()

    set_seed()
    device = select_device()

    print("\n" + "=" * 80)
    print("IMDb Sentiment Classification - Neural Models")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Smoke-test banner
    # -----------------------------------------------------------------

    if args.smoke_test:
        print_smoke_mode_banner(
            experiment_type="Neural",
            train_size=SMOKE_TRAIN_SIZE,
            test_size=SMOKE_TEST_SIZE,
            extra_lines=[
                f"Epochs           : {n_epochs}",
                "FastText          : temporary 300-D embeddings"
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
    # 2. Build vocabulary
    # -----------------------------------------------------------------

    print("\nBuilding neural vocabulary...")

    word2idx, idx2word = build_vocabulary(train_texts)
    vocab_size = len(word2idx)

    print(f"Vocabulary size: {vocab_size:,}")

    if args.smoke_test:
        validator.add(
            "Vocabulary created",
            vocab_size > 2
            and word2idx.get("<PAD>") == 0
            and word2idx.get("<UNK>") == 1
        )

    # -----------------------------------------------------------------
    # 3. Create PyTorch datasets and dataloaders
    # -----------------------------------------------------------------

    print("\nCreating PyTorch datasets and dataloaders...")

    train_ds, test_ds, train_loader, test_loader = create_dataloaders(
        train_texts=train_texts,
        y_train=y_train,
        test_texts=test_texts,
        y_test=y_test,
        word2idx=word2idx
    )

    print(f"Training dataset: {len(train_ds):,} reviews")
    print(f"Test dataset    : {len(test_ds):,} reviews")

    if args.smoke_test:
        validator.add(
            "PyTorch DataLoaders created",
            len(train_ds) == SMOKE_TRAIN_SIZE
            and len(test_ds) == SMOKE_TEST_SIZE
            and len(train_loader) > 0
            and len(test_loader) > 0
        )

        validator.add(
            "Training sequence lengths are valid",
            has_valid_sequence_lengths(
                train_ds.lengths,
                MAX_LEN
            )
        )

        validator.add(
            "Test sequence lengths are valid",
            has_valid_sequence_lengths(
                test_ds.lengths,
                MAX_LEN
            )
        )

        validator.add(
            "Variable-length sequences present",
            any(length < MAX_LEN for length in train_ds.lengths)
            and any(length < MAX_LEN for length in test_ds.lengths)
        )

    criterion = nn.BCEWithLogitsLoss()

    all_results = []
    loss_histories = {}

    # -----------------------------------------------------------------
    # 4. RNN
    # -----------------------------------------------------------------

    rnn_model = RNNSentiment(vocab_size)
    optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=1e-3)

    rnn_results, rnn_losses, _, _ = train_and_evaluate_neural(
        model_name="RNN",
        model=rnn_model,
        optimizer=optimizer_rnn,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        n_epochs=n_epochs,
        figure_dir=figure_dir
    )

    all_results.append(rnn_results)
    loss_histories["RNN"] = rnn_losses

    if args.smoke_test:
        register_model_validation(
            validator,
            "RNN",
            rnn_results,
            rnn_losses,
            n_epochs
        )

    # -----------------------------------------------------------------
    # 5. LSTM
    # -----------------------------------------------------------------

    lstm_model = LSTMSentiment(vocab_size)
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=1e-3)

    lstm_results, lstm_losses, _, _ = train_and_evaluate_neural(
        model_name="LSTM",
        model=lstm_model,
        optimizer=optimizer_lstm,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        n_epochs=n_epochs,
        figure_dir=figure_dir
    )

    all_results.append(lstm_results)
    loss_histories["LSTM"] = lstm_losses

    if args.smoke_test:
        register_model_validation(
            validator,
            "LSTM",
            lstm_results,
            lstm_losses,
            n_epochs
        )

    # -----------------------------------------------------------------
    # 6. FastText / temporary smoke embeddings
    # -----------------------------------------------------------------

    if args.smoke_test:
        print("\n" + "=" * 80)
        print("Preparing Temporary Smoke-Test Embeddings")
        print("=" * 80)

        fasttext_start_time = time.perf_counter()
        fasttext_weights = create_smoke_embeddings(vocab_size)

        if not has_zero_padding_embedding(fasttext_weights):
            raise RuntimeError(
                "Smoke embedding validation failed: "
                "PAD embedding row must remain zero."
            )

        fasttext_runtime = time.perf_counter() - fasttext_start_time
        fasttext_preparation_minutes = fasttext_runtime / 60

        validator.add(
            "BiLSTM embedding matrix created",
            fasttext_weights.shape == (vocab_size, 300)
        )

        validator.add(
            "BiLSTM PAD embedding is zero",
            has_zero_padding_embedding(fasttext_weights)
        )

        print(
            "Temporary 300-dimensional embeddings created. "
            "Real FastText is used only in the full experiment."
        )

    else:
        print("\n" + "=" * 80)
        print("Preparing FastText Embeddings")
        print("=" * 80)

        fasttext_start_time = time.perf_counter()

        ft_model = load_fasttext_embeddings()
        fasttext_weights = build_embedding_matrix(word2idx, ft_model)

        if not has_zero_padding_embedding(fasttext_weights):
            raise RuntimeError(
                "FastText embedding matrix validation failed: "
                "PAD embedding row must remain zero."
            )

        fasttext_runtime = time.perf_counter() - fasttext_start_time
        fasttext_preparation_minutes = fasttext_runtime / 60

        print(
            f"\nFastText preparation runtime: "
            f"{fasttext_preparation_minutes:.2f} minutes"
        )

        del ft_model
        gc.collect()

    # -----------------------------------------------------------------
    # 7. BiLSTM + FastText
    # -----------------------------------------------------------------

    bilstm_model = BiLSTMSentiment(
        vocab_size,
        pretrained_embeddings=fasttext_weights
    )

    optimizer_bilstm = optim.Adam(
        bilstm_model.parameters(),
        lr=5e-4
    )

    bilstm_results, bilstm_losses, _, _ = train_and_evaluate_neural(
        model_name="BiLSTM + FastText",
        model=bilstm_model,
        optimizer=optimizer_bilstm,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        n_epochs=n_epochs,
        figure_dir=figure_dir
    )

    all_results.append(bilstm_results)
    loss_histories["BiLSTM + FastText"] = bilstm_losses

    if args.smoke_test:
        register_model_validation(
            validator,
            "BiLSTM",
            bilstm_results,
            bilstm_losses,
            n_epochs
        )

    del fasttext_weights
    gc.collect()

    # -----------------------------------------------------------------
    # 8. Self-Attention
    # -----------------------------------------------------------------

    self_attn_model = SelfAttnSentiment(vocab_size)

    optimizer_attn = optim.Adam(
        self_attn_model.parameters(),
        lr=5e-4
    )

    scheduler = make_scheduler(
        optimizer_attn,
        n_epochs,
        warmup_epochs=1 if args.smoke_test else 2
    )

    attn_results, attn_losses, _, _ = train_and_evaluate_neural(
        model_name="Self-Attention",
        model=self_attn_model,
        optimizer=optimizer_attn,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        n_epochs=n_epochs,
        figure_dir=figure_dir,
        scheduler=scheduler
    )

    all_results.append(attn_results)
    loss_histories["Self-Attention"] = attn_losses

    if args.smoke_test:
        register_model_validation(
            validator,
            "Self-Attention",
            attn_results,
            attn_losses,
            n_epochs
        )

    # -----------------------------------------------------------------
    # 9. Final comparison
    # -----------------------------------------------------------------

    comparison_df = build_comparison_table(all_results)
    print_comparison_table(comparison_df)

    if args.smoke_test:
        validator.add(
            "All four neural models included in comparison",
            len(comparison_df) == 4
        )

    # -----------------------------------------------------------------
    # 10. Total runtime
    # -----------------------------------------------------------------

    total_runtime = time.perf_counter() - total_start_time
    total_runtime_minutes = total_runtime / 60

    print("\n" + "=" * 80)
    print(f"Total neural experiment runtime: {total_runtime_minutes:.2f} minutes")
    print("=" * 80)

    # -----------------------------------------------------------------
    # 11. Save artifacts
    # -----------------------------------------------------------------

    (
        csv_path,
        json_path,
        report_path,
        loss_metrics_path,
        loss_comparison_path
    ) = save_experiment_artifacts(
        all_results,
        comparison_df,
        loss_histories,
        total_runtime_minutes,
        fasttext_preparation_minutes,
        figure_dir,
        metrics_dir,
        report_dir,
        args.smoke_test
    )

    # -----------------------------------------------------------------
    # 12. Smoke-test artifact validation + summary
    # -----------------------------------------------------------------

    if args.smoke_test:
        validator.add_artifact("Comparison CSV generated", csv_path)
        validator.add_artifact("Comparison JSON generated", json_path)
        validator.add_artifact("Training-loss CSV generated", loss_metrics_path)
        validator.add_artifact("Combined loss figure generated", loss_comparison_path)
        validator.add_artifact("Experiment report generated", report_path)

        smoke_passed = validator.print_summary()

        if not smoke_passed:
            sys.exit(1)


if __name__ == "__main__":
    main()