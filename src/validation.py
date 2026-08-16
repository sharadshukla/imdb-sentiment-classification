"""
Validation utilities for IMDb sentiment-classification experiments.

This module provides reusable smoke-test support for both
classical and neural experiment runners.

Responsibilities:
- create balanced smoke-test subsets
- register validation checks
- verify generated artifacts
- print colored PASS/FAIL summaries
- provide a final boolean validation result
"""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# ANSI terminal colors
# ---------------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def select_balanced_subset(texts, labels, total_size):
    """
    Select a balanced positive/negative subset.

    Intended primarily for smoke tests so that a small sample still
    contains equal numbers of both sentiment classes.

    Parameters
    ----------
    texts : sequence
        Review texts.

    labels : array-like
        Binary labels where 0 = negative and 1 = positive.

    total_size : int
        Total number of samples requested.

    Returns
    -------
    subset_texts : list
        Selected review texts.

    subset_labels : np.ndarray
        Selected binary labels.
    """

    labels = np.asarray(labels)

    if total_size <= 0:
        raise ValueError("total_size must be greater than zero.")

    if total_size % 2 != 0:
        raise ValueError(
            "total_size must be even so the smoke-test subset "
            "can remain balanced."
        )

    per_class = total_size // 2

    negative_idx = np.where(labels == 0)[0]
    positive_idx = np.where(labels == 1)[0]

    if len(negative_idx) < per_class or len(positive_idx) < per_class:
        raise ValueError(
            "Not enough samples are available to create the requested "
            "balanced subset."
        )

    selected_idx = np.concatenate([
        negative_idx[:per_class],
        positive_idx[:per_class]
    ])

    subset_texts = [texts[idx] for idx in selected_idx]
    subset_labels = labels[selected_idx]

    return subset_texts, subset_labels


def is_balanced(labels):
    """
    Check whether binary labels contain equal numbers of 0 and 1.
    """

    labels = np.asarray(labels)

    return (
        np.sum(labels == 0)
        == np.sum(labels == 1)
    )


def is_valid_metric(value):
    """
    Check whether a metric value lies within the valid [0, 1] range.
    """

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return 0.0 <= numeric_value <= 1.0


def artifact_exists(path):
    """
    Check whether an expected artifact exists and is non-empty.
    """

    if path is None:
        return False

    path = Path(path)

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > 0
    )


class ValidationTracker:
    """
    Collect and report validation checks for smoke tests.

    Example
    -------
    tracker = ValidationTracker("CLASSICAL PIPELINE")

    tracker.add(
        "IMDb dataset loaded",
        len(train_texts) > 0
    )

    passed = tracker.print_summary()
    """

    def __init__(self, title):
        self.title = title
        self.checks = []

    def add(self, name, status):
        """
        Add one validation check.
        """

        self.checks.append({
            "name": name,
            "status": bool(status)
        })

    def add_artifact(self, name, path):
        """
        Add a file-artifact validation check.
        """

        self.add(
            name,
            artifact_exists(path)
        )

    def passed_count(self):
        """
        Return number of successful validation checks.
        """

        return sum(
            check["status"]
            for check in self.checks
        )

    def total_count(self):
        """
        Return total number of registered checks.
        """

        return len(self.checks)

    def all_passed(self):
        """
        Return True only if every registered check passed.
        """

        return (
            self.total_count() > 0
            and self.passed_count() == self.total_count()
        )

    def print_summary(self):
        """
        Print a colored PASS/FAIL validation summary.

        Returns
        -------
        bool
            True if all checks passed, otherwise False.
        """

        print("\n" + "=" * 80)
        print(f"{CYAN}{self.title} SMOKE TEST SUMMARY{RESET}")
        print("=" * 80)

        for check in self.checks:
            if check["status"]:
                print(
                    f"{GREEN}[PASS] ✓{RESET} "
                    f"{check['name']}"
                )
            else:
                print(
                    f"{RED}[FAIL] ✗{RESET} "
                    f"{check['name']}"
                )

        passed = self.passed_count()
        total = self.total_count()

        print("\n" + "-" * 80)
        print(f"Result: {passed}/{total} checks passed")

        if self.all_passed():
            print(
                f"\n{GREEN}SMOKE TEST: PASSED{RESET}"
            )
        else:
            print(
                f"\n{RED}SMOKE TEST: FAILED{RESET}"
            )

        print("=" * 80)

        return self.all_passed()


def print_smoke_mode_banner(
    experiment_type,
    train_size,
    test_size,
    extra_lines=None
):
    """
    Print a consistent smoke-test mode banner.

    Parameters
    ----------
    experiment_type : str
        Example: "Classical" or "Neural".

    train_size : int
        Number of smoke-test training samples.

    test_size : int
        Number of smoke-test test samples.

    extra_lines : list[str], optional
        Additional experiment-specific information.
    """

    print(
        f"\n{YELLOW}SMOKE TEST MODE ENABLED "
        f"({experiment_type}){RESET}"
    )

    print(f"Training reviews : {train_size:,}")
    print(f"Test reviews     : {test_size:,}")

    if extra_lines:
        for line in extra_lines:
            print(line)