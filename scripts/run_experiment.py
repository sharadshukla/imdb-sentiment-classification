"""
Master experiment runner for IMDb sentiment classification.

Usage examples:

    Full classical experiment:
        python scripts/run_experiment.py --mode classical

    Classical smoke test:
        python scripts/run_experiment.py --mode classical --smoke-test

    Full neural experiment:
        python scripts/run_experiment.py --mode neural

    Neural smoke test:
        python scripts/run_experiment.py --mode neural --smoke-test
"""

from pathlib import Path
import argparse
import subprocess
import sys


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def parse_args():
    """
    Parse command-line options.
    """

    parser = argparse.ArgumentParser(
        description="Run IMDb sentiment-classification experiments."
    )

    parser.add_argument(
        "--mode",
        choices=["classical", "neural"],
        required=True,
        help="Choose which experiment pipeline to run."
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the selected pipeline in fast smoke-test mode."
    )

    return parser.parse_args()


def build_command(mode, smoke_test):
    """
    Build the command for the selected experiment runner.
    """

    if mode == "classical":
        script_path = SCRIPTS_DIR / "run_classical.py"
    else:
        script_path = SCRIPTS_DIR / "run_neural.py"

    command = [
        sys.executable,
        str(script_path)
    ]

    if smoke_test:
        command.append("--smoke-test")

    return command


def main():
    """
    Dispatch execution to the selected experiment runner.
    """

    args = parse_args()

    command = build_command(
        mode=args.mode,
        smoke_test=args.smoke_test
    )

    run_type = "Smoke Test" if args.smoke_test else "Full Experiment"

    print("=" * 80)
    print("IMDb Sentiment Classification - Master Runner")
    print("=" * 80)
    print(f"Mode     : {args.mode}")
    print(f"Run Type : {run_type}")
    print("=" * 80)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT
    )

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("Experiment completed successfully.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(
            f"Experiment failed with exit code "
            f"{result.returncode}."
        )
        print("=" * 80)

        sys.exit(result.returncode)


if __name__ == "__main__":
    main()