"""
This script calculates various evaluation metrics for steganalysis models based on their predictions and ground truth labels.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
from typing import List
import argparse
import ast

def metrics_report(y_true: list[bool], y_pred: list[bool]) -> str:
    """Generate a metrics report given true and predicted labels."""

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    report = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"Confusion Matrix:\n"
        f"  TP: {tp}, FP: {fp}\n"
        f"  FN: {fn}, TN: {tn}\n"
    )
    return report

def parse_bool(line: str) -> bool:
    """Convert a single line to a boolean.

    Rules:
    - Empty string or "None" (case-insensitive) -> False
    - Python literals `True`/`False` and integers `1`/`0` are supported
    - Common textual representations (yes/no, y/n, t/f) are supported
    - Any unparsable value raises ValueError to avoid silent mistakes
    """
    if line is None:
        return False
    s = line.strip()
    if s == "" or s.lower() == "none":
        return False

    # Try Python literal first (True/False/1/0)
    try:
        val = ast.literal_eval(s)
        if isinstance(val, bool):
            return val
        if isinstance(val, int):
            return bool(val)
    except Exception:
        # fall through to textual checks
        pass

    low = s.lower()
    if low in ("true", "t", "1", "yes", "y"):
        return True
    if low in ("false", "f", "0", "no", "n"):
        return False

    raise ValueError(f"Cannot parse boolean value from line: {line!r}")

def evaluate_predictions(folder_path: Path, negative_group: str = "cover", positive_group: str = "stego") -> str:
    """Evaluate predictions in the given folder and print metrics report.

    This function reads predictions from `<negative_group>.txt` and/or
    `<positive_group>.txt` in `folder_path`. At least one file must exist.
    Each line is converted to a bool (with empty/`None` -> False).
    It returns the generated metrics report string.
    """

    folder_path = Path(folder_path)
    cover_file = folder_path / f"{negative_group}.txt"
    stego_file = folder_path / f"{positive_group}.txt"

    if not cover_file.exists() and not stego_file.exists():
        raise FileNotFoundError(
            f"Missing required files: expected at least one of {cover_file} or {stego_file}"
        )

    cover_vals: List[bool] = []
    stego_vals: List[bool] = []

    if cover_file.exists():
        with cover_file.open("r", encoding="utf-8") as f:
            cover_vals = [parse_bool(line) for line in f]

    if stego_file.exists():
        with stego_file.open("r", encoding="utf-8") as f:
            stego_vals = [parse_bool(line) for line in f]

    # Build y_pred and y_true with the requested ordering
    y_pred = cover_vals + stego_vals
    y_true = [False] * len(cover_vals) + [True] * len(stego_vals)

    if not y_true:
        raise ValueError("No predictions found: input file(s) are empty")

    report = metrics_report(y_true, y_pred)
    print(report)
    return report

def resolve_subset(run_name: str, subset: str | None) -> str:
    """Resolve the subset directory for a run.

    - If the run has exactly one subfolder, always use it.
    - If the run has multiple subfolders, `subset` must be provided and valid.
    """
    run_dir = Path("predictions") / run_name
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    subfolders = sorted(p.name for p in run_dir.iterdir() if p.is_dir())

    if len(subfolders) == 1:
        return subfolders[0]

    if len(subfolders) == 0:
        raise ValueError(f"No subset folders found under {run_dir}")

    if subset is None:
        raise ValueError(
            f"Run '{run_name}' has multiple subsets {subfolders}. "
            "Please pass --subset <name>."
        )

    if subset not in subfolders:
        raise ValueError(
            f"Subset '{subset}' not found for run '{run_name}'. Available: {subfolders}"
        )

    return subset


def main(run_name: str, subset: str | None = None, negative_group: str = "cover", positive_group: str = "stego"):
    """Main function to evaluate a specific run and save the metrics report."""
    resolved_subset = resolve_subset(run_name, subset)
    folder = Path(f"predictions/{run_name}/{resolved_subset}")
    report = evaluate_predictions(folder, negative_group=negative_group, positive_group=positive_group)

    # Ensure the top-level metrics directory exists and write the report there.
    # We keep a per-run subdirectory to avoid filename collisions: metrics/<run_name>/<subset>.txt
    out_dir = Path("metrics") / f"{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{resolved_subset}.txt"
    out_file.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for selecting run/subset and label file names."""
    parser = argparse.ArgumentParser(description="Evaluate steganalysis predictions and write a metrics report.")
    parser.add_argument("run_name", help="Run folder name under predictions/ (for example: run_37)")
    parser.add_argument(
        "--subset",
        default=None,
        help="Subset folder under the run. Required only when a run has multiple subset folders.",
    )
    parser.add_argument("--negative-group", default="cover", help="Negative label filename stem (default: cover)")
    parser.add_argument("--positive-group", default="stego", help="Positive label filename stem (default: stego)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        run_name=args.run_name,
        subset=args.subset,
        negative_group=args.negative_group,
        positive_group=args.positive_group,
    )
