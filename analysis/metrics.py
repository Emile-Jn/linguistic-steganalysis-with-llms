"""
This script calculates various evaluation metrics for steganalysis models based on their predictions and ground truth labels.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
from typing import List
import argparse

def metrics_report(y_true: list[bool], y_pred: list[bool]) -> str:
    """Generate a metrics report given true and predicted labels."""

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

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

def evaluate_predictions(folder_path: Path, negative_group: str = "cover", positive_group: str = "stego") -> str:
    """Evaluate predictions in the given folder and print metrics report.

    This function expects two files in `folder_path`: `cover.txt` and `stego.txt`.
    Each line in those files will be converted to a bool (with empty/`None` -> False).
    It returns the generated metrics report string.
    """

    folder_path = Path(folder_path)
    cover_file = folder_path / f"{negative_group}.txt"
    stego_file = folder_path / f"{positive_group}.txt"

    # Ensure required files exist
    if not cover_file.exists():
        raise FileNotFoundError(f"Missing required file: {cover_file}")
    if not stego_file.exists():
        raise FileNotFoundError(f"Missing required file: {stego_file}")

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
        import ast
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

    # Read and parse files
    with cover_file.open("r", encoding="utf-8") as f:
        cover_vals: List[bool] = [parse_bool(line) for line in f]

    with stego_file.open("r", encoding="utf-8") as f:
        stego_vals: List[bool] = [parse_bool(line) for line in f]

    # Build y_pred and y_true with the requested ordering
    y_pred = cover_vals + stego_vals
    y_true = [False] * len(cover_vals) + [True] * len(stego_vals)

    if len(y_pred) != len(y_true):
        raise ValueError("Predicted and true label lists are not the same length")

    report = metrics_report(y_true, y_pred)
    print(report)
    return report

def main(run_name: str, subset: str = "ac", negative_group: str = "cover", positive_group: str = "stego"):
    """Main function to evaluate a specific run and save the metrics report."""
    folder = Path(f"predictions/{run_name}/{subset}")
    report = evaluate_predictions(folder, negative_group=negative_group, positive_group=positive_group)

    # Ensure the top-level metrics directory exists and write the report there.
    # We keep a per-run subdirectory to avoid filename collisions: metrics/<run_name>/<subset>.txt
    out_dir = Path("metrics") / f"{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{subset}.txt"
    out_file.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for selecting run/subset and label file names."""
    parser = argparse.ArgumentParser(description="Evaluate steganalysis predictions and write a metrics report.")
    parser.add_argument("run_name", help="Run folder name under predictions/ (for example: run_37)")
    parser.add_argument("--subset", default="ac", help="Subset folder under the run (default: ac)")
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
