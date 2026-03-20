"""
This script evaluates the results provided by a steganalysis model on a set of sample texts.
It creates new files in the `predictions/` directory where each line is converted to a boolean value (True for steganographic, False for non-steganographic).
"""

import argparse
from pathlib import Path


def _creation_or_modified_time(path: Path) -> float:
    """Return creation time when available, otherwise last-modified time."""
    stat = path.stat()
    return getattr(stat, "st_birthtime", stat.st_mtime)


def find_most_recent_run(outputs_dir: Path) -> Path | None:
    """Find the most recently created top-level run_* directory in outputs/."""
    run_dirs = [p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not run_dirs:
        return None

    # Sort by creation/modified time first, then by name for deterministic tie-breaking.
    return max(run_dirs, key=lambda p: (_creation_or_modified_time(p), p.name))

def coerce_txt_to_bool(label: str) -> bool | None:
    """Convert label string to boolean value."""
    label = label.lower()
    if label.startswith("yes") or label.startswith("steganographic"):
        return True
    elif label.startswith("no") or label.startswith("non-steganographic"):
        return False
    else:
        return None

def convert_txt_file(file_path: Path, verbose: bool = False) :
    """Make a new version of a txt file where each line is converted to a boolean value.

    The output will be written under the `predictions/` directory while preserving
    the folder structure that appears under `outputs/` in the original path. For
    example: `outputs/run_5/ac/stego.txt` -> `predictions/run_5/ac/stego.txt`.

    If the input file is not under a `outputs/` directory, the function will try to
    preserve the file's repository-relative path. If that isn't possible it will
    fall back to writing `predictions/<filename>`.


    """
    # Read and convert lines
    text = file_path.read_text(encoding="utf-8")
    lines = text.strip().splitlines() if text.strip() else []
    bool_lines = []
    unexpected_lines = []
    for line in lines:
        b = coerce_txt_to_bool(line)
        if b is None:
            unexpected_lines.append(line)
        bool_lines.append(str(b))

    print(f"Processing {file_path} -> unexpected lines: {len(unexpected_lines)}")
    if verbose:
        print("Unexpected lines:")
        for ul in set(unexpected_lines):
            print(f"  {ul}")
        print('\n- - - - - -\n')

    # Determine target path under predictions/
    repo_root = Path.cwd()
    target = None
    parts = file_path.parts
    if "outputs" in parts:
        # preserve structure under outputs/
        idx = parts.index("outputs")
        rel = Path(*parts[idx+1:])  # everything after 'outputs'
        target = Path("predictions") / rel
    else:
        # try to preserve repo-relative path
        try:
            rel_full = file_path.relative_to(repo_root)
            target = Path("predictions") / rel_full
        except Exception:
            # fallback to filename only
            target = Path("predictions") / file_path.name

    # Ensure parent directory exists
    target.parent.mkdir(parents=True, exist_ok=True)

    # Write boolean lines to the target path (use same filename as source)
    target.write_text("\n".join(bool_lines), encoding="utf-8")

def convert_folder(folder_path: Path, verbose: bool = False) -> None:
    """Convert all txt files in a folder to boolean predictions.

    Recursively searches up to 3 levels deep (folder itself counts as level 1).
    """
    max_depth = 3
    if not folder_path.exists() or not folder_path.is_dir():
        return

    for path in folder_path.rglob("*.txt"):
        try:
            rel = path.relative_to(folder_path)
        except Exception:
            continue
        # number of path components (files directly under folder -> 1)
        if len(rel.parts) <= max_depth:
            if path.is_file():
                convert_txt_file(path, verbose=verbose)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert model output txt labels in an outputs/ subfolder into boolean prediction files."
    )
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        help=(
            "Optional subfolder under outputs/ whose .txt files should be converted "
            "(e.g. run_37/ac). If omitted, uses the most recently created run_* folder "
            "under outputs/."
        ),
    )
    args = parser.parse_args()

    outputs_dir = Path("outputs")
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        parser.error(f"outputs/ directory does not exist or is not a directory: {outputs_dir}")

    # Treat CLI input as relative to outputs/ when provided.
    if args.folder is not None:
        requested = args.folder
        if requested.parts and requested.parts[0] == "outputs":
            requested = Path(*requested.parts[1:])
        folder_path = outputs_dir / requested
    else:
        latest_run = find_most_recent_run(outputs_dir)
        if latest_run is None:
            parser.error("No run_* directories found under outputs/.")
        folder_path = latest_run
        print(f"No folder specified; using latest outputs run: {folder_path}")

    if not folder_path.exists() or not folder_path.is_dir():
        parser.error(
            f"Folder does not exist or is not a directory under outputs/: {folder_path}"
        )

    convert_folder(folder_path, verbose=True)

if __name__ == "__main__":
    main()