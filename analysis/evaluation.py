"""
This script evaluates the results provided by a steganalysis model on a set of sample texts.
"""

from pathlib import Path

def coerce_txt_to_bool(label: str) -> bool:
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
    the folder structure that appears under `logs/` in the original path. For
    example: `logs/run_5/ac/stego.txt` -> `predictions/run_5/ac/stego.txt`.

    If the input file is not under a `logs/` directory, the function will try to
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
        for ul in unexpected_lines:
            print(f"  {ul}")
        print('\n- - - - - -\n')

    # Determine target path under predictions/
    repo_root = Path.cwd()
    target = None
    parts = file_path.parts
    if "logs" in parts:
        # preserve structure under logs/
        idx = parts.index("logs")
        rel = Path(*parts[idx+1:])  # everything after 'logs'
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
    convert_folder(Path("logs/run_28"), verbose=True)

if __name__ == "__main__":
    main()