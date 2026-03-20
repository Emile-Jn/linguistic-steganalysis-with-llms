# -*- coding: utf-8 -*-
"""
Test LoRA fine-tuned Llama model for steganographic text detection

run this on the slurm cluster:
sbatch --partition=GPU-a100 run.sh test.py -nmax 5
sbatch --partition=GPU-a100 run.sh test.py -nmax 5 --data-path "qwen-generated"
"""

# Third-party modules
from time import time
import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import argparse
from typing import Optional
from datetime import datetime, timezone

# Custom modules
from utils import get_device, get_root_dir

from pathlib import Path

# Use a smaller batch to reduce GPU memory spikes during generation.
BATCH_SIZE = 8


def read_txt_lines(path: str):
    """Read a .txt file and return a list of non-empty stripped lines."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    # filter out empty lines
    return [l for l in lines if l]


def save_outputs_to_file(outputs, path: str) -> None:
    """Save a list of outputs to a text file, one output per line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for out in outputs:
            f.write(out.replace("\n", " ").strip() + "\n")


# Add a helper to atomically write the manifest so periodic saves are safe.
def write_manifest_atomic(run_dir: Path, summary: dict, total_lines_processed: int, start_inf: float, start_time: float, cli_args: Optional[dict] = None):
    """Write the run manifest to run_dir/manifest.json atomically.

    Args:
        run_dir: Path to the run directory
        summary: per-dataset summary dict
        total_lines_processed: integer total of lines processed so far
        start_inf: inference start time (seconds since epoch)
        start_time: overall run start time (seconds since epoch)
    """
    manifest = {
        "run_dir": str(run_dir),
        # Use ISO 8601 UTC timestamp without microseconds
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summary,
        "total_lines_processed": total_lines_processed,
        # total processing time in seconds (rounded to milliseconds)
        "total_inference_time_seconds": round(time() - start_inf, 3),
        "total_time_seconds": round(time() - start_time, 3)
    }
    # Include CLI arguments in the manifest for reproducibility (empty dict if None)
    manifest["cli_args"] = cli_args if cli_args is not None else {}
    tmp_path = run_dir / "manifest.json.tmp"
    final_path = run_dir / "manifest.json"
    # Write using UTF-8 and then rename to be atomic on most platforms
    tmp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        tmp_path.replace(final_path)
    except Exception:
        # fallback to simple write if replace() fails for any reason
        final_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def get_next_run_dir(logs_root: Path) -> Path:
    """Return a new run directory path under logs_root named run_1, run_2, ..."""
    logs_root.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in logs_root.iterdir() if d.is_dir() and d.name.startswith("run_")]
    nums = []
    for name in existing:
        try:
            n = int(name.split("run_")[1])
            nums.append(n)
        except Exception:
            continue
    next_n = max(nums) + 1 if nums else 1
    return logs_root / f"run_{next_n}"


def load_model_and_tokenizer(use_lora: bool = True):
    """
    New implementation of load_model_and_tokenizer with improved LoRA loading.
    """
    base_model_path = "linhvu/decapoda-research-llama-7b-hf"
    print("Using base model:", base_model_path)

    # adapter_path is the folder containing LoRA adapter and tokenizer files
    adapter_path = "adapter"
    if use_lora:
        tokenizer = LlamaTokenizer.from_pretrained(adapter_path)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model_path)

    model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=True,
        device_map="auto",
    )

    if use_lora:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer, get_device()

def run_inference(texts, model, tokenizer, device, batch_size=BATCH_SIZE):
    """New version of run_inference with batching, produces strange outputs.
    Without padding_side='left' it works only for batch_size=1."""
    if not texts:
        return [], []

    # Fix weird output with left-padding instead of right-padding
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    prompts = [
        f"{tokenizer.bos_token or ''}"
        f"### Text:\n{t.strip()}\n\n### Question:\nIs the above text steganographic?\n\n### Answer:\n"
        for t in texts
    ]

    answers, token_ids = [], []

    model.eval()

    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(
            prompts[i:i + batch_size],
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=10,
                min_new_tokens=2,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=False
            )

        for seq in gen.sequences:
            gen_ids = seq[enc.input_ids.shape[1]:]
            token_ids.append(gen_ids.tolist())
            answers.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        # Helps fragmentation in long jobs
        del enc, gen
        torch.cuda.empty_cache()

    return answers, token_ids


def _resolve_paths(cli_args: Optional[dict]) -> tuple[Path, Path]:
    """Return (data_root, outputs_root) derived from the project root and optional CLI args.

    Args:
        cli_args: parsed CLI arguments dict; may contain 'data_path' to override the default.

    Returns:
        A tuple of (data_root, outputs_root) as absolute Path objects.
    """
    try:
        root = Path(get_root_dir())
    except Exception:
        root = Path.cwd()

    if isinstance(cli_args, dict) and cli_args.get('data_path'):
        data_root = root / "data" / cli_args['data_path']
    else:
        data_root = root / "data" / "baseline"

    outputs_root = root / "outputs"
    return data_root, outputs_root


def _load_model(use_lora: bool, dry_run: bool):
    """Load and return (model, tokenizer, device), or (None, None, None) in dry-run mode.

    Args:
        use_lora: whether to apply the LoRA adapter on top of the base model.
        dry_run: when True, skip loading entirely and return None placeholders.

    Returns:
        Tuple of (model, tokenizer, device). All None when dry_run is True.

    Raises:
        SystemExit-equivalent: prints an error and returns None triplet on failure.
    """
    if dry_run:
        return None, None, None
    try:
        return load_model_and_tokenizer(use_lora=use_lora)
    except Exception as e:
        print("Failed to load model/tokenizer:", e)
        return None, None, None


def _process_txt_file(
    txt_file: Path,
    out_dir: Path,
    model,
    tokenizer,
    device,
    *,
    nmax: Optional[int],
    chunk_size: int,
    print_tokens: bool,
    dry_run: bool,
    on_chunk_done,
) -> int:
    """Run inference on a single .txt file and write results incrementally.

    Args:
        txt_file: input text file to process.
        out_dir: directory where the output file (and optional tokens JSONL) will be written.
        model, tokenizer, device: loaded model components (may be None in dry-run mode).
        nmax: maximum number of input lines to process; None means no limit.
        chunk_size: number of lines to process per batch before flushing to disk.
        print_tokens: if True, write a parallel .tokens.jsonl file with generated token IDs.
        dry_run: if True, produce fake outputs without calling the model.
        on_chunk_done: callback(lines_in_chunk: int) called after each chunk is written,
                       used by the caller to update counters and write periodic manifests.

    Returns:
        Number of lines processed (0 on failure or empty file).
    """
    print(f"Processing {txt_file}...")
    texts = read_txt_lines(str(txt_file))

    if nmax is not None and len(texts) > nmax:
        texts = texts[:nmax]
        print(f"  Truncated to first {nmax} lines for {txt_file}")

    if not texts:
        print(f"  Skipping empty or missing file {txt_file}")
        return 0

    out_path = out_dir / txt_file.name
    tokens_path = out_path.with_suffix('.tokens.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines_processed = 0
    wrote_any = False
    try:
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]

            if dry_run:
                outputs = [f"DRYRUN_OUTPUT_LINE_{j}" for j in range(len(chunk))]
                token_lists = [[] for _ in chunk]
            else:
                outputs, token_lists = run_inference(chunk, model, tokenizer, device, batch_size=BATCH_SIZE)

            mode = 'w' if not wrote_any else 'a'
            with out_path.open(mode, encoding='utf-8') as of:
                for out in outputs:
                    of.write(out.replace('\n', ' ').strip() + '\n')

            if print_tokens:
                tokens_path.parent.mkdir(parents=True, exist_ok=True)
                mode_t = 'w' if not wrote_any else 'a'
                with tokens_path.open(mode_t, encoding='utf-8') as tf:
                    for ids in token_lists:
                        tf.write(json.dumps(ids, ensure_ascii=False) + '\n')

            lines_processed += len(chunk)
            wrote_any = True
            on_chunk_done(len(chunk))

    except Exception as e:
        print(f"  Inference failed for {txt_file}: {e}")
        return 0

    return lines_processed


def process_dataset_dir(
    dataset_dir: Path,
    out_root: Path,
    model,
    tokenizer,
    device,
    *,
    nmax: Optional[int],
    chunk_size: int,
    file_name: Optional[str],
    cover_only: bool,
    stego_only: bool,
    print_tokens: bool,
    dry_run: bool,
    on_chunk_done,
) -> tuple[int, int]:
    """Process all .txt files in a single dataset directory.

    Args:
        dataset_dir: input directory containing .txt files to process.
        out_root: root output directory; results are written to out_root/dataset_dir.name/.
        model, tokenizer, device: loaded model components (may be None in dry-run mode).
        nmax: maximum number of input lines per file; None means no limit.
        chunk_size: lines per inference batch.
        file_name: if set, only process the exact file with this name in each dataset folder.
        cover_only: if True, only process a file named exactly 'cover.txt'.
        stego_only: if True, only process .txt files whose names start with 'stego'.
        print_tokens: if True, write a parallel .tokens.jsonl for each output file.
        dry_run: if True, produce fake outputs without calling the model.
        on_chunk_done: callback(lines_in_chunk: int) forwarded to _process_txt_file.

    Returns:
        A tuple of (processed_files, lines_processed) for this directory.
    """
    out_dataset_dir = out_root / dataset_dir.name
    out_dataset_dir.mkdir(parents=True, exist_ok=True)

    if file_name:
        candidate = dataset_dir / file_name
        txt_files = [candidate] if candidate.exists() else []
    elif cover_only:
        candidate = dataset_dir / 'cover.txt'
        txt_files = [candidate] if candidate.exists() else []
    elif stego_only:
        txt_files = sorted(dataset_dir.glob("stego*.txt"))
    else:
        txt_files = sorted(dataset_dir.glob("*.txt"))

    processed_files = 0
    lines_processed = 0
    for txt_file in txt_files:
        n = _process_txt_file(
            txt_file, out_dataset_dir, model, tokenizer, device,
            nmax=nmax, chunk_size=chunk_size, print_tokens=print_tokens,
            dry_run=dry_run, on_chunk_done=on_chunk_done,
        )
        if n > 0:
            lines_processed += n
            processed_files += 1

    return processed_files, lines_processed


def run_all_tests(nmax: Optional[int] = None, print_tokens: bool = False, use_lora: bool = True, manifest_threshold: int = 1000, dry_run: bool = False, cli_args: Optional[dict] = None):
    """Top-level pipeline:
    - Resolves data and logs paths
    - Creates a new run_X folder inside logs/
    - Runs inference on every .txt file in data_root and each of its subdirectories,
      writing outputs that mirror the input directory structure

    Args:
        nmax: maximum number of lines to take from each .txt file. None means all lines.
        print_tokens: whether to write generated token IDs to a sidecar .tokens.jsonl file.
        use_lora: whether to load LoRA adapter weights into the model.
        manifest_threshold: write a periodic manifest every this many output lines.
        dry_run: skip model loading; produce fake outputs (useful for testing I/O logic).
        cli_args: parsed CLI arguments dict, embedded in the manifest for reproducibility.
    """
    file_name = cli_args.get('file_name') if isinstance(cli_args, dict) else None
    cover_only = bool(cli_args.get('cover_only')) if isinstance(cli_args, dict) else False
    stego_only = bool(cli_args.get('stego_only')) if isinstance(cli_args, dict) else False
    chunk_size = int(manifest_threshold) if manifest_threshold and manifest_threshold > 0 else 1000

    data_root, outputs_root = _resolve_paths(cli_args)
    if not data_root.exists():
        print(f"Data root {data_root} does not exist. Nothing to process.")
        return

    run_dir = get_next_run_dir(outputs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating new run directory: {run_dir}")

    model, tokenizer, device = _load_model(use_lora=use_lora, dry_run=dry_run)
    if not dry_run and model is None:
        return

    summary: dict = {}
    total_lines_processed = 0
    start_time = time()
    start_inf = time()
    next_manifest_threshold = chunk_size

    def on_chunk_done(n: int) -> None:
        """Update the running total and write a periodic manifest when the threshold is crossed."""
        nonlocal total_lines_processed, next_manifest_threshold
        total_lines_processed += n
        while total_lines_processed >= next_manifest_threshold:
            write_manifest_atomic(run_dir, summary, total_lines_processed, start_inf, start_time, cli_args=cli_args)
            print(f"Wrote periodic manifest at {total_lines_processed} total lines (threshold {next_manifest_threshold}).")
            next_manifest_threshold += chunk_size

    common_kwargs = dict(
        model=model, tokenizer=tokenizer, device=device,
        nmax=nmax, chunk_size=chunk_size, file_name=file_name,
        cover_only=cover_only, stego_only=stego_only,
        print_tokens=print_tokens, dry_run=dry_run, on_chunk_done=on_chunk_done,
    )

    # Process .txt files sitting directly in data_root, then each subdirectory.
    dirs_to_process = [data_root] + sorted([d for d in data_root.iterdir() if d.is_dir()])
    for dataset_dir in dirs_to_process:
        has_txt = (
            (dataset_dir / file_name).exists() if file_name
            else (dataset_dir / 'cover.txt').exists() if cover_only
            else any(dataset_dir.glob("stego*.txt")) if stego_only
            else any(dataset_dir.glob("*.txt"))
        )
        if has_txt:
            processed_files, _ = process_dataset_dir(dataset_dir, run_dir, **common_kwargs)
            summary[dataset_dir.name] = f"{processed_files} files"

    write_manifest_atomic(run_dir, summary, total_lines_processed, start_inf, start_time, cli_args=cli_args)
    print("Run finished. Summary:", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run steganographic detection tests over text datasets")
    parser.add_argument("-nmax", type=int, default=None, help="maximum number of lines to read from each .txt file (default: all lines)")
    parser.add_argument("-file-name", "--file-name", dest="file_name", type=str, default=None,
                        help="only run inference on the exact file name given within each dataset folder")
    # If set, write a .tokens.jsonl file next to each generated .txt with token id arrays per line
    parser.add_argument("--print-tokens", action="store_true", dest="print_tokens",
                        help="write per-output token id JSONL files next to the .txt outputs")
    # If set, do NOT load LoRA weights (i.e. use base model only)
    parser.add_argument("--no-lora", action="store_true", dest="no_lora", help="use LoRA weights (default: use LoRA)")
    # How often (in number of outputs) to write the run manifest
    parser.add_argument("--manifest-threshold", dest="manifest_threshold", type=int, default=1000,
                        help="number of outputs between periodic manifest writes (default 1000)")
    # Quick dry-run to test file/manifest writes without loading the ML model
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False,
                        help="simulate generation without loading model (for testing manifest/output writes)")
    # If set, only process files named 'cover.txt' in each dataset folder
    parser.add_argument("--cover-only", dest="cover_only", action="store_true", default=False,
                        help="only run inference on files named 'cover.txt' (default: process all .txt files)")
    # If set, only process .txt files whose names start with 'stego' in each dataset folder
    parser.add_argument("--stego-only", dest="stego_only", action="store_true", default=False,
                        help="only run inference on .txt files whose names start with 'stego' (default: process all .txt files)")
    # Optional: override the data path (relative to project root) instead of using default data/baseline
    parser.add_argument("--data-path", dest="data_path", type=str, default=None,
                        help="path (relative to the data dir in the project root) to the data directory to process (overrides default 'data/baseline')")
    args = parser.parse_args()
    if args.file_name and (args.cover_only or args.stego_only):
        parser.error("--file-name cannot be combined with --cover-only or --stego-only")
    # Pass all parsed CLI argument values into the run manifest for reproducibility
    run_all_tests(nmax=args.nmax, print_tokens=args.print_tokens, use_lora=not args.no_lora, manifest_threshold=args.manifest_threshold, dry_run=args.dry_run, cli_args=vars(args))
