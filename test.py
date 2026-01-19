# -*- coding: utf-8 -*-
"""
Test LoRA fine-tuned Llama model for steganographic text detection
"""

# Third-party modules
from time import time
start = time()
import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    PeftModel,
)
import argparse
from typing import Optional
from datetime import datetime, timezone

# Custom modules
from utils import get_device, get_root_dir

from pathlib import Path

# Prefer 32 for A100 throughput
BATCH_SIZE = 32


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
def write_manifest_atomic(run_dir: Path, summary: dict, total_lines_processed: int, start_inf: float, start_time: float, lora_weight_path: str, cli_args: Optional[dict] = None):
    """Write the run manifest to run_dir/manifest.json atomically.

    Args:
        run_dir: Path to the run directory
        summary: per-dataset summary dict
        total_lines_processed: integer total of lines processed so far
        start_inf: inference start time (seconds since epoch)
        start_time: overall run start time (seconds since epoch)
        lora_weight_path: descriptive path used for this run
    """
    manifest = {
        "run_dir": str(run_dir),
        "lora_weight_path": lora_weight_path,
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


def load_model_and_tokenizer_old(use_lora: bool = True, lora_weight_path: str = "checkpoint-22000"):
    """Old version
    Load Llama model, tokenizer and optionally apply LoRA weights. Returns (model, tokenizer, device).
    """
    # Import torch lazily so running parts of the script that don't need the model
    # (e.g. listing files or dry-run) won't require PyTorch to be installed.
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required to load the model. Install torch or run with --dry-run") from e

    device = get_device()
    print("Using device:", device)
    model_name_or_path = "linhvu/decapoda-research-llama-7b-hf"
    print("Using model:", model_name_or_path)

    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)

    # debugging: check that eos tokens match
    print('\n- - - - - Debug info - - - - -\n')
    print(tokenizer.special_tokens_map)
    print(tokenizer.eos_token_id)
    print(model.config.eos_token_id)
    print('\n- - - - - - - - - - -  - - - -\n')

    # Load LoRA configuration and weights
    # Ensure JSON config files are read with explicit UTF-8 encoding
    with open("configs/TrainLM_llama-7b-hf.json", "r", encoding="utf-8") as mf:
        model_config = json.load(mf)
    with open("configs/lora_config.json", "r", encoding="utf-8") as lf:
        lora_hyperparams = json.load(lf)
    target_modules = ["query_key_value"]
    if model_config.get('model_type') == "llama":
        target_modules = lora_hyperparams.get('lora_target_modules', target_modules)
    config = LoraConfig(
        r=lora_hyperparams['lora_r'],
        lora_alpha=lora_hyperparams['lora_alpha'],
        target_modules=target_modules,
        lora_dropout=lora_hyperparams['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if use_lora:
        lora_weight_path = f"adapter/{lora_weight_path}/pytorch_model.bin"
        if Path(lora_weight_path).exists():
            ckpt_name = lora_weight_path
            lora_weight = torch.load(ckpt_name)
            set_peft_model_state_dict(model, lora_weight)
        else:
            print(f"Warning: LoRA weight path adapter/{lora_weight_path}/pytorch_model.bin does not exist; continuing without LoRA weights")
    model.eval()
    return model, tokenizer, device

def load_model_and_tokenizer(use_lora: bool = True, lora_weight_path: str = ""):
    """
    New implementation of load_model_and_tokenizer with improved LoRA loading.
    """
    base_model_path = "linhvu/decapoda-research-llama-7b-hf"
    print("Using base model:", base_model_path)

    if use_lora:
        adapter_path = "adapter"
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

    # debugging: check that eos tokens match
    print('\n- - - - - Debug info - - - - -\n')
    print(tokenizer.special_tokens_map)
    print(tokenizer.eos_token_id)
    print(model.config.eos_token_id)
    print('\n- - - - - - - - - - -  - - - -\n')

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


def run_all_tests(nmax: int = -1, lora_weight_path: str = "checkpoint-22000", print_tokens: bool = False, use_lora: bool = True, manifest_threshold: int = 1000, dry_run: bool = False, cli_args: Optional[dict] = None):
    """Top-level pipeline (no args):
    - Creates logs/ if missing
    - Creates a new run_X folder inside logs/
    - For each subfolder in data/, runs inference on every .txt file and writes outputs preserving folder structure

    Args:
        nmax: maximum number of lines to take from each .txt file. If -1, take all lines.
        lora_weight_path: path to the LoRA weights folder
        print_tokens: whether to print generated token IDs to console
        use_lora: whether to load LoRA weights (fine-tuned weights from a checkpoint) into the model
    """
    # Backwards-compatible: if the caller passed a cover_only flag (from CLI), accept it from cli_args
    # Prefer explicit parameter in the future; for now, check cli_args dict if provided.
    cover_only = False
    if isinstance(cli_args, dict) and 'cover_only' in cli_args:
        cover_only = bool(cli_args.get('cover_only'))

    # The function signature does not accept cover_only directly to minimize changes to other call sites.
    try:
        # try to resolve project root via utils.get_root_dir()
        root = Path(get_root_dir())
    except Exception:
        # fallback to current working directory when project name lookup fails
        root = Path.cwd()
    # Track total lines processed and timing for the whole run
    total_lines_processed = 0
    start_time = time()
    data_root = root / "data"
    logs_root = root / "logs"

    # Decide run directory name. Assumption: use underscore form 'run_X' (safe on filesystems).
    run_dir = get_next_run_dir(logs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating new run directory: {run_dir}")

    # Initialize next threshold for periodic manifest writes
    # The threshold is configurable via the `manifest_threshold` parameter (default 1000)
    next_manifest_threshold = int(manifest_threshold) if manifest_threshold and manifest_threshold > 0 else 1000

    # Load model and tokenizer once (skip if dry_run)
    model = tokenizer = device = None
    if not dry_run:
        try:
            model, tokenizer, device = load_model_and_tokenizer(use_lora=use_lora, lora_weight_path=lora_weight_path)
        except Exception as e:
            print("Failed to load model/tokenizer:", e)
            return

    summary = {}

    if not data_root.exists():
        print(f"Data root {data_root} does not exist. Nothing to process.")
        return

    start_inf = time()
    for dataset_dir in sorted([d for d in data_root.iterdir() if d.is_dir()]):
        rel_dataset = dataset_dir.name
        out_dataset_dir = run_dir / rel_dataset
        out_dataset_dir.mkdir(parents=True, exist_ok=True)
        processed_files = 0
        # Choose which files to iterate based on cover_only flag
        if cover_only:
            # Only process a file named exactly 'cover.txt' if it exists in the dataset dir
            candidate = dataset_dir / 'cover.txt'
            txt_files = [candidate] if candidate.exists() else []
        else:
            txt_files = sorted(dataset_dir.glob("*.txt"))

        for txt_file in txt_files:
            print(f"Processing {txt_file}...")
            texts = read_txt_lines(str(txt_file))
            # apply nmax truncation if requested
            if nmax is not None and nmax != -1:
                if len(texts) > nmax:
                    texts = texts[:nmax]
                    print(f"  Truncated to first {nmax} lines for {txt_file}")
            if not texts:
                print(f"  Skipping empty or missing file {txt_file}")
                continue
            # Process the file in chunks so outputs are flushed periodically (including writing
            # the outputs file and optional token JSONL). This ensures large input files do not
            # only get written once at the end.
            chunk_size = int(manifest_threshold) if manifest_threshold and manifest_threshold > 0 else 1000
            out_path = out_dataset_dir / txt_file.name
            tokens_path = out_path.with_suffix('.tokens.jsonl')
            # Ensure parent exists
            out_path.parent.mkdir(parents=True, exist_ok=True)
            wrote_any = False
            try:
                for i in range(0, len(texts), chunk_size):
                    chunk = texts[i:i+chunk_size]
                    # In dry-run mode, simulate outputs quickly without calling the model
                    if dry_run:
                        outputs = [f"DRYRUN_OUTPUT_LINE_{j}" for j in range(len(chunk))]
                        token_lists = [[] for _ in chunk]
                    else:
                        # Run batched inference (tokenization+generation). Single call only.
                        outputs, token_lists = run_inference(chunk, model, tokenizer, device, batch_size=BATCH_SIZE)
                    # Write outputs to file incrementally. Use 'w' for the first write to
                    # replace any existing file, then append for subsequent chunks.
                    mode = 'w' if not wrote_any else 'a'
                    with out_path.open(mode, encoding='utf-8') as of:
                        for out in outputs:
                            of.write(out.replace('\n', ' ').strip() + '\n')
                    # If requested, append token id lists to the tokens JSONL file
                    if print_tokens:
                        tokens_path.parent.mkdir(parents=True, exist_ok=True)
                        mode_t = 'w' if not wrote_any else 'a'
                        with tokens_path.open(mode_t, encoding='utf-8') as tf:
                            for ids in token_lists:
                                tf.write(json.dumps(ids, ensure_ascii=False) + '\n')

                    # Update counters after chunk saved
                    total_lines_processed += len(chunk)
                    wrote_any = True

                    # Write manifest periodically every `manifest_threshold` outputs.
                    while total_lines_processed >= next_manifest_threshold:
                        write_manifest_atomic(run_dir, summary, total_lines_processed, start_inf, start_time, lora_weight_path, cli_args=cli_args)
                        print(f"Wrote periodic manifest at {total_lines_processed} total lines (threshold {next_manifest_threshold}).")
                        next_manifest_threshold += chunk_size
                processed_files += 1
            except Exception as e:
                print(f"  Inference failed for {txt_file}: {e}")
                # don't count this file as processed
                continue
        summary[rel_dataset] = f"{processed_files} files"

    # Save a small manifest for the run (final write) including CLI args
    write_manifest_atomic(run_dir, summary, total_lines_processed, start_inf, start_time, lora_weight_path, cli_args=cli_args)
    print("Run finished. Summary:", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run steganographic detection tests over text datasets")
    parser.add_argument("-nmax", type=int, default=-1, help="maximum number of lines to read from each .txt file (-1 for all)")
    # Path to LoRA weights (folder containing pytorch_model.bin or similar)
    parser.add_argument("--lora-path",dest='lora_path', type=str, default="checkpoint-22000", help="path to the LoRA weights folder (default: checkpoint-22000)")
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
    args = parser.parse_args()
    # Pass all parsed CLI argument values into the run manifest for reproducibility
    run_all_tests(nmax=args.nmax, lora_weight_path=args.lora_path, print_tokens=args.print_tokens, use_lora=not args.no_lora, manifest_threshold=args.manifest_threshold, dry_run=args.dry_run, cli_args=vars(args))
