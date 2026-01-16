# -*- coding: utf-8 -*-
"""
Test LoRA fine-tuned Llama model for steganographic text detection
"""

# Third-party modules
from time import time
start = time()
import torch
import json
from transformers import LlamaForCausalLM, LlamaTokenizer
import safetensors
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from tqdm import tqdm
import argparse
from typing import Optional
from datetime import datetime, timezone

# Custom modules
from utils import get_device, get_root_dir

from pathlib import Path


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


def load_model_and_tokenizer(use_lora: bool = True, lora_weight_path: str = "checkpoint-22000"):
    """Load Llama model, tokenizer and optionally apply LoRA weights. Returns (model, tokenizer, device)."""
    device = get_device()
    print("Using device:", device)
    model_name_or_path = "linhvu/decapoda-research-llama-7b-hf"
    print("Using model:", model_name_or_path)

    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
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
        lora_weight_path = f"{lora_weight_path}/pytorch_model.bin"
        if Path(lora_weight_path).exists():
            ckpt_name = lora_weight_path
            lora_weight = torch.load(ckpt_name)
            set_peft_model_state_dict(model, lora_weight)
        else:
            print(f"Warning: LoRA weight path {lora_weight_path} does not exist; continuing without LoRA weights")
    model.eval()
    return model, tokenizer, device


def run_inference(texts, model, tokenizer, device):
    """Run generation for a list of texts using an already-loaded model/tokenizer.

    Returns a list of generated string outputs. If `return_tokens=True` is passed
    (see wrapper use below), this function can also return a parallel list of
    generated token id lists for each input (one list per line).
    """
    answers = []
    token_ids_lists = []
    for text in tqdm(texts):
        input_text = f"### Text:\n{text.strip()}\n\n### Question:\nIs the above text steganographic?\n\n### Answer:\n"
        input_text = tokenizer.bos_token + input_text if tokenizer.bos_token is not None else input_text
        inputs = tokenizer([input_text], return_tensors="pt").to(device)
        predict = model.generate(**inputs, num_beams=1, do_sample=False, max_new_tokens=10, min_new_tokens=2)
        gen_ids = predict[0][inputs.input_ids.shape[1]:]
        # Convert token ids to plain Python list (move to CPU first)
        try:
            ids_list = gen_ids.cpu().tolist()
        except Exception:
            # fallback if gen_ids is already a list-like
            ids_list = list(gen_ids)
        token_ids_lists.append(ids_list)
        # skip special tokens and clean up spaces
        output = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        # make output UTF-8 safe (replace undecodable/surrogate characters)
        output = output.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        answers.append(output.strip())
    # Return both lists; caller may choose to ignore tokens if not needed
    return answers, token_ids_lists


def run_all_tests(nmax: int = -1, lora_weight_path: str = "checkpoint-22000", print_tokens: bool = False, use_lora: bool = True):
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

    # Load model and tokenizer once
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
        for txt_file in sorted(dataset_dir.glob("*.txt")):
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
            try:
                outputs, token_lists = run_inference(texts, model, tokenizer, device)
            except Exception as e:
                print(f"  Inference failed for {txt_file}: {e}")
                continue
            out_path = out_dataset_dir / txt_file.name
            save_outputs_to_file(outputs, str(out_path))
            # If requested, write token id lists to a JSONL file next to the output .txt
            if print_tokens:
                # Replace .txt suffix with .tokens.jsonl (e.g. cover.txt -> cover.tokens.jsonl)
                tokens_path = out_path.with_suffix('.tokens.jsonl')
                tokens_path.parent.mkdir(parents=True, exist_ok=True)
                with tokens_path.open('w', encoding='utf-8') as tf:
                    for ids in token_lists:
                        tf.write(json.dumps(ids, ensure_ascii=False) + '\n')
            # Only count lines after successful generation + save
            total_lines_processed += len(texts)
            processed_files += 1
        summary[rel_dataset] = f"{processed_files} files"

    # Save a small manifest for the run
    manifest = {
        "run_dir": str(run_dir),
        "lora_weight_path": lora_weight_path,
        # Use ISO 8601 UTC timestamp without microseconds, e.g. 2026-01-15T14:32:00Z
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summary,
        "total_lines_processed": total_lines_processed,
        # total processing time in seconds (rounded to milliseconds)
        "total_inference_time_seconds": round(time() - start_inf, 3),
        "total_time_seconds": round(time() - start_time, 3)
    }
    # Write manifest explicitly with UTF-8 encoding and preserve non-ASCII characters
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Run finished. Summary:", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run steganographic detection tests over text datasets")
    parser.add_argument("-nmax", type=int, default=-1, help="maximum number of lines to read from each .txt file (-1 for all)")
    # Path to LoRA weights (folder containing pytorch_model.bin or similar)
    parser.add_argument("--lora-path",dest='lora_path', type=str, default="checkpoint-22000", help="path to the LoRA weights folder (default: checkpoint-22000)")
    # If set, write a .tokens.jsonl file next to each generated .txt with token id arrays per line
    parser.add_argument("--print-tokens", action="store_true", dest="print_tokens",
                        help="write per-output token id JSONL files next to the .txt outputs")
    parser.add_argument("--use-lora", action="store_true", dest="use_lora", default=True, help="use LoRA weights (default: True)")
    args = parser.parse_args()
    run_all_tests(nmax=args.nmax, lora_weight_path=args.lora_path, print_tokens=args.print_tokens, use_lora=args.use_lora)
