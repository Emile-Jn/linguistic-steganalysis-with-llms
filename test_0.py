"""
A minimal version of test.ipynb, to run inference on example data.

This script loads a model/tokenizer directly from Hugging Face transformers (default: gpt2)
and runs short generation on sample texts found under the `data/` folder.

Usage examples:
  python3 test_0.py --model_name_or_path gpt2 --num_samples 2
  python3 test_0.py --text "Some sentence to classify." --model_name_or_path gpt2
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple
# import protobuf # required to download the model linhvu/decapoda-research-llama-7b-hf
import tiktoken # required for transformers.slow_tokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("test_script")


def find_sample_texts(data_root: Path, max_samples: int = 4) -> List[Tuple[str, str]]:
    """Search data/* directories for cover.txt and stego.txt and return list of (label, text).

    Returns up to max_samples texts (mix of cover/stego if available).
    """
    samples: List[Tuple[str, str]] = []
    if not data_root.exists():
        return samples

    for sub in sorted(data_root.iterdir()):
        if not sub.is_dir():
            continue
        # look for cover.txt and stego.txt
        for fname, label in (("cover.txt", "non-steganographic"), ("stego.txt", "steganographic")):
            p = sub / fname
            if p.exists():
                try:
                    text = p.read_text(encoding="utf-8").strip().splitlines()
                    if not text:
                        continue
                    # take the first non-empty line
                    for line in text:
                        s = line.strip()
                        if s:
                            samples.append((label, s))
                            break
                except Exception:
                    continue
        if len(samples) >= max_samples:
            break
    return samples[:max_samples]


PROMPT_TEMPLATE = (
    "### Text:\n{txt}\n\n"
    "### Question:\nIs the above text steganographic or non-steganographic?\n\n"
    "### Answer:\n"
)


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(txt=text.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="linhvu/decapoda-research-llama-7b-hf", help="Hugging Face model name or local path")
    parser.add_argument("--device", type=str, default=None, help="torch device (e.g. cpu or cuda). If not set, will use cuda if available.")
    parser.add_argument("--num_samples", type=int, default=4, help="How many sample texts to run (from data/).")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="Generation length for answers")
    parser.add_argument("--text", type=str, default=None, help="Optional single text to run instead of reading data/")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    # Collect samples
    samples: List[Tuple[str, str]] = []
    if args.text:
        samples = [("unknown", args.text.strip())]
    else:
        samples = find_sample_texts(Path("data"), max_samples=args.num_samples)

    if not samples:
        # fallback sample
        samples = [("unknown", "i guess if a film has magic i do n't need it to be fluid or seamless")]
        LOGGER.info("No sample texts found in data/; using fallback sample.")

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    except Exception as e:
        LOGGER.error("Failed to load tokenizer for %s: %s", args.model_name_or_path, e)
        raise

    # Some small models (gpt2) don't have a pad token; set it to eos to avoid warnings when batching
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    except Exception as e:
        LOGGER.error("Failed to load model %s: %s", args.model_name_or_path, e)
        raise

    model.to(device)
    model.eval()

    # Run generation for each sample
    for i, (label, text) in enumerate(samples, start=1):
        prompt = build_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

        with torch.no_grad():
            try:
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except Exception as e:
                LOGGER.error("Generation failed for sample %d: %s", i, e)
                continue

        # Decode only the newly generated part (after the prompt)
        generated = out[0][input_ids.shape[1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()

        print("--- Sample %d ---" % i)
        print("Label:", label)
        print("Text:", text)
        print("Prompt:", prompt.replace('\n', '\\n'))
        print("Model answer:", decoded)
        print()


if __name__ == "__main__":
    main()
