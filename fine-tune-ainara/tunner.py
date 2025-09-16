
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Iterable, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    logging as hf_logging,
)


DEFAULT_MODEL = os.environ.get("BASE_MODEL", "gpt2-medium")
DEFAULT_OUT = "./NARA-1r-beta1"
DEFAULT_MAXLEN = 256
DEFAULT_EPOCHS = 2
DEFAULT_BATCH = 4
DEFAULT_LR = 3e-5
DEFAULT_SAVE_STEPS = 1000
DEFAULT_LOGGING_STEPS = 100


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
hf_logging.set_verbosity_warning()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--knowledge", type=str, help="Path to single knowledge.json (optional)")
parser.add_argument("--data_dir", type=str, help="Directory containing .jsonl/.json files (optional). If provided, overrides --knowledge.")
parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base HF model name")
parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output dir to save model/tokenizer")
parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
parser.add_argument("--maxlen", type=int, default=DEFAULT_MAXLEN)
parser.add_argument("--lr", type=float, default=DEFAULT_LR)
parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS)
parser.add_argument("--logging_steps", type=int, default=DEFAULT_LOGGING_STEPS)
parser.add_argument("--streaming", action="store_true", help="Use streaming mode (low memory) for large datasets")
args = parser.parse_args()


print("PyTorch version:", torch.__version__)
use_cuda = torch.cuda.is_available()
print("CUDA available:", use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    try:
        torch.cuda.set_device(0)
        print("CUDA device count:", torch.cuda.device_count())
        print("Using device:", device, "name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("Could not set cuda:0, falling back. Error:", e)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device to be used:", device)

def extract_prompt_response(item: Dict) -> Tuple[str, str]:
    """
    Support varied formats:
     - {"prompt":"...","response":"..."}
     - {"question":"...","answer":"..."}
     - {"q":"...","a":"..."}
     - single-key mapping {"hi": "Hello..."} -> prompt=key, response=value
     - list of dicts [ {"question":"..","answer":".."}, ... ]
    """
    if item is None:
        return "", ""
    if isinstance(item, str):
        return item.strip(), ""
    if not isinstance(item, dict):
        return "", ""
 
    if "prompt" in item and "response" in item:
        return str(item.get("prompt", "")).strip(), str(item.get("response", "")).strip()
    if "question" in item and "answer" in item:
        return str(item.get("question", "")).strip(), str(item.get("answer", "")).strip()
    if "q" in item and "a" in item:
        return str(item.get("q", "")).strip(), str(item.get("a", "")).strip()

    # If it's a single mapping like {"hi": "Hello..."}
    string_fields = [k for k, v in item.items() if isinstance(v, str)]
    if len(string_fields) == 1 and len(item) == 1:
        k = list(item.keys())[0]
        v = item[k]
        return str(k).strip(), str(v).strip()

    # If multiple string fields, take first two as prompt/response
    if len(string_fields) >= 2:
        k0, k1 = string_fields[0], string_fields[1]
        return str(item[k0]).strip(), str(item[k1]).strip()

 
    for k, v in item.items():
        if isinstance(v, str) and v.strip():
            return v.strip(), ""
    return "", ""

def iter_files_as_items(files: Iterable[str]) -> Iterable[Any]:
    """
    Iterate over files yielding parsed JSON objects (dicts or primitives).
    Supports .jsonl (one JSON object per line) and .json (either dict or list).
    For a top-level dict in .json, yields single-key dicts like {"prompt": "..."}? No -
    we yield the original dict entries (so extract_prompt_response handles mapping).
    """
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        if p.suffix == ".jsonl":
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception as e:
                        logger.warning("Skipping invalid json line in %s: %s", f, e)
                        continue
                 
                    yield obj
        elif p.suffix == ".json":
            try:
                with p.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                logger.warning("Skipping invalid json file %s: %s", f, e)
                continue
           
            if isinstance(data, dict):
          
                all_vals_are_strings = all(isinstance(v, str) for v in data.values())
                if all_vals_are_strings:
                    for k, v in data.items():
                        yield {k: v}
                else:
                   
                    yield data
            elif isinstance(data, list):
                for item in data:
                    yield item
            else:
            
                continue
        else:
        
            continue

dataset = None
if args.data_dir:
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"data_dir not found: {data_path}")
        sys.exit(1)
    files = sorted([str(p) for p in data_path.rglob("*.jsonl")] + [str(p) for p in data_path.rglob("*.json")])
    if not files:
        logger.error(f"No .jsonl or .json files found under {data_path}")
        sys.exit(1)
    logger.info(f"Found {len(files)} files under {data_path}; streaming={args.streaming}")

    if args.streaming:
       
        items_iter = iter_files_as_items(files)
       
        dataset = items_iter  
    else:
      
        texts = []
        for item in iter_files_as_items(files):
            p, r = extract_prompt_response(item)
            if p:
                texts.append(f"User: {p}\nAI: {r}")
        if not texts:
            logger.error("No examples parsed from data_dir files.")
            sys.exit(1)
        dataset = Dataset.from_dict({"text": texts})
elif args.knowledge:
    knowledge_path = Path(args.knowledge)
    if not knowledge_path.exists():
        logger.error(f"knowledge file not found: {knowledge_path}")
        sys.exit(1)
   
    if args.streaming:
       
        items_iter = iter_files_as_items([str(knowledge_path)])
        dataset = items_iter
    else:
        
        try:
            with knowledge_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            logger.error("Failed to load knowledge file: %s", e)
            sys.exit(1)
        texts = []
        if isinstance(data, dict):
          
            all_vals_are_strings = all(isinstance(v, str) for v in data.values())
            if all_vals_are_strings:
                for k, v in data.items():
                    texts.append(f"User: {k}\nAI: {v}")
            else:
                
                p, r = extract_prompt_response(data)
                if p:
                    texts.append(f"User: {p}\nAI: {r}")
        elif isinstance(data, list):
            for item in data:
                p, r = extract_prompt_response(item)
                if p:
                    texts.append(f"User: {p}\nAI: {r}")
        else:
            logger.error("Unsupported knowledge.json structure.")
            sys.exit(1)
        if not texts:
            logger.error("No examples parsed from knowledge.json")
            sys.exit(1)
        dataset = Dataset.from_dict({"text": texts})
else:
    logger.error("Either --data_dir or --knowledge must be provided.")
    sys.exit(1)



base_model = args.model
print("Loading tokenizer from:", base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model from:", base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model.to(device)


def build_text_from_item(item: Dict) -> str:
    p, r = extract_prompt_response(item)
    p = "" if p is None else str(p).strip()
    r = "" if r is None else str(r).strip()
    return f"User: {p}\nAI: {r}"

def tokenize_batch(batch):
    out = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=args.maxlen)
    out["labels"] = [ids.copy() for ids in out["input_ids"]]
    return out


train_dataset = None
if args.streaming:

    from torch.utils.data import IterableDataset

    class TokenizedIterableDataset(IterableDataset):
        def __init__(self, items_iterable, tokenizer, maxlen):
            self.items_iterable = items_iterable
            self.tokenizer = tokenizer
            self.maxlen = maxlen

        def __iter__(self):
            for item in self.items_iterable:
                text = build_text_from_item(item)
                if not text.strip():
                    continue
                enc = self.tokenizer(text, truncation=True, max_length=self.maxlen, padding="max_length")
                ids = enc["input_ids"]
                att = enc["attention_mask"]
                yield {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(att, dtype=torch.long),
                    "labels": torch.tensor(ids, dtype=torch.long),
                }

    if isinstance(dataset, Iterable):
        train_dataset = TokenizedIterableDataset(dataset, tokenizer, args.maxlen)
    else:
    
        train_dataset = TokenizedIterableDataset(iter([]), tokenizer, args.maxlen)
else:
   
    if not isinstance(dataset, Dataset):
        logger.error("Internal error: expected Dataset in non-streaming mode.")
        sys.exit(1)
    logger.info("Dataset contains %d examples before tokenization.", len(dataset))
    logger.info("Tokenizing dataset (padding to max_length, labels set)...")
    tokenized = dataset.map(tokenize_batch, batched=True, batch_size=1000, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataset = tokenized
    logger.info("Tokenization complete. Columns: %s", train_dataset.column_names)

outdir = args.out
os.makedirs(outdir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=outdir,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    logging_dir="./logs",
    learning_rate=args.lr,
    weight_decay=0.01,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    save_total_limit=2,
    fp16=use_cuda,
    report_to=[], 
)

print("TrainingArguments:", training_args)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


print("Starting training. This will use GPU if available.")
trainer.train()


print("Saving model and tokenizer to", outdir)
model.save_pretrained(outdir)
tokenizer.save_pretrained(outdir)
print("Training finished and model saved to", outdir)
