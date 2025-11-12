#!/usr/bin/env python
"""
Combined Chess Model Fine-tuning and Evaluation Script
Handles data splitting, training, and evaluation in one workflow.
"""
import os
import sys

# Cache models in storage to avoid filling 30GB home quota
os.environ['HF_HOME'] = '~/scratch/huggingface_cache'
os.environ['TORCH_HOME'] = '~/scratch/torch_cache'
from pathlib import Path
import argparse
import json
import time
import re
import zstandard as zstd
import pandas as pd
from tqdm import tqdm
import chess
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, PeftModel


# =========================================================
# CLI Arguments
# =========================================================
parser = argparse.ArgumentParser(description="Fine-tune and evaluate chess model")
parser.add_argument("--dataset", required=True, help="Path to lichess_db_eval.jsonl.zst")
parser.add_argument("--model", required=True, help="Hugging Face model name")
parser.add_argument("--mode", choices=["train", "eval", "both"], default="both",
                    help="Run training, evaluation, or both")
parser.add_argument("--output_dir", default="./finetuned_gpt_20", help="Output directory for trained model")
parser.add_argument("--eval_samples", type=int, default=100, help="Number of samples for evaluation")

# Training hyperparameters
parser.add_argument("--train_samples", type=int, default=70000, help="Number of training samples")
parser.add_argument("--val_samples", type=int, default=1000, help="Number of validation samples")
parser.add_argument("--test_samples", type=int, default=1000, help="Number of test samples")
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--learning_rate", type=float, default=1.5e-5, help="Learning rate")
parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
parser.add_argument("--eval_dtype",choices=["fp16", "bf16", "fp32"],default="bf16",help="dtype for evaluation (fp16 recommended for many HF LLMs)")

parser.add_argument("--save_strategy", choices=["no", "steps", "epoch"], default="steps",
                    help="When to save checkpoints")
parser.add_argument("--save_steps", type=int, default=1000,
                    help="Save every N steps when --save_strategy=steps")
parser.add_argument("--eval_strategy", choices=["no", "steps", "epoch"], default="steps",
                    help="When to run evaluation")
parser.add_argument("--load_best_model_at_end", action="store_true",
                    help="Load the best checkpoint (by eval_loss) at end of training")
parser.add_argument("--save_total_limit", type=int, default=3,
                    help="How many recent checkpoints to keep")
parser.add_argument("--resume", action="store_true",
                    help="Resume training from the latest checkpoint in output_dir if present")

args = parser.parse_args()


# =========================================================
# Directory Setup
# =========================================================
repo_root = os.path.dirname(__file__)
models_dir = os.path.join(repo_root, "models")
results_dir = os.path.join(repo_root, "results")
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Device: {device}")
print(f"[INIT] Repository root: {repo_root}")
print(f"[INIT] Models directory: {models_dir}")
print(f"[INIT] Results directory: {results_dir}")


# =========================================================
# Utility Functions
# =========================================================
def sanitize_repo_id(repo_id: str) -> str:
    """Convert repo ID to valid directory name."""
    return repo_id.replace("/", "_").replace(":", "_")

def _str_to_dtype(name: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def extract_best_move(obj):
    """Extract the top-rated move from evaluation data."""
    fen = obj.get("fen")
    if not fen:
        return None, None
    
    board = chess.Board(fen)
    
    # Get the best move from the first PV
    for ev in obj.get("evals", []):
        for pv in ev.get("pvs", []):
            line = pv.get("line")
            if not line:
                continue
            first_uci = line.split()[0]
            try:
                mv = chess.Move.from_uci(first_uci)
                if mv in board.legal_moves:
                    return fen, board.san(mv)
            except Exception:
                continue
    
    return None, None


def extract_top_moves(obj, max_moves=5):
    """Extract top SAN moves from evals->pvs->line (first UCI of each PV)."""
    fen = obj.get("fen")
    if not fen:
        return []
    board = chess.Board(fen)

    moves = []
    for ev in obj.get("evals", []):
        for pv in ev.get("pvs", []):
            line = pv.get("line")
            if not line:
                continue
            first_uci = line.split()[0]
            try:
                mv = chess.Move.from_uci(first_uci)
                if mv in board.legal_moves:
                    moves.append(board.san(mv))
            except Exception:
                continue

    # Deduplicate, preserve order
    seen, uniq = set(), []
    for m in moves:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq[:max_moves]


def load_data_splits(path, train_n, val_n, test_n):
    """
    Load data and split into train/val/test sets.
    Returns three lists of samples.
    """
    print(f"[DATA] Loading {train_n + val_n + test_n} total samples from {path}")
    print(f"[DATA] Split: {train_n} train, {val_n} val, {test_n} test")
    
    total_needed = train_n + val_n + test_n
    all_samples = []
    
    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(f) as reader:
            buffer = b""
            for chunk in iter(lambda: reader.read(16384), b""):
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    try:
                        obj = json.loads(line)
                        fen, best_move = extract_best_move(obj)
                        top_moves = extract_top_moves(obj)
                        if fen and best_move and top_moves:
                            all_samples.append({
                                "fen": fen,
                                "best_move": best_move,
                                "top_moves": top_moves
                            })
                        if len(all_samples) >= total_needed:
                            break
                    except Exception:
                        continue
                if len(all_samples) >= total_needed:
                    break
    
    print(f"[DATA] Loaded {len(all_samples)} total samples")
    
    # Split the data
    train_data = all_samples[:train_n]
    val_data = all_samples[train_n:train_n + val_n]
    test_data = all_samples[train_n + val_n:train_n + val_n + test_n]
    
    print(f"[DATA] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data

def discover_linear_leaf_names(model, max_examples=50):
    """
    Return a sorted set of *leaf* module names that are nn.Linear.
    (Lora target_modules should be leaf names like 'q_proj', 'k_proj', 'o_proj', etc.)
    """
    names = set()
    for name, module in model.named_modules():
        # keep only leaves
        if any(True for _ in module.children()):
            continue
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            names.add(leaf)
            if len(names) >= max_examples:
                break
    return sorted(names)

# =========================================================
# Training Functions
# =========================================================
def format_training_example(fen, best_move, tokenizer):
    """Format a training example."""
    board = chess.Board(fen)
    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    board_display = board.unicode(empty_square="·", borders=True)
    
    messages = [
        {"role": "system", "content":
            "You are a chess assistant. Reply with exactly two lines:\n"
            "My reasoning: <15 words max>\n"
            "Best Move: <legal SAN only>"
        },
        {"role": "user", "content":
            f"Side to move: {side_to_move}\nBoard:\n{board_display}"
        },
        {"role": "assistant", "content":
            f"My reasoning: Strong tactical position.\nBest Move: {best_move}"
        }
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    return text


def create_dataset(data, tokenizer, max_length):
    """Convert raw data to tokenized dataset."""    
    texts = [format_training_example(item["fen"], item["best_move"], tokenizer) 
             for item in tqdm(data, desc="Formatting")]
    
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    return tokenized


def train_model(train_data, val_data, model_path, tokenizer, model):
    """Fine-tune the model."""
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    train_dataset = create_dataset(train_data, tokenizer, args.max_length)
    val_dataset = create_dataset(val_data, tokenizer, args.max_length)
    
    print(f"[TRAIN] Training samples: {len(train_dataset)}")
    print(f"[TRAIN] Validation samples: {len(val_dataset)}")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,

        # QLoRA: keep per-device batch size tiny and use grad accumulation
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,

        learning_rate=args.learning_rate,  # 1e-4 to 2e-4 typical for LoRA; your default is ok
        warmup_steps=50,

        # Logging / eval / saving cadence
        logging_steps=200,
        eval_strategy=args.eval_strategy,                   # keep your CLI option
        eval_steps=args.save_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,

        # Checkpoint retention & best model
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Precision + memory
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        save_safetensors=True,
        save_only_model=True,          # keep checkpoints small

        # Optimizer that plays well with bitsandbytes paging
        optim="adamw_torch",

        # Reduce eval memory spikes
        eval_accumulation_steps=32,

        report_to="none",
        overwrite_output_dir=False,
    )

    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("[TRAIN] Starting fine-tuning...")
    start_time = time.time()
    if args.resume:
        last = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
        if last is not None:
            print(f"[TRAIN] Resuming from checkpoint: {last}")
            resume_ckpt = last
        else:
            print("[TRAIN] No checkpoint found; starting fresh.")
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"[TRAIN] Training completed in {training_time/60:.2f} minutes")
    print(f"[TRAIN] Saving model to {args.output_dir}")
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training info
    info = {
        "base_model": args.model,
        "training_samples": len(train_data),
        "validation_samples": len(val_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "training_time_minutes": training_time / 60,
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print("[TRAIN] Training info saved")
    return trainer.model


# =========================================================
# Evaluation Functions
# =========================================================
SAN_RE = re.compile(r"\b(O-O-O|O-O|[KQRNB]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRNB])?[+#]?)\b")


class StopAfterBestMove(StoppingCriteria):
    """Stop generation after detecting a move following 'Best Move:'"""
    def __init__(self, tokenizer, start_idx):
        self.tok = tokenizer
        self.start_idx = start_idx
        self.marker = "best move:"

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tok.decode(input_ids[0, self.start_idx:], skip_special_tokens=True)
        i = text.lower().find(self.marker)
        if i == -1:
            return False
        tail = text[i + len(self.marker):]
        return SAN_RE.search(tail) is not None


def query_model(fen, tokenizer, model, idx=None):
    """Query model and return (san_move, full_reply)."""
    board = chess.Board(fen)
    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    board_display = board.unicode(empty_square="·", borders=True)

    messages = [
        {"role": "system", "content":
            "You are a chess assistant. Reply with exactly two lines:\n"
            "My reasoning: <15 words max>\n"
            "Best Move: <legal SAN only>"
        },
        {"role": "user", "content":
            f"Side to move: {side_to_move}\nBoard:\n{board_display}"
        },
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # simple instruct-style fallback
        prompt_text = (
            "You are a chess assistant. Reply with exactly two lines:\n"
            "My reasoning: <15 words max>\n"
            "Best Move: <legal SAN only>\n\n"
            f"Side to move: {side_to_move}\nBoard:\n{board_display}\n"
            "Answer:\n"
        )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        stopping_criteria = StoppingCriteriaList([StopAfterBestMove(tokenizer, input_len)])
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            no_repeat_ngram_size=5,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            stopping_criteria=stopping_criteria,
        )

    new_tokens = output_ids[0, input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse after "Best Move:"
    window = completion
    j = window.lower().rfind("best move:")
    window = window[j + len("best move:"):].strip() if j != -1 else ""

    san = ""
    if window:
        m = SAN_RE.search(window)
        cand = m.group(1) if m else ""
        if cand:
            cleaned = re.sub(r"[+#]+$", "", cand.strip())
            try:
                mv = board.parse_san(cleaned)
                san = board.san(mv)
            except Exception:
                for move in board.legal_moves:
                    if board.san(move).replace("+", "").replace("#", "").lower() == cleaned.lower():
                        san = board.san(move)
                        break

    if idx is not None:
        print(f"[EVAL] #{idx}: SAN='{san}'")
        if not san:
            print("[WARN] Could not parse a valid SAN after 'Best Move:'.")
    
    return san, completion


def evaluate_move(fen, move_san, top_moves_san):
    """Check if move is legal and if it's in top moves."""
    board = chess.Board(fen)
    legal = False
    try:
        parsed = board.parse_san(move_san)
        legal = parsed in board.legal_moves
    except Exception:
        legal = False
    good = move_san in top_moves_san
    return legal, good


def evaluate_model(test_data, tokenizer, model, model_name):
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    print(f"[EVAL] Evaluating on {len(test_data)} test samples...")
    
    records = []
    for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        try:
            fen = sample["fen"]
            top_moves = sample["top_moves"]
            
            move_san, full_reply = query_model(fen, tokenizer, model, idx=i)
            legal, good = evaluate_move(fen, move_san, top_moves)
            
            records.append({
                "id": i,
                "fen": fen,
                "model_move": move_san,
                "full_reply": full_reply,
                "legal": legal,
                "good": good,
                "top_moves": ",".join(top_moves)
            })
        except Exception as e:
            print(f"[WARN] Error on sample #{i}: {e}")
            continue
    
    # Calculate metrics
    df = pd.DataFrame(records)
    if df.empty or any(col not in df.columns for col in ("legal", "good")):
        print("[ERROR] No successful generations; saving raw outputs for debugging.")
        outfile = os.path.join(results_dir, f"evaluate_{sanitize_repo_id(model_name)}.csv")
        df.to_csv(outfile, index=False)
        return df

    legal_rate = df["legal"].mean() * 100
    good_rate = df["good"].mean() * 100
    
    print(f"\n[RESULTS] Evaluation completed on {len(records)} samples")
    print(f"[RESULTS] Legal move rate: {legal_rate:.2f}%")
    print(f"[RESULTS] Top move rate: {good_rate:.2f}%")
    
    # Save results
    outfile = os.path.join(results_dir, f"evaluate_{sanitize_repo_id(model_name)}.csv")
    df.to_csv(outfile, index=False)
    print(f"[SAVE] Results saved to {outfile}")
    
    # Save summary
    summary = {
        "model": model_name,
        "test_samples": len(records),
        "legal_move_rate": legal_rate,
        "top_move_rate": good_rate,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_file = os.path.join(results_dir, f"summary_{sanitize_repo_id(model_name)}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Summary saved to {summary_file}")
    
    return df


# =========================================================
# Model Loading
# =========================================================
def _is_mxfp4(model_path: str) -> bool:
    cfgs = ["quantization_config.json", "config.json"]
    for fn in cfgs:
        p = Path(model_path) / fn
        if p.is_file():
            try:
                j = json.loads(p.read_text())
                q = j.get("quantization_config", j)
                t = str(q.get("_type") or q.get("type") or q.get("quant_method") or "").lower()
                if "mxfp4" in t:
                    return True
            except Exception:
                pass
    return False

def load_model(model_path, for_training=False, force_dtype=None, adapters_path=None):
    print(f"\n[MODEL] Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    compute_dtype = (force_dtype if force_dtype is not None
                     else (torch.bfloat16 if torch.cuda.is_available() else torch.float32))

    # MXFP4 loads via HF-native path; no BitsAndBytes
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=False,
        torch_dtype=compute_dtype,      # bf16 compute recommended
        device_map="auto",
        attn_implementation="eager",
    )

    # Disable KV cache during training to save memory
    if hasattr(base, "config"):
        base.config.use_cache = False
        if getattr(base, "generation_config", None) is not None:
            base.generation_config.use_cache = False

    if for_training:
        # ----- LoRA attach with discovered targets -----
        leaf_linear_names = discover_linear_leaf_names(base)
        print("[LoRA] Leaf Linear module names detected:", leaf_linear_names)

        preferred = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
                    "dense","fc1","fc2","w1","w2","w3"}
        targets = sorted(list(preferred.intersection(leaf_linear_names)))
        if not targets:
            # fall back to all detected linear leaves
            targets = leaf_linear_names

        print("[LoRA] Using target_modules:", targets)

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
            # modules_to_save=["lm_head"],  # optional
        )

        model = get_peft_model(base, lora_cfg)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()      # <— key line for LoRA + checkpointing

        # sanity check: we must have trainable params
        
        model.train()
    else:
        if adapters_path and os.path.isfile(os.path.join(adapters_path, "adapter_config.json")):
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, adapters_path)
        else:
            model = base
        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model, "config"):
            model.config.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model



def ensure_model_downloaded(model_id):
    """Ensure model is downloaded locally."""
    local_model_dir = os.path.join(models_dir, sanitize_repo_id(model_id))
    
    def has_required_files(path):
        if not os.path.isdir(path):
            return False
        has_config = os.path.isfile(os.path.join(path, "config.json"))
        has_weights = any(
            fn.endswith(".safetensors") or (fn.startswith("pytorch_model") and fn.endswith(".bin"))
            for fn in os.listdir(path)
        )
        has_tok = (
            os.path.isfile(os.path.join(path, "tokenizer.json")) or
            os.path.isfile(os.path.join(path, "tokenizer.model"))
        )
        return has_config and has_weights and has_tok
    
    if not has_required_files(local_model_dir):
        print(f"[DOWNLOAD] Fetching '{model_id}' into {local_model_dir}...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_model_dir,
            local_dir_use_symlinks=False,
            allow_patterns=None,
            ignore_patterns=None
        )
        print(f"[DOWNLOAD] Completed: {local_model_dir}")
    else:
        print(f"[CACHE] Using existing local model at {local_model_dir}")
    
    return local_model_dir


# =========================================================
# Main Execution
# =========================================================
def main():
    print("="*60)
    print("CHESS MODEL TRAINING & EVALUATION")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Base model: {args.model}")
    print(f"Dataset: {args.dataset}")
    eval_dtype = _str_to_dtype(args.eval_dtype)
    # Load and split data
    if args.mode in ["train", "both"]:
        train_data, val_data, test_data = load_data_splits(
            args.dataset,
            args.train_samples,
            args.val_samples,
            args.test_samples
        )
    else:
        # Just need test data for evaluation
        _, _, test_data = load_data_splits(
            args.dataset,
            0,
            0,
            args.eval_samples
        )
    
    # Training phase
    if args.mode in ["train", "both"]:
        # Download base model
        local_model_path = ensure_model_downloaded(args.model)
        tokenizer, model = load_model(local_model_path, for_training=True)

        # Train
        trained_model = train_model(train_data, val_data, local_model_path, tokenizer, model)

        # For "both" mode, reload the trained model for evaluation
        if args.mode == "both":
            print("\n[INFO] Reloading trained model for evaluation...")
            # Load base (quantized) and apply LoRA adapters from output_dir
            local_model_path = ensure_model_downloaded(args.model)
            tokenizer, model = load_model(
                local_model_path,
                for_training=False,
                force_dtype=eval_dtype,
                adapters_path=args.output_dir,     # ← key addition
            )
            model_name = f"finetuned_{sanitize_repo_id(args.model)}"

    # Evaluation phase
    if args.mode in ["eval", "both"]:
        if args.mode == "eval":
            if os.path.exists(args.output_dir) and os.path.isfile(
                os.path.join(args.output_dir, "adapter_config.json")  # ← checks for adapters
            ):
                print(f"[INFO] Evaluating fine-tuned adapters from {args.output_dir}")
                local_model_path = ensure_model_downloaded(args.model)
                tokenizer, model = load_model(
                    local_model_path,
                    for_training=False,
                    force_dtype=eval_dtype,
                    adapters_path=args.output_dir,   # ← load adapters for eval
                )
                model_name = f"finetuned_{sanitize_repo_id(args.model)}"
            else:
                print(f"[INFO] Evaluating base model: {args.model}")
                local_model_path = ensure_model_downloaded(args.model)
                tokenizer, model = load_model(
                    local_model_path,
                    for_training=False,
                    force_dtype=eval_dtype,
                )
                model_name = sanitize_repo_id(args.model)

        # Evaluate
        results_df = evaluate_model(test_data, tokenizer, model, model_name)
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
