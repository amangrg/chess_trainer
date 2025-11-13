import os
import sys
import argparse
import json
import time
import re
import zstandard as zstd
import pandas as pd
from tqdm import tqdm
import signal

import chess
import torch
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

# ---------------------------------------------------------
# Custom Trainer that skips saving optimizer/scheduler state
# ---------------------------------------------------------
class NoOptimTrainer(Trainer):
    """A Trainer subclass that does not save the optimizer or scheduler state.

    This reduces checkpoint size and avoids disk quota errors when writing
    optimizer state. Useful when you only need the model weights for
    inference/fine‑tuning and not to resume the optimizer.
    """

    def _save_optimizer_and_scheduler(self, output_dir: str) -> None:  # type: ignore[override]
        # Override to do nothing (do not save optimizer or scheduler state)
        return


# =========================================================
# CLI Arguments
# =========================================================
parser = argparse.ArgumentParser(description="Fine‑tune and evaluate a chess model")
parser.add_argument("--dataset", required=True, help="Path to lichess_db_eval.jsonl.zst")
parser.add_argument("--model", required=True, help="Hugging Face model name (e.g. mistralai/Mistral-7B-Instruct-v0.3)")
parser.add_argument(
    "--mode",
    choices=["train", "eval", "both"],
    default="both",
    help="Run training, evaluation, or both",
)
parser.add_argument(
    "--output_dir",
    default="./finetuned_model",
    help="Output directory for trained model and checkpoints",
)
parser.add_argument(
    "--eval_samples",
    type=int,
    default=100,
    help="Number of samples for evaluation when mode=='eval'",
)

# Training hyperparameters
parser.add_argument("--train_samples", type=int, default=5000, help="Number of training samples")
parser.add_argument("--val_samples", type=int, default=500, help="Number of validation samples")
parser.add_argument("--test_samples", type=int, default=500, help="Number of test samples")
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size per device")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
parser.add_argument(
    "--eval_dtype",
    choices=["fp16", "bf16", "fp32"],
    default="fp16",
    help="dtype for evaluation (fp16 recommended for many HF LLMs)",
)

# Checkpointing / resume
parser.add_argument(
    "--save_strategy",
    choices=["no", "steps", "epoch"],
    default="steps",
    help="Strategy to save checkpoints during training",
)
parser.add_argument(
    "--save_steps",
    type=int,
    default=500,
    help="Save a checkpoint every N steps when save_strategy='steps'",
)
parser.add_argument(
    "--save_total_limit",
    type=int,
    default=2,
    help="Maximum number of checkpoints to keep (older checkpoints are removed)",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume training from the last checkpoint found in output_dir",
)

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
    """Convert repo ID to a valid directory name."""
    return repo_id.replace("/", "_").replace(":", "_")


def _str_to_dtype(name: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def extract_best_move(obj):
    """Extract the top‑rated move from evaluation data."""
    fen = obj.get("fen")
    if not fen:
        return None, None
    board = chess.Board(fen)
    # Get the best move from the first principal variation (PV)
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
    # Deduplicate while preserving order
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
    total_needed = train_n + val_n + test_n
    print(f"[DATA] Loading {total_needed} total samples from {path}")
    print(f"[DATA] Split: {train_n} train, {val_n} val, {test_n} test")
    all_samples = []
    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)
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
                                "top_moves": top_moves,
                            })
                        if len(all_samples) >= total_needed:
                            break
                    except Exception:
                        continue
                if len(all_samples) >= total_needed:
                    break
    print(f"[DATA] Loaded {len(all_samples)} total samples")
    train_data = all_samples[:train_n]
    val_data = all_samples[train_n : train_n + val_n]
    test_data = all_samples[train_n + val_n : train_n + val_n + test_n]
    print(f"[DATA] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data


# =========================================================
# Training Functions
# =========================================================
def format_training_example(fen, best_move, tokenizer):
    """Format a single training example as a chat conversation."""
    board = chess.Board(fen)
    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    board_display = board.unicode(empty_square="·", borders=True)
    # Compose a chat: system instructs the assistant to return only the best move in SAN,
    # user provides the board, and assistant returns the SAN move.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a chess assistant. Given a chess position and side to move, "
                "respond with the best legal move in standard algebraic notation (SAN). "
                "Do not provide any reasoning."
            ),
        },
        {
            "role": "user",
            "content": f"Side to move: {side_to_move}\nBoard:\n{board_display}",
        },
        {
            "role": "assistant",
            "content": best_move,
        },
    ]
    # Prefer to use chat templates when supported. Some tokenizers expose
    # apply_chat_template() but do not define a chat_template, which causes
    # ValueError on invocation. We defensively fall back to a simple
    # instruct-style prompt in that case.
    use_chat_template = False
    if hasattr(tokenizer, "apply_chat_template"):
        # Only attempt if a chat template is defined and non-empty
        chat_tmpl = getattr(tokenizer, "chat_template", None)
        if chat_tmpl:
            use_chat_template = True
    if use_chat_template:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            # Something went wrong using the template; fall back
            use_chat_template = False
    if not use_chat_template:
        # Fallback to instruct-style plain text
        prompt = (
            "You are a chess assistant. Given a chess position and side to move, "
            "respond with the best legal move in SAN.\n\n"
            f"Side to move: {side_to_move}\nBoard:\n{board_display}\n"
            "Answer:\n"
            f"{best_move}"
        )
        text = prompt
    return text


def create_dataset(data, tokenizer, max_length):
    """Convert raw chess data to a tokenized HuggingFace Dataset."""
    texts = [format_training_example(item["fen"], item["best_move"], tokenizer) for item in tqdm(data, desc="Formatting")]
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
        desc="Tokenizing",
    )
    return tokenized


def train_model(train_data, val_data, model_path, tokenizer, model):
    """Fine‑tune the model on the training data and return the trained model."""
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    train_dataset = create_dataset(train_data, tokenizer, args.max_length)
    val_dataset = create_dataset(val_data, tokenizer, args.max_length)
    print(f"[TRAIN] Training samples: {len(train_dataset)}")
    print(f"[TRAIN] Validation samples: {len(val_dataset)}")
    # Configure training arguments with checkpointing
    # Construct TrainingArguments without passing evaluation_strategy or eval_steps for
    # compatibility with older transformers versions. Evaluation will be handled
    # separately after training.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        logging_steps=max(50, args.save_steps // 5),
        # We do not pass `evaluation_strategy` or `eval_steps` to support
        # transformers versions that do not accept these parameters. Evaluation
        # will be performed after training by a custom function.
        save_strategy=args.save_strategy,
        save_steps=(args.save_steps if args.save_strategy == "steps" else None),
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        bf16=True,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        report_to="none",
        overwrite_output_dir=True,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    # Use NoOptimTrainer to skip saving optimizer/scheduler state
    trainer = NoOptimTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    # Preemption‑friendly signal handler
    def _save_and_exit(signum, frame):
        try:
            print(f"[SIGNAL] Caught signal {signum}. Saving emergency checkpoint...")
            emergency_dir = os.path.join(args.output_dir, "checkpoint-preempt")
            os.makedirs(emergency_dir, exist_ok=True)
            trainer.save_model(emergency_dir)
            trainer.save_state()
            print(f"[SIGNAL] Emergency checkpoint saved to {emergency_dir}")
        except Exception as e:
            print(f"[SIGNAL] Failed to save emergency checkpoint: {e}")
        finally:
            sys.exit(0)
    signal.signal(signal.SIGTERM, _save_and_exit)
    signal.signal(signal.SIGUSR1, _save_and_exit)
    # Resume if requested. Only resume when a valid trainer_state.json exists in the latest checkpoint.
    last_ckpt = None
    if args.resume and os.path.isdir(args.output_dir):
        ckpt = get_last_checkpoint(args.output_dir)
        if ckpt and os.path.isfile(os.path.join(ckpt, "trainer_state.json")):
            last_ckpt = ckpt
            print(f"[TRAIN] Resuming from checkpoint: {last_ckpt}")
        elif ckpt:
            print(
                f"[WARN] Found checkpoint '{ckpt}' but no trainer_state.json; ignoring and starting from scratch."
            )
    # Train
    print("[TRAIN] Starting fine‑tuning...")
    start_time = time.time()
    trainer.train(resume_from_checkpoint=last_ckpt)
    training_time = time.time() - start_time
    print(f"[TRAIN] Training completed in {training_time / 60:.2f} minutes")
    # Save final model and tokenizer
    print(f"[TRAIN] Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
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
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
    }
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("[TRAIN] Training info saved")
    return trainer.model


# =========================================================
# Evaluation Functions
# =========================================================
SAN_RE = re.compile(r"\b(O-O-O|O-O|[KQRNB]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRNB])?[+#]?)\b")


class StopAfterSAN(StoppingCriteria):
    """Stop generation after detecting any SAN in the newly generated text."""
    def __init__(self, tokenizer, start_idx):
        self.tok = tokenizer
        self.start_idx = start_idx
    def __call__(self, input_ids, scores, **kwargs):
        text = self.tok.decode(input_ids[0, self.start_idx :], skip_special_tokens=True)
        # Stop generation as soon as a SAN move appears in the output.
        return SAN_RE.search(text) is not None


def query_model(fen, tokenizer, model, idx=None):
    """Query the model for a best move and return (san_move, full_reply)."""
    board = chess.Board(fen)
    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    board_display = board.unicode(empty_square="·", borders=True)
    # Compose a chat conversation: system instructs the assistant to return only the best move
    # in SAN, and user provides the board. At inference, we don't include the assistant's reply.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a chess assistant. Given a chess position and side to move, "
                "respond with the best legal move in standard algebraic notation (SAN). "
                "Do not provide any reasoning."
            ),
        },
        {
            "role": "user",
            "content": f"Side to move: {side_to_move}\nBoard:\n{board_display}",
        },
    ]
    # Build the prompt. Prefer chat template when both apply_chat_template and a template exist.
    use_chat_template = False
    if hasattr(tokenizer, "apply_chat_template"):
        chat_tmpl = getattr(tokenizer, "chat_template", None)
        if chat_tmpl:
            use_chat_template = True
    if use_chat_template:
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            use_chat_template = False
    if not use_chat_template:
        prompt_text = (
            "You are a chess assistant. Given a chess position and side to move, "
            "respond with the best legal move in SAN.\n\n"
            f"Side to move: {side_to_move}\nBoard:\n{board_display}\n"
            "Answer:\n"
        )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        stopping_criteria = StoppingCriteriaList([StopAfterSAN(tokenizer, input_len)])
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            no_repeat_ngram_size=5,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            stopping_criteria=stopping_criteria,
        )
    new_tokens = output_ids[0, input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Extract the first SAN move from the completion
    san = ""
    m = SAN_RE.search(completion)
    cand = m.group(1) if m else ""
    if cand:
        cleaned = re.sub(r"[+#]+$", "", cand.strip())
        try:
            mv = board.parse_san(cleaned)
            san = board.san(mv)
        except Exception:
            # Try to match ignoring check or mate symbols
            for move in board.legal_moves:
                if (
                    board.san(move)
                    .replace("+", "")
                    .replace("#", "")
                    .lower()
                    == cleaned.lower()
                ):
                    san = board.san(move)
                    break
    if idx is not None:
        print(f"[EVAL] #{idx}: SAN='{san}'")
        if not san:
            print("[WARN] Could not parse a valid SAN in the model's reply.")
    return san, completion


def evaluate_move(fen, move_san, top_moves_san):
    """Check if the predicted move is legal and if it's in the list of top moves."""
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
    """Evaluate the model on the test set and return a DataFrame of results."""
    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)
    print(f"[EVAL] Evaluating on {len(test_data)} test samples...")
    records = []
    for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        try:
            fen = sample["fen"]
            top_moves = sample["top_moves"]
            move_san, full_reply = query_model(fen, tokenizer, model, idx=i)
            legal, good = evaluate_move(fen, move_san, top_moves)
            records.append(
                {
                    "id": i,
                    "fen": fen,
                    "model_move": move_san,
                    "full_reply": full_reply,
                    "legal": legal,
                    "good": good,
                    "top_moves": ",".join(top_moves),
                }
            )
        except Exception as e:
            print(f"[WARN] Error on sample #{i}: {e}")
            continue
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
    # Save results and summary
    outfile = os.path.join(results_dir, f"evaluate_{sanitize_repo_id(model_name)}.csv")
    df.to_csv(outfile, index=False)
    print(f"[SAVE] Results saved to {outfile}")
    summary = {
        "model": model_name,
        "test_samples": len(records),
        "legal_move_rate": legal_rate,
        "top_move_rate": good_rate,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_file = os.path.join(results_dir, f"summary_{sanitize_repo_id(model_name)}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Summary saved to {summary_file}")
    return df


# =========================================================
# Model Loading
# =========================================================
def load_model(model_path, for_training=False, force_dtype=None):
    """Load the tokenizer and model from a local directory or Hugging Face repo."""
    print(f"\n[MODEL] Loading model from: {model_path}")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if force_dtype is not None:
        dtype = force_dtype
    else:
        dtype = torch.bfloat16 if for_training else (torch.float16 if device == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=False,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    # Ensure pad token is defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    # Disable KV cache explicitly
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    if not for_training:
        model.eval()
    print(f"[MODEL] Loaded in {time.time() - start:.1f}s | dtype={dtype}")
    return tokenizer, model


def ensure_model_downloaded(model_id):
    """
    Ensure the base model is downloaded locally to ``models_dir``.

    This implementation avoids ``snapshot_download`` (which can hit a bug when
    metadata files are missing) by leveraging the standard
    ``AutoTokenizer/AutoModelForCausalLM`` download mechanism with a
    specified ``cache_dir``. If the model directory already contains
    ``config.json``, at least one weight file, and a tokenizer file, it is
    assumed to be complete and reused. Otherwise the model and tokenizer
    are downloaded via ``from_pretrained`` into the cache directory and
    then saved into the ``local_model_dir`` for future runs.
    """
    local_model_dir = os.path.join(models_dir, sanitize_repo_id(model_id))

    def has_required_files(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        has_config = os.path.isfile(os.path.join(path, "config.json"))
        has_weights = any(
            fn.endswith(".safetensors") or (fn.startswith("pytorch_model") and fn.endswith(".bin"))
            for fn in os.listdir(path)
        )
        has_tok = (
            os.path.isfile(os.path.join(path, "tokenizer.json"))
            or os.path.isfile(os.path.join(path, "tokenizer.model"))
        )
        return has_config and has_weights and has_tok

    if has_required_files(local_model_dir):
        print(f"[CACHE] Using existing local model at {local_model_dir}")
        return local_model_dir

    # If not present, download using transformers API. We first load the
    # tokenizer and model with the cache_dir pointing to a temporary cache
    # location (within models_dir) so that huggingface does not re-download
    # from the internet if the files exist in the default hub cache. Once
    # loaded, we save the files into ``local_model_dir``.
    print(f"[DOWNLOAD] Fetching '{model_id}' into {local_model_dir}...")
    os.makedirs(local_model_dir, exist_ok=True)
    # Use a temporary cache dir under models_dir to isolate downloads
    tmp_cache_dir = os.path.join(models_dir, "hf_cache")
    os.makedirs(tmp_cache_dir, exist_ok=True)
    try:
        # Download tokenizer and model into tmp_cache_dir
        tmp_tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=False, cache_dir=tmp_cache_dir
        )
        tmp_model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=False, cache_dir=tmp_cache_dir
        )
        # Save the loaded model and tokenizer into local_model_dir
        tmp_model.save_pretrained(local_model_dir)
        tmp_tokenizer.save_pretrained(local_model_dir)
        print(f"[DOWNLOAD] Completed: {local_model_dir}")
    except Exception as e:
        # Clean up partially downloaded files if something went wrong
        print(f"[ERROR] Failed to download model '{model_id}': {e}")
        if os.path.isdir(local_model_dir):
            try:
                import shutil

                shutil.rmtree(local_model_dir)
            except Exception:
                pass
        raise
    return local_model_dir


# =========================================================
# Main Execution
# =========================================================
def main():
    print("=" * 60)
    print("CHESS MODEL TRAINING & EVALUATION")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Base model: {args.model}")
    print(f"Dataset: {args.dataset}")
    eval_dtype = _str_to_dtype(args.eval_dtype)
    # Load data
    if args.mode in ["train", "both"]:
        train_data, val_data, test_data = load_data_splits(args.dataset, args.train_samples, args.val_samples, args.test_samples)
    else:
        _, _, test_data = load_data_splits(args.dataset, 0, 0, args.eval_samples)
    # Training phase
    if args.mode in ["train", "both"]:
        local_model_path = ensure_model_downloaded(args.model)
        tokenizer, model = load_model(local_model_path, for_training=True)
        trained_model = train_model(train_data, val_data, local_model_path, tokenizer, model)
        if args.mode == "both":
            print("\n[INFO] Reloading trained model for evaluation...")
            tokenizer, model = load_model(args.output_dir, for_training=False, force_dtype=eval_dtype)
            model_name = f"finetuned_{sanitize_repo_id(args.model)}"
    # Evaluation phase
    if args.mode in ["eval", "both"]:
        if args.mode == "eval":
            if os.path.exists(args.output_dir) and os.path.isfile(os.path.join(args.output_dir, "config.json")):
                print(f"[INFO] Evaluating fine‑tuned model from {args.output_dir}")
                tokenizer, model = load_model(args.output_dir, for_training=False, force_dtype=eval_dtype)
                model_name = f"finetuned_{sanitize_repo_id(args.model)}"
            else:
                print(f"[INFO] Evaluating base model: {args.model}")
                local_model_path = ensure_model_downloaded(args.model)
                tokenizer, model = load_model(local_model_path, for_training=False, force_dtype=eval_dtype)
                model_name = sanitize_repo_id(args.model)
        # Evaluate
        results_df = evaluate_model(test_data, tokenizer, model, model_name)
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
