#!/usr/bin/env python
import os, argparse, json, random, time
import zstandard as zstd
import pandas as pd
from tqdm import tqdm
import chess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# =========================================================
# Setup repo-local directories and cache
# =========================================================
repo_root = os.path.dirname(__file__)
os.environ["HF_HOME"] = os.path.join(repo_root, "models")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(repo_root, "models")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

results_dir = os.path.join(repo_root, "results")
os.makedirs(results_dir, exist_ok=True)

print(f"[INIT] Repository root: {repo_root}")
print(f"[INIT] Models directory: {os.environ['HF_HOME']}")
print(f"[INIT] Results directory: {results_dir}")


# =========================================================
# Parse CLI
# =========================================================
parser = argparse.ArgumentParser(description="Evaluate LLM chess move quality.")
parser.add_argument("--dataset", required=True, help="Path to lichess_db_eval.jsonl.zst")
parser.add_argument("--model", required=True, help="Hugging Face model name")
parser.add_argument("--samples", type=int, default=100)
args = parser.parse_args()

print(f"[ARGS] Dataset: {args.dataset}")
print(f"[ARGS] Model: {args.model}")
print(f"[ARGS] Samples: {args.samples}")


# =========================================================
# Load model
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[MODEL] Loading {args.model} on {device} ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
print(f"[MODEL] Loaded successfully in {time.time()-t0:.1f}s")


# =========================================================
# Dataset utilities
# =========================================================
def extract_top_moves(obj, max_moves=3):
    """Extract top SAN moves from evals->pvs->line"""
    fen = obj.get("fen")
    if not fen:
        return []
    board = chess.Board(fen)
    moves = []
    for eval_entry in obj.get("evals", []):
        for pv in eval_entry.get("pvs", []):
            line = pv.get("line")
            if line:
                first_move = line.split()[0]
                try:
                    move = chess.Move.from_uci(first_move)
                    moves.append(board.san(move))
                except Exception:
                    continue
    # Deduplicate preserving order
    seen, uniq = set(), []
    for m in moves:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq[:max_moves]


def sample_positions(path, n=100):
    """Stream compressed JSONL and sample first n valid positions"""
    print(f"[DATA] Sampling up to {n} positions from {path} ...")
    samples = []
    count, valid = 0, 0
    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(f) as reader:
            buffer = b""
            for chunk in iter(lambda: reader.read(16384), b""):
                count += 1
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    try:
                        obj = json.loads(line)
                        fen = obj.get("fen")
                        top_moves = extract_top_moves(obj)
                        if fen and top_moves:
                            valid += 1
                            samples.append((fen, top_moves))
                        if len(samples) >= n:
                            print(f"[DATA] Reached {n} samples after reading {count} chunks ({valid} valid)")
                            return samples
                    except Exception:
                        continue
    print(f"[DATA] Collected {len(samples)} total valid positions")
    return samples


positions = sample_positions(args.dataset, args.samples)
print(f"[DATA] Loaded {len(positions)} positions for evaluation.")


# =========================================================
# Model querying and evaluation
# =========================================================


# Compile once
UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)   # e2e4, e7e8q
SAN_RE = re.compile(
    r"\b(O-O-O|O-O|[KQRNB]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRNB])?[+#]?)\b"
)  # SAN like Nf3, exd5, O-O, e8=Q+

def _to_san(board, maybe_move):
    """Try UCI first, else parse SAN; return SAN or None."""
    # UCI → SAN
    try:
        m = chess.Move.from_uci(maybe_move.lower())
        if m in board.legal_moves:
            return board.san(m)
    except Exception:
        pass
    # SAN direct
    try:
        m = board.parse_san(maybe_move)
        return board.san(m)
    except Exception:
        return None

def query_model(fen, idx=None):
    """Query model and return (move_san, full_reply)."""
    board = chess.Board(fen)

    user_msg = (
        "You are a chess grandmaster. Given the FEN below, output ONLY the best move "
        "in UCI notation (like e2e4, g1f3, e7e8q). No explanations, one move only."
        f"\nFEN: {fen}\nBest move:"
    )

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = user_msg

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0,
            pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )

    new_tokens = output_ids[0, input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Extract possible move (UCI or SAN)
    import re
    uci_match = re.search(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", completion)
    move = uci_match.group(1) if uci_match else completion.split()[-1]

    # Convert to SAN for consistency
    try:
        san = board.san(chess.Move.from_uci(move))
    except Exception:
        san = move

    # Log clearly
    print(f"[MODEL] #{idx}: FEN={fen[:25]}... → {san}")
    print(f"[FULL REPLY] {completion}\n")

    return san, completion


def evaluate_move(fen, move_san, top_moves_san):
    board = chess.Board(fen)
    legal = False
    try:
        parsed = board.parse_san(move_san)
        legal = parsed in board.legal_moves
    except Exception:
        legal = False
    good = move_san in top_moves_san
    return legal, good


# =========================================================
# Main loop
# =========================================================
records = []
for i, (fen, top_moves) in enumerate(tqdm(positions, desc="Evaluating")):
    try:
        move_san, full_reply = query_model(fen, idx=i)
        legal, good = evaluate_move(fen, move_san, top_moves)
        records.append({
            "id": i,
            "fen": fen,
            "model_move": move_san,
            "full_reply": full_reply,
            "legal": legal,
            "good": good,
            "top_moves": top_moves
        })
    except Exception as e:
        print(f"[WARN] Error on sample #{i}: {e}")
        continue

print(f"[EVAL] Completed {len(records)} evaluations.")


# =========================================================
# Save results
# =========================================================
outfile = os.path.join(results_dir, f"evaluate_{args.model.replace('/', '_')}.csv")
pd.DataFrame(records).to_csv(outfile, index=False)
print(f"[SAVE] ✅ Results written to {outfile}")
