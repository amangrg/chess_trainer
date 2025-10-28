#!/usr/bin/env python
import os, argparse, json, time, re
import zstandard as zstd
import pandas as pd
from tqdm import tqdm
import chess
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from huggingface_hub import snapshot_download


# =========================================================
# CLI
# =========================================================
parser = argparse.ArgumentParser(description="Evaluate LLM chess move quality.")
parser.add_argument("--dataset", required=True, help="Path to lichess_db_eval.jsonl.zst")
parser.add_argument("--model", required=True, help="Hugging Face model name (e.g., mistralai/Mistral-7B-Instruct-v0.3)")
parser.add_argument("--samples", type=int, default=100)
args = parser.parse_args()


# =========================================================
# Paths
# =========================================================
repo_root   = os.path.dirname(__file__)
models_dir  = os.path.join(repo_root, "models")
results_dir = os.path.join(repo_root, "results")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "_").replace(":", "_")

local_model_dir = os.path.join(models_dir, sanitize_repo_id(args.model))

print(f"[INIT] Repository root: {repo_root}")
print(f"[INIT] Models directory: {models_dir}")
print(f"[INIT] Results directory: {results_dir}")
print(f"[INIT] Target model folder: {local_model_dir}")


# =========================================================
# Ensure model is present in ./models/<model_name>
# =========================================================
def _has_required_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    # minimal set to consider it usable
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    # weights may be sharded; accept any *.safetensors or pytorch_model*.bin
    has_weights = any(
        fn.endswith(".safetensors") or fn.startswith("pytorch_model") and fn.endswith(".bin")
        for fn in os.listdir(path)
    )
    # tokenizer files
    has_tok = (
        os.path.isfile(os.path.join(path, "tokenizer.json")) or
        os.path.isfile(os.path.join(path, "tokenizer.model"))  # SentencePiece
    )
    return has_config and has_weights and has_tok

if not _has_required_files(local_model_dir):
    print(f"[DOWNLOAD] Fetching '{args.model}' into {local_model_dir} ...")
    # Download straight into local_model_dir; avoid symlinks to a cache
    # (This still uses a temp cache during download, but files end up in local_model_dir)
    snapshot_download(
        repo_id=args.model,
        local_dir=local_model_dir,
        local_dir_use_symlinks=False,
        allow_patterns=None,  # all files
        ignore_patterns=None  # keep all
    )
    print(f"[DOWNLOAD] Completed: {local_model_dir}")
else:
    print(f"[CACHE] Using existing local model at {local_model_dir}")


# =========================================================
# Load model *from the local folder*
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[MODEL] Loading from {local_model_dir} on {device} ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    trust_remote_code=True,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
model.eval()
print(f"[MODEL] Loaded successfully in {time.time()-t0:.1f}s")


# =========================================================
# Dataset utilities (lichess evals schema)
# =========================================================
def extract_top_moves(obj, max_moves=5):
    """Extract top SAN moves from evals->pvs->line (first UCI of each PV), using the FEN."""
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


def sample_positions(path, n=100):
    """Stream compressed JSONL and sample first n valid positions."""
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
                            print(f"[DATA] Reached {n} samples after {count} chunks ({valid} valid)")
                            return samples
                    except Exception:
                        continue
    print(f"[DATA] Collected {len(samples)} total valid positions")
    return samples

positions = sample_positions(args.dataset, args.samples)
print(f"[DATA] Loaded {len(positions)} positions for evaluation.")

SAN_RE = re.compile(r"\b(O-O-O|O-O|[KQRNB]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRNB])?[+#]?)\b")

class StopAfterBestMove(StoppingCriteria):
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


def query_model(fen, idx=None):
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
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]


    with torch.no_grad():
        stopping_criteria = StoppingCriteriaList([StopAfterBestMove(tokenizer, input_len)])
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,                 # greedy
            no_repeat_ngram_size=5,          # curb “Answer with only …” loops
            repetition_penalty=1.1,          # mild, optional
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            stopping_criteria=stopping_criteria,
        )

    new_tokens = output_ids[0, input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # --- Parse after "Best Move:" ---
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

    print(f"[MODEL] #{idx}: SAN='{san}'")
    print(f"[REPLY]\n{completion}\n")
    if not san:
        print("[WARN] Could not parse a valid SAN after 'Best Move:'.")
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
print(f"[EVAL] Starting evaluation over {len(positions)} samples...")
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
outfile = os.path.join(results_dir, f"evaluate_{sanitize_repo_id(args.model)}.csv")
pd.DataFrame(records).to_csv(outfile, index=False)
print(f"[SAVE] ✅ Results written to {outfile}")
