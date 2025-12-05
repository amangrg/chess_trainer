
A pipeline to train chess playing llm agents

Steps:

1. Download dataset
wget https://database.lichess.org/lichess_db_eval.jsonl.zst

Use: 

module load anaconda3

conda activate /storage/ice1/9/7/agarg78/conda_envs/chess_eval

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model openai/gpt-oss-20b --mode eval --eval_samples 100 --eval_dtype bf16

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model openai/gpt-oss-20b --mode train --train_samples 10000 --val_samples 500 --test_samples 0 --epochs 1 --batch_size 2  --max_length 512 --learning_rate 2e-5  --output_dir ./ft_model

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model openai/gpt-oss-20b --mode eval --eval_samples 100 --eval_dtype bf16 --output_dir ./ft_model


# chess trainer frontend 

Add your evaluation file named as "data.csv" in the chess_frontend folder

Run the following command locally: python3 -m http.server 8000 

Access the page at http://localhost:8000/chess_frontend/


# Chess-Playing LLM Assistant
**Optimizing Language Models for Strategic Reasoning**

Team: Aman Garg, Keller Smith, Shabib Ahmed, Tejas Ghone

---

## Overview

This project explores teaching Large Language Models to play chess through supervised fine-tuning with structured reasoning. Our approach enables LLMs to explain their strategic thinking while making moves, unlike traditional black-box chess engines.

**Key Results:** Fine-tuned GPT-OSS-20B achieves 60% legal move generation and 29.2% best move prediction, demonstrating that LLMs can acquire spatial and strategic reasoning capabilities.

---

## Motivation

**Why Chess for LLM Research?**

1. **Objective Testing:** Chess has a narrow zone of correct moves in most positions, making it easy to objectively test whether an LLM's reasoning is sound or just pattern-matching.

2. **Explainable Reasoning:** Unlike black-box models like AlphaZero, LLMs can articulate their reasoning, making chess ideal for studying whether language models develop genuine understanding.

3. **Novel Challenge:** Very few prior publications exist in this area, making it a frontier research topic.

---

## Key Problems Addressed

**Move Hallucination:** Models sometimes generate illegal moves that violate chess rules.

**Positional Understanding:** Evaluating complex board positions remains challenging for language models.

**Strategic Planning:** Limited ability to craft and follow long-term strategic plans.

**Tactical Calculation:** Inconsistent calculation of forcing sequences and combinations.

---

## Dataset

**Source:** Lichess Evaluation Database (https://database.lichess.org/#evals)

**Scale:** 289 million chess positions with Stockfish evaluations and best moves

**Schema:**
```json
{
  "fen": "position in FEN notation",
  "evals": [{
    "knodes": "nodes searched by engine",
    "depth": "search depth",
    "pvs": [{
      "cp": "centipawn evaluation",
      "mate": "mate evaluation (if applicable)",
      "line": "principal variation in UCI format"
    }]
  }]
}
```

---

## Pipeline Overview

**1. Data Preparation**
- Stream data chunks using z-standard compression
- Extract FEN strings, evaluations, candidate moves, and best moves
- Convert SAN formatted moves to UCI notation using python-chess library
- Create training/test datasets with Hugging Face tokenizer

**2. Supervised Fine-Tuning**
- Full parameter fine-tuning (not LoRA for best results)
- Format data into conversational examples
- Train model to predict structured responses with checkpointing

**3. Evaluation**
- Parse model responses to extract predicted moves
- Validate move legality using python-chess
- Compare with Stockfish best moves
- Generate CSV results and JSON summaries

**4. Website Integration**
- Interactive chess board interface for testing
- Compare user moves with model predictions
- Visualize model performance

---

## Fine-Tuning Strategies

### Strategy 1: Best Move Only (No Reasoning)

**Format:**
```
System: You are a chess assistant. Reply with the best move only.
User: Side to move: White/Black
      Board: <unicode board>
Assistant: Best Move: Nf3
```

**Results:** Improved from 10% to 60% legal move rate on GPT-OSS-20B, but lacks interpretability.

---

### Strategy 2: Simple Reasoning (15-word limit)

**Format:**
```
System: Reply with exactly two lines:
        My reasoning: <15 words max>
        Best Move: <legal SAN only>
```

**Problem:** Reasoning collapsed without expert examples; model learned to use generic phrases.

---

### Strategy 3: XML-Style Step-by-Step Reasoning (BEST APPROACH)

**Format:**
```xml
<board_position>the board position is FEN: {fen}</board_position>
<evaluate>{eval_bucket}</evaluate>
<perceive>{all_legal_moves_san}</perceive>
<predict>{top_moves_san}</predict>
<choose>the best move for {color} is {best_move}</choose>
```

**Results:**
- Model learns to evaluate positions (winning/losing/equal)
- Lists all legal moves reliably
- Identifies top candidate moves
- Most interpretable approach
- Occasionally struggles to output single move consistently in choose tag

---

## Prompting Evolution

**Zero-shot baseline:** Simple instruction led to illegal moves and no positional understanding.

**Few-shot with SAN examples:** Slight improvement but models overfit to examples.

**Few-shot with short reasoning:** Initial success, but reasoning collapsed over time.

**Few-shot XML structured reasoning:** Most successful format, model reliably outputs eval and legal moves.

**Research Support:** Structured prompts reduce hallucination (Google PaLM paper), XML-style prompting improves controllability (Anthropic Constitutional AI, 2023).

---

## System Requirements

### GPU Memory Requirements

| Model Size | Method | GPU Memory | Training Time (10k samples) |
|-----------|--------|------------|---------------------------|
| <5B params | Full fine-tuning | 1x H100 (80GB) | ~2 hours |
| ~20B params | Full fine-tuning | 2x H200 (141GB) | ~4 hours |
| ~20B params | LoRA | 1x H100 (80GB) | ~2 hours (poor results) |

**Note:** Full fine-tuning is required for chess reasoning. LoRA experiments showed that parameter-efficient methods are insufficient for this task.

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (H100 or H200 recommended for large models)
- Anaconda or Miniconda

### Environment Setup

```bash
# Load Anaconda module (if on HPC cluster)
module load anaconda3

# Create and activate conda environment
conda create -n chess_eval python=3.9
conda activate chess_eval
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: transformers, torch, python-chess, zstandard, pandas, accelerate

### Download Dataset

```bash
wget https://database.lichess.org/lichess_db_eval.jsonl.zst
mkdir -p data
mv lichess_db_eval.jsonl.zst data/
```

---

## Usage

### Evaluation Only

```bash
python evaluate.py \
  --dataset data/lichess_db_eval.jsonl.zst \
  --model openai/gpt-oss-20b \
  --mode eval \
  --eval_samples 100 \
  --eval_dtype bf16
```

### Training

```bash
python evaluate.py \
  --dataset data/lichess_db_eval.jsonl.zst \
  --model openai/gpt-oss-20b \
  --mode train \
  --train_samples 10000 \
  --val_samples 500 \
  --test_samples 0 \
  --epochs 1 \
  --batch_size 2 \
  --max_length 512 \
  --learning_rate 2e-5 \
  --output_dir ./ft_model
```

### Evaluate Fine-tuned Model

```bash
python evaluate.py \
  --dataset data/lichess_db_eval.jsonl.zst \
  --model openai/gpt-oss-20b \
  --mode eval \
  --eval_samples 100 \
  --eval_dtype bf16 \
  --output_dir ./ft_model
```

### Training + Evaluation

```bash
python evaluate.py \
  --dataset data/lichess_db_eval.jsonl.zst \
  --model openai/gpt-oss-20b \
  --mode both \
  --train_samples 10000 \
  --val_samples 500 \
  --test_samples 500 \
  --epochs 2 \
  --batch_size 2 \
  --learning_rate 2e-5 \
  --max_length 512 \
  --output_dir ./ft_model \
  --eval_samples 500 \
  --eval_dtype bf16
```

---

## Script Parameters

### Base Arguments
- `--dataset`: Path to zstandard-compressed training data
- `--model`: Hugging Face model identifier (e.g., "openai/gpt-oss-20b")
- `--mode`: Operation mode - "train", "eval", or "both"
- `--output_dir`: Directory for fine-tuned model (default: ./finetuned_model)
- `--eval_samples`: Number of positions to evaluate (default: 100)

### Data Split
- `--train_samples`: Positions for training (default: 5000)
- `--val_samples`: Positions for validation (default: 500)
- `--test_samples`: Positions for testing (default: 500)

### Hyperparameters
- `--epochs`: Complete passes through training data (default: 2)
- `--batch_size`: Samples per training step (default: 8)
- `--learning_rate`: Optimization step size (default: 2e-5)
- `--max_length`: Maximum token sequence length (default: 512)
- `--eval_dtype`: Data type for inference - "fp16", "bf16", or "fp32"

---

## Chess Frontend

### Setup

```bash
# Copy evaluation CSV to frontend folder
cp evaluation_results.csv chess_frontend/data.csv

# Start local server
cd chess_frontend
python3 -m http.server 8000
```

### Access
Open browser to: http://localhost:8000

**Features:**
- Load positions from FEN strings
- Interactive chessboard
- Compare your moves with model predictions
- View model's reasoning process
- Track performance across positions

---

## Evaluation Metrics

### Move Legality
Percentage of generated moves that obey chess rules. Target: ≥95% legality with robust handling of edge cases (castling, promotion, en passant).

### Move Quality
- Best Move Rate: Percentage matching Stockfish's top move
- Top-N Move Rate: Percentage in Stockfish's top N moves
- Correlation with Stockfish centipawn evaluation

### Strategic Performance
Rating progression during evaluation matches, win/loss ratio, and consistency across different position types.

---

## Results

### Performance Across Models
*Evaluation on 500 testing samples (Fine-tuned on 10,000 samples)*

| Model | Few-shot Legal | Few-shot Best | Fine-tuned Legal | Fine-tuned Best |
|-------|---------------|---------------|------------------|-----------------|
| GPT-OSS-20B | 35 (7.0%) | 2 (0.4%) | **300 (60.0%)** | **146 (29.2%)** |
| Mistral-7B | 34 (6.8%) | 3 (0.6%) | 272 (54.4%) | 57 (11.4%) |
| Qwen2.5-7B | 29 (5.8%) | 1 (0.2%) | 193 (38.6%) | 93 (18.6%) |
| Qwen2.5-3B | 27 (5.4%) | 1 (0.2%) | 244 (48.8%) | 32 (6.4%) |
| Deepseek-7B | 14 (2.8%) | 0 (0.0%) | 214 (42.8%) | 19 (3.8%) |
| Phi-3-mini-4k | 18 (3.6%) | 7 (1.4%) | 153 (30.6%) | 69 (13.8%) |

**Key Findings:**
- GPT-OSS-20B achieved best overall performance
- Fine-tuning dramatically improved both legality and best move rates
- All models showed 5-10x improvement in legal move generation
- Larger models benefited more from fine-tuning

### Accuracy vs Training Set Size

Models saturate quickly, showing diminishing returns on training data after initial gains. Achieving high "Best Move" percentage is challenging because models plateau after 20k-30k training samples.

---

## LoRA Experiments

**Advantages:**
- Much lower GPU memory usage
- Faster training (50k samples in ~12 hours)
- Smaller checkpoints

**Results with GPT-OSS-20B:**
- LoRA with 50k samples, 256 char reply: 0% legal moves
- LoRA with 50k samples, 1000 char reply: 20% legal moves

**Conclusion:** LoRA results were unsuccessful for chess reasoning. Outputs were lengthy and didn't converge on single moves. Full fine-tuning is required for this task.

---

## Project Structure

```
chess_trainer/
├── evaluate.py              # Main training and evaluation script
├── requirements.txt         # Python dependencies
├── data/
│   └── lichess_db_eval.jsonl.zst
├── ft_model/               # Fine-tuned model output
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files
├── results/
│   ├── evaluation_results.csv
│   └── summary.json
├── chess_frontend/
│   ├── index.html
│   ├── chess.js
│   ├── styles.css
│   └── data.csv
└── README.md
```

---

## Challenges & Future Work

### Current Challenges

**Computational Cost:** RL training on millions of positions requires substantial GPU resources.

**Generalization:** Generalizing to unseen tactics and long-term strategies remains difficult.

**Reasoning Collapse:** Complex objectives and optimizations like LoRA reduce reasoning quality.

**Single Move Selection:** XML structured format sometimes produces multiple moves in choose block.

### Next Steps

**Self-Play Reinforcement Learning:** Implement self-play RL to improve beyond supervised learning.

**GRPO (Generalized Reward Policy Optimization):** Use GRPO to encourage high-confidence single answers in choose block.

**Expert Annotations:** Incorporate expert chess commentary to improve reasoning quality and explainability.

**Stronger Evaluation:** Test against strong chess engines (Stockfish, Leela Chess Zero) and human players of various skill levels.

**Transfer to Other Games:** Apply approach to other strategic board games (Go, Shogi, Xiangqi).

---

## Key Contributions

1. First comprehensive study of LLM chess playing with explainable reasoning
2. XML-structured prompting enables models to output evaluation and legal moves reliably
3. Demonstrated that full fine-tuning is necessary; LoRA insufficient for complex reasoning
4. Achieved 60% legal move rate and 29% best move rate on GPT-OSS-20B
5. Interactive platform for comparing human vs. model chess decisions

---

## Impact

**Educational:** Chess serves as stress test for general intelligence

**Research:** Bridges natural language understanding and complex game play

**AI Interpretability:** Models can articulate their strategic thinking

**Benchmark:** Provides objective measure of LLM reasoning capabilities

---

## References

1. Xiangqi-R1: Enhancing Spatial Strategic Reasoning in LLMs for Chinese Chess via Reinforcement Learning
2. Complete Chess Games Enable LLM Become A Chess Master
3. [1712.01815] Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero)
4. [2501.12948] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
5. [2205.11916] Large Language Models are Zero-Shot Reasoners
6. [2201.11903] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

---

## Team Contributions

**Aman Garg:** Model downloading and evaluation script, dataset parsing, reasoning experiments

**Shabib Ahmed:** Fine-tuning script development, model fine-tuning execution, general system improvements

**Keller Smith:** LoRA experiments, model fine-tuning execution, performance optimization

**Tejas Ghone:** Frontend web application, model fine-tuning execution, UI/UX design and integration

---

## Citation

```bibtex
@misc{chess_llm_2025,
  title={Chess-Playing LLM Assistant: Optimizing Language Models for Strategic Reasoning},
  author={Garg, Aman and Smith, Keller and Ahmed, Shabib and Ghone, Tejas},
  year={2025},
  howpublished={GitHub repository},
  url={https://github.com/amangrg/chess_trainer}
}
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- Lichess for providing the comprehensive chess evaluation database
- Hugging Face for model hosting and transformers library
- Stockfish for ground truth evaluations
- Our institution for providing GPU computational resources

---

## Contact

**GitHub Issues:** https://github.com/amangrg/chess_trainer/issues

**Project Link:** https://github.com/amangrg/chess_trainer
