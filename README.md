# chess_trainer
A pipeline to train chess playing llm agents


Steps:

1. Download dataset
wget https://database.lichess.org/lichess_db_eval.jsonl.zst

Use: 

module load anaconda3

conda activate /storage/ice1/9/7/agarg78/conda_envs/chess_eval

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --samples 100
