# chess_trainer
A pipeline to train chess playing llm agents


Steps:

1. Download dataset
wget https://database.lichess.org/lichess_db_eval.jsonl.zst

Use: 

module load anaconda3

conda activate /storage/ice1/9/7/agarg78/conda_envs/chess_eval

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --mode eval --eval_samples 100

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --mode train --train_samples 5000 --val_samples 500 --test_samples 0 --epochs 1 --batch_size 1  --max_length 256 --learning_rate 2e-5  --output_dir ./ft_model

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --mode eval --eval_samples 200 --output_dir ./ft_model