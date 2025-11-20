# chess_trainer
A pipeline to train chess playing llm agents


Steps:

1. Download dataset
wget https://database.lichess.org/lichess_db_eval.jsonl.zst

Use: 

module load anaconda3

conda activate /storage/ice1/9/7/agarg78/conda_envs/chess_eval

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model openai/gpt-oss-20b --mode eval --eval_samples 100 --eval_dtype bf16

New Full Fine Tune - 

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model openai/gpt-oss-20b --mode train --train_samples 10000 --val_samples 500 --test_samples 500 --epochs 1 --batch_size 4 --learning_rate 2e-5 --max_length 512 --eval_dtype bf16 --output_dir ./ft_model_full

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model openai/gpt-oss-20b --mode eval --eval_samples 10 --eval_dtype bf16 --eval_dir ./ft_model_full_final

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model ./ft_model_full --mode train --train_samples 5000 --val_samples 100 --test_samples 100 --epochs 1 --batch_size 32 --learning_rate 2e-5 --max_length 512 --eval_dtype bf16 --output_dir ./ft_model_full_continued

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model ./ft_model_full_continued --mode train --train_samples 5000 --val_samples 100 --test_samples 100 --epochs 1 --batch_size 32 --learning_rate 2e-5 --max_length 512 --eval_dtype bf16 --output_dir ./ft_model_full_final

LoRA - 

python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model openai/gpt-oss-20b --mode train --train_samples 100000 --val_samples 500 --test_samples 0 --epochs 1 --batch_size 16 --max_length 512 --learning_rate 2e-5 --output_dir ./ft_model_modified

# LoRA eval (XML format)
python evaluate.py --dataset data/lichess_db_eval.jsonl.zst --model openai/gpt-oss-20b --mode eval --eval_samples 100 --eval_dtype bf16 --eval_dir ./ft_model_modified

