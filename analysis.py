import csv

# Change this to your CSV filename
csv_file = 'evaluate_Qwen_Qwen2.5-7B-Instruct prompt 2.csv'

total_moves = 0
legal_moves = 0
good_moves = 0

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        total_moves += 1
        
        # Convert string 'True'/'False' to boolean
        is_legal = row['legal'].strip().lower() == 'true'
        is_good = row['good'].strip().lower() == 'true'
        
        if is_legal:
            legal_moves += 1
        if is_good:
            good_moves += 1

# Calculate percentages
legal_percentage = (legal_moves / total_moves * 100) if total_moves > 0 else 0
good_percentage = (good_moves / total_moves * 100) if total_moves > 0 else 0

# Print results
print(f"Total moves: {total_moves}")
print(f"Legal moves: {legal_moves} ({legal_percentage:.2f}%)")
print(f"Good moves: {good_moves} ({good_percentage:.2f}%)")